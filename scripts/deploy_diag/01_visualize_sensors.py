"""Stage S3 / Phase B — sensors-only live mosaic.

Opens every discovered RealSense + GelSight at sensible defaults
(no cfg required), renders a mosaic for `--duration` seconds, prints
per-stream fps + a histogram of frame intervals at the end. The robot
stays powered OFF — this script never talks to the controller.

Discovery-driven: there's no per-camera "role" name (platform/hand)
here because no task is loaded; tiles are labelled by device serial /
slot. To check that the role-vs-serial mapping in your task matches
the rig, use `00_inventory.py --task <name>` first.

Usage:
    python -m scripts.deploy_diag.01_visualize_sensors --duration 30

    # Cap RealSense fps if your USB bus can't sustain the full set:
    python -m scripts.deploy_diag.01_visualize_sensors --rs-fps 10 --duration 30

Go/no-go criteria are listed in scripts/deploy_diag/SENSOR_DEBUG_PLAN.md.
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from trainflow.env.ur3_bir.discover import (
    detect_realsense_cameras,
    discover_gelsight_devices,
    gelsight_serial_from_path,
)
from trainflow.env.ur3_bir.sensor_clients import GelsightClient, RealsenseClient


def _build_clients(rs_width: int, rs_height: int, rs_fps: int,
                   gs_width: int, gs_height: int, gs_fps: int) -> dict:
    """Open every discovered RealSense + GelSight at the given defaults.
    Returns {label: client}. Errors are recorded as `producer_errors`
    on each client but never raised — one missing device shouldn't
    block the rest of the mosaic."""
    clients: dict[str, object] = {}

    rs_devices = detect_realsense_cameras()
    print(f"[realsense] discovered {len(rs_devices)} device(s):")
    for c in rs_devices:
        print(f"  serial={c['serial']}  name={c['name']}")
        cfg = OmegaConf.create({
            "serial": c["serial"],
            "width": rs_width, "height": rs_height, "fps": rs_fps,
        })
        cli = RealsenseClient(cfg)
        try:
            cli.start()
            clients[f"rs.{c['serial']}"] = cli
        except Exception as e:
            print(f"  ... FAILED to start serial={c['serial']}: {e}")

    gs_paths = discover_gelsight_devices()
    print(f"[gelsight] discovered {len(gs_paths)} device(s):")
    for slot, p in enumerate(gs_paths):
        print(f"  slot {slot}  serial={gelsight_serial_from_path(p)}  {p}")
        cfg = OmegaConf.create({
            "paths": list(gs_paths),
            "record_size": [gs_width, gs_height],
            "fps": gs_fps,
            "fourcc": "MJPG",
            "warmup_frames": 30,
        })
        try:
            cli = GelsightClient(cfg, slot=slot)
            cli.start()
            clients[f"gs.{slot}"] = cli
        except Exception as e:
            print(f"  ... FAILED to start gelsight slot={slot}: {e}")

    return clients


def _stop_all(clients: dict) -> None:
    for name, cli in clients.items():
        try:
            cli.stop()  # type: ignore[attr-defined]
        except Exception as e:
            print(f"  ... stop {name} failed: {e}")


def _mosaic(frames: dict[str, np.ndarray | None], tile=(320, 240)) -> np.ndarray:
    """Compose a 2-col mosaic. Missing slots get a 'no signal' grey tile."""
    w, h = tile
    tiles = []
    for name in sorted(frames):
        img = frames[name]
        if img is None:
            tile_img = np.full((h, w, 3), 64, dtype=np.uint8)
            cv2.putText(tile_img, f"{name}: no frame", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            tile_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            cv2.putText(tile_img, name, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        tiles.append(tile_img)
    if not tiles:
        return np.zeros((h, w * 2, 3), dtype=np.uint8)
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    while len(tiles) < rows * cols:
        tiles.append(blank)
    rows_imgs = [np.hstack(tiles[i * cols:(i + 1) * cols]) for i in range(rows)]
    return np.vstack(rows_imgs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--rs-width", type=int, default=640)
    ap.add_argument("--rs-height", type=int, default=480)
    ap.add_argument("--rs-fps", type=int, default=15)
    ap.add_argument("--gs-width", type=int, default=320)
    ap.add_argument("--gs-height", type=int, default=240)
    ap.add_argument("--gs-fps", type=int, default=25)
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    clients = _build_clients(
        args.rs_width, args.rs_height, args.rs_fps,
        args.gs_width, args.gs_height, args.gs_fps,
    )
    if not clients:
        print("[fatal] no sensor clients started; aborting.", file=sys.stderr)
        return 2

    last_ts: dict[str, float] = {}
    intervals: dict[str, collections.deque] = {
        name: collections.deque(maxlen=1024) for name in clients
    }
    frame_count: dict[str, int] = {name: 0 for name in clients}

    t_start = time.monotonic()
    last_display = 0.0
    try:
        while time.monotonic() - t_start < args.duration:
            frames: dict[str, np.ndarray | None] = {}
            for name, cli in clients.items():
                try:
                    out = cli.get_latest()  # type: ignore[attr-defined]
                except Exception:
                    out = None
                if out is None:
                    frames[name] = None
                    continue
                rgb = out.get("rgb")
                ts = float(out.get("ts", 0.0))
                if ts:
                    prev = last_ts.get(name)
                    if prev:
                        intervals[name].append(ts - prev)
                    if not prev or ts != prev:
                        frame_count[name] += 1
                    last_ts[name] = ts
                frames[name] = rgb

            if not args.no_display:
                now = time.monotonic()
                if now - last_display > 0.05:
                    mosaic = _mosaic(frames)
                    cv2.imshow("01_visualize_sensors (q to quit)", mosaic)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    last_display = now
            else:
                time.sleep(0.01)
    finally:
        _stop_all(clients)
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print()
    print("=== fps summary ===")
    elapsed = max(time.monotonic() - t_start, 1e-6)
    for name, count in frame_count.items():
        fps = count / elapsed
        ivs = np.array(intervals[name]) if intervals[name] else np.array([0.0])
        print(
            f"  {name:24s}  frames={count:5d}  fps={fps:6.2f}  "
            f"interval mean={ivs.mean()*1000:6.1f}ms  "
            f"p99={np.percentile(ivs, 99)*1000:6.1f}ms  "
            f"max={ivs.max()*1000:6.1f}ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
