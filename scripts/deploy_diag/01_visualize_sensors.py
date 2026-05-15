"""Phase B — sensors-only live visualisation.

Open every camera + gelsight named in the hardware cfg, render a 2x2
OpenCV mosaic for `--duration` seconds, and print per-stream fps + a
histogram of frame intervals at the end. The robot stays powered OFF.

Usage:
    python -m scripts.deploy_diag.01_visualize_sensors \\
        --hw trainflow/config/hardware/ur3_bir_default.yaml \\
        --duration 30

Go/no-go criteria are listed in scripts/deploy_diag/README.md (Phase B).
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


def _build_camera_clients(hw_cfg) -> dict:
    """Build a RealsenseClient per enabled camera plus a GelsightClient
    per discovered gelsight slot. Errors are logged but never raised
    (one missing camera shouldn't block the rest of the mosaic)."""
    clients: dict[str, object] = {}

    discovered_rs = {c["serial"]: c["name"] for c in detect_realsense_cameras()}
    print(f"[realsense] discovered {len(discovered_rs)} device(s):")
    for serial, name in discovered_rs.items():
        print(f"  {serial}  {name}")

    cams = hw_cfg.get("cameras", {}) or {}
    for cam_name, cam_cfg in cams.items():
        if not cam_cfg.get("enabled", False):
            continue
        cfg_serial = cam_cfg.get("serial", None)
        match = (cfg_serial in discovered_rs) if cfg_serial else None
        print(f"[realsense] cfg.{cam_name}.serial={cfg_serial!r} "
              f"{'OK' if match else 'MISSING' if cfg_serial else 'unset'}")
        if not cfg_serial:
            print(f"  ... skip {cam_name}; populate cfg.cameras.{cam_name}.serial first")
            continue
        try:
            cli = RealsenseClient(cam_cfg)
            cli.start()
            clients[f"cam.{cam_name}"] = cli
        except Exception as e:
            print(f"  ... FAILED to start {cam_name}: {e}")

    gs_paths = discover_gelsight_devices()
    print(f"[gelsight] discovered {len(gs_paths)} sensor(s):")
    for p in gs_paths:
        print(f"  slot=? serial={gelsight_serial_from_path(p)!r}  {p}")
    gs_cfg = hw_cfg.get("tactile", {}).get("gelsight", {})
    for slot in range(len(gs_paths)):
        try:
            cli = GelsightClient(gs_cfg, slot=slot)
            cli.start()
            clients[f"gelsight.{slot}"] = cli
        except Exception as e:
            print(f"  ... FAILED to start gelsight slot={slot}: {e}")

    return clients


def _stop_all(clients: dict) -> None:
    for name, cli in clients.items():
        try:
            cli.stop()
        except Exception as e:
            print(f"  ... stop {name} failed: {e}")


def _mosaic(frames: dict[str, np.ndarray | None], tile=(320, 240)) -> np.ndarray:
    """Compose a 2-col mosaic. Missing slots get a 'no signal' gray tile."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", type=Path, default=Path(
        "trainflow/config/hardware/ur3_bir_default.yaml"))
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    if not args.hw.exists():
        print(f"[fatal] hw cfg not found: {args.hw}", file=sys.stderr)
        return 2
    hw_cfg = OmegaConf.load(str(args.hw))

    clients = _build_camera_clients(hw_cfg)
    if not clients:
        print("[fatal] no sensor clients started; aborting.", file=sys.stderr)
        return 2

    # Per-stream stats: last_ts, interval samples, frame count.
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
                    out = cli.get_latest()
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
            f"  {name:20s}  frames={count:5d}  fps={fps:6.2f}  "
            f"interval mean={ivs.mean()*1000:6.1f}ms  "
            f"p99={np.percentile(ivs, 99)*1000:6.1f}ms  "
            f"max={ivs.max()*1000:6.1f}ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
