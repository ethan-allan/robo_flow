"""Stage S2 — per-sensor RealSense smoke.

Open ONE RealSense camera by serial, pull frames for `--duration`
seconds, render in a window (unless `--no-display`), and print fps +
warmup state. The robot stays OFF. Use this BEFORE
`01_visualize_sensors.py` so per-camera failures are easy to localise.

Usage:
    python -m scripts.deploy_diag.01a_smoke_realsense \\
        --serial 248622300418 --duration 10
"""
from __future__ import annotations

import argparse
import sys
import time

from omegaconf import OmegaConf

from trainflow.env.ur3_bir.sensor_clients import RealsenseClient


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", required=False,
                    help="RealSense device serial. Use `00_inventory.py` to list.")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    cam_cfg = OmegaConf.create({
        "serial": args.serial,
        "width": args.width, "height": args.height, "fps": args.fps,
    })
    print(f"[info] opening RealSense serial={args.serial}  "
          f"{args.width}x{args.height}@{args.fps}fps")
    client = RealsenseClient(cam_cfg)
    try:
        client.start()
    except Exception as e:
        print(f"[fatal] start failed: {e}", file=sys.stderr)
        return 2

    cv2 = None
    if not args.no_display:
        import cv2  # noqa: F811

    t_start = time.monotonic()
    n_frames = 0
    ts_seen: set[float] = set()
    last_shape = None
    try:
        while time.monotonic() - t_start < args.duration:
            out = client.get_latest()
            if out is None:
                time.sleep(0.005)
                continue
            ts = float(out["ts"])
            if ts in ts_seen:
                if cv2 is not None:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                time.sleep(0.005)
                continue
            ts_seen.add(ts)
            n_frames += 1
            img = out["rgb"]
            last_shape = img.shape
            if cv2 is not None:
                cv2.imshow("01a_smoke_realsense (q to quit)", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        client.stop()
        if cv2 is not None:
            try: cv2.destroyAllWindows()
            except Exception: pass

    elapsed = max(time.monotonic() - t_start, 1e-6)
    print(f"=== summary ===")
    print(f"  frames: {n_frames}  duration: {elapsed:.2f}s  fps: {n_frames/elapsed:.2f}")
    print(f"  shape:  {last_shape}")
    errs = client.producer_errors
    if errs:
        print(f"  producer errors ({len(errs)}, last 5):")
        for e in errs[-5:]:
            print(f"    {e}")
    return 0 if n_frames > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
