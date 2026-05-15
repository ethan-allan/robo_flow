"""Stage S2 — per-sensor GelSight smoke.

Open ONE GelSight Mini by slot index (0 or 1), pull frames for
`--duration` seconds, render in a window unless `--no-display`. The
slot mapping is the sorted /dev/v4l/by-id order; check 00_inventory
output to know which physical sensor is which slot.

Usage:
    python -m scripts.deploy_diag.01b_smoke_gelsight --slot 0 --duration 10
"""
from __future__ import annotations

import argparse
import sys
import time

from omegaconf import OmegaConf

from trainflow.env.ur3_bir.sensor_clients import GelsightClient


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--width", type=int, default=320,
                    help="record_size width (matches the lab default).")
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    gs_cfg = OmegaConf.create({
        "record_size": [args.width, args.height],
        "fps": args.fps,
        "fourcc": "MJPG",
        "warmup_frames": 30,
    })
    print(f"[info] opening GelSight slot={args.slot}")
    client = GelsightClient(gs_cfg, slot=args.slot)
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
                cv2.imshow(f"01b_smoke_gelsight slot={args.slot} (q to quit)", img)
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
