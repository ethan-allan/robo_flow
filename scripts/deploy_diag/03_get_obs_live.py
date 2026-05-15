"""Phase D — live `env.get_obs()` dry run.

Builds `Ur3BirEnv` against live sensors, starts the producer thread,
and at the configured `control_fps` polls `env.get_obs()` for
`--duration` seconds. Displays the trailing rgb obs in a cv2 window
and logs per-tick latency + eef_state / eef_force samples to `--out`.

The robot is NOT commanded — manual teach-jog the arm during the run
to confirm `eef_state` follows motion smoothly.

Usage:
    python -m scripts.deploy_diag.03_get_obs_live \\
        --task peg_insertion_vrr_5fps \\
        --duration 30 \\
        --out /tmp/obs_live

Go/no-go criteria are in scripts/deploy_diag/README.md (Phase D).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from hydra import initialize_config_dir, compose

from trainflow.env.ur3_bir.ur3_bir_env import Ur3BirEnv


REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "trainflow" / "config"


def _load_task(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg = compose(config_name=f"task/{name}")
    return cfg.task


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    help="task yaml name under config/task/, e.g. peg_insertion_vrr_5fps")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--n-obs-steps", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("/tmp/obs_live"))
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    task_cfg = _load_task(args.task)
    control_fps = float(task_cfg.hardware.get("control_fps", 5))
    interval = 1.0 / control_fps

    env = Ur3BirEnv(
        task_cfg=task_cfg,
        n_obs_steps=args.n_obs_steps,
        obs_temporal_downsample_ratio=1,
    )
    env.start()
    try:
        # Wait for buffer warmup.
        warm_t = time.monotonic()
        while len(env._buffer) < args.n_obs_steps:
            if time.monotonic() - warm_t > 5.0:
                print("[fatal] buffer warmup timed out; producer errors:")
                for e in env.producer_errors[-5:]:
                    print(f"  {e}")
                return 2
            time.sleep(0.02)

        log_fp = open(args.out / "ticks.jsonl", "w", buffering=1)
        latencies = []
        t_start = time.monotonic()
        next_t = t_start

        while time.monotonic() - t_start < args.duration:
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(next_t - now, 0.005))
                continue
            next_t += interval

            t0 = time.monotonic()
            try:
                obs = env.get_obs()
            except Exception as e:
                print(f"[get_obs] {e}")
                continue
            lat_ms = (time.monotonic() - t0) * 1000.0
            latencies.append(lat_ms)

            row: dict = {"t": time.time(), "latency_ms": lat_ms}
            if "eef_state" in obs:
                row["eef_state_last"] = obs["eef_state"][-1].tolist()
            if "eef_force" in obs:
                row["eef_force_last"] = obs["eef_force"][-1].tolist()
            log_fp.write(json.dumps(row) + "\n")

            if not args.no_display and "rgb" in obs:
                # rgb is (T, C, H, W) float32 in [0,1]; show the most recent frame.
                rgb = obs["rgb"][-1]                                # (C, H, W)
                img = (np.moveaxis(rgb, 0, -1) * 255).astype(np.uint8)
                # Display in BGR for cv2 (obs is RGB after bgr_to_rgb op).
                img_bgr = img[..., ::-1].copy()
                cv2.putText(
                    img_bgr,
                    f"latency={lat_ms:5.1f}ms",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
                cv2.imshow("03_get_obs_live (q to quit)", img_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        log_fp.close()
    finally:
        env.stop()
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print()
    print("=== latency summary ===")
    if not latencies:
        print("  no ticks recorded")
        return 2
    arr = np.array(latencies)
    print(f"  ticks={len(arr)}  mean={arr.mean():.1f}ms  "
          f"p50={np.percentile(arr,50):.1f}ms  p99={np.percentile(arr,99):.1f}ms  "
          f"max={arr.max():.1f}ms")
    on_budget = float((arr < (interval * 1000)).sum()) / len(arr)
    print(f"  within {interval*1000:.0f}ms control period: {on_budget*100:.2f}%")
    if env.producer_errors:
        print(f"  producer errors (last 5): {env.producer_errors[-5:]}")
    print(f"  log: {args.out/'ticks.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
