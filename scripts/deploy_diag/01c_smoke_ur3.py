"""Stage S2 — per-sensor UR3 smoke (RTDE read-only).

Wraps `UR3Client` to print one row per tick at `--rate` Hz for
`--duration` seconds. No motion commands. Use this BEFORE
`02_robot_readonly.py` — both are read-only but this one is faster to
diagnose connection issues; `02` runs longer and computes a
stationary-stddev baseline.

Loads the UR3 connection cfg from `hardware/ur3/ur3_base.yaml` via
hydra so all UR3 cfgs (smoke, env, runner) share one source of truth.
Override the host on the CLI to point at a different controller.

Usage:
    python -m scripts.deploy_diag.01c_smoke_ur3 --duration 10 --rate 5

    # Point at a different controller without editing the cfg:
    python -m scripts.deploy_diag.01c_smoke_ur3 --host 192.168.20.66
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from hydra import initialize_config_dir, compose

from trainflow.env.ur3_bir.sensor_clients import UR3Client


REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "trainflow" / "config"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=None,
                    help="Override cfg.host (ur3_base.yaml).")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--rate", type=float, default=5.0)
    args = ap.parse_args()

    with initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg = compose(config_name="hardware/ur3/ur3_base")
    robot_cfg = cfg.hardware.ur3
    if args.host is not None:
        robot_cfg.host = args.host
    if not robot_cfg.get("host"):
        print("[fatal] ur3 host unset (pass --host)", file=sys.stderr)
        return 2

    print(f"[info] opening UR3 at {robot_cfg.host}:{robot_cfg.get('rtde_port', 30004)}")
    client = UR3Client(robot_cfg)
    try:
        client.start()
    except Exception as e:
        print(f"[fatal] start failed: {e}", file=sys.stderr)
        return 2

    interval = 1.0 / float(args.rate)
    n_ticks = 0
    t_start = time.monotonic()
    next_t = t_start
    try:
        while time.monotonic() - t_start < args.duration:
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(next_t - now, 0.05))
                continue
            next_t += interval

            out = client.get_latest()
            if out is None:
                continue
            n_ticks += 1
            tcp = out["tcp_pose6"]
            force = out["eef_force"]
            gw = float(out["gripper_width"][0])
            cm = float(out["control_mode"][0])
            elapsed = now - t_start
            print(
                f"t={elapsed:6.2f}s  "
                f"tcp=[{tcp[0]:+.4f} {tcp[1]:+.4f} {tcp[2]:+.4f} "
                f"{tcp[3]:+.3f} {tcp[4]:+.3f} {tcp[5]:+.3f}]  "
                f"|F|={np.linalg.norm(force[:3]):5.2f}N  "
                f"|tau|={np.linalg.norm(force[3:]):5.2f}Nm  "
                f"grip={gw:.4f}  cmode={int(cm)}"
            )
    except KeyboardInterrupt:
        print("[interrupt] stopping")
    finally:
        client.stop()

    print(f"=== summary === ticks={n_ticks}")
    errs = client.producer_errors
    if errs:
        print(f"  producer errors ({len(errs)}, last 5):")
        for e in errs[-5:]:
            print(f"    {e}")
    return 0 if n_ticks > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
