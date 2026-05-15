"""Phase C — robot read-only via RTDE.

Open RTDE + gripper, sample at `--rate` for `--duration` seconds, print
the running state, and at the end compute a per-channel stationary
stddev so cable/noise issues fail loudly. The script NEVER sends a
motion command — Phase C runs with the e-stop pressed and the operator
verifying that the teach-pendant readout matches what's printed.

Loads the UR3 connection cfg from `hardware/ur3/ur3_base.yaml` via
hydra. Override the host on the CLI to point at a different controller.

Usage:
    python -m scripts.deploy_diag.02_robot_readonly --duration 60 --rate 5

Go/no-go criteria are in scripts/deploy_diag/README.md (Phase C).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from hydra import initialize_config_dir, compose

from trainflow.env.ur3_bir.discover import detect_ur_robot
from trainflow.env.ur3_bir.sensor_clients import UR3Client


REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "trainflow" / "config"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None,
                    help="Override cfg.host (ur3_base.yaml).")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--rate", type=float, default=5.0)
    ap.add_argument("--skip-probe", action="store_true",
                    help="Skip the ping/RTDE probe; assume cfg.host is reachable.")
    args = ap.parse_args()

    with initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg = compose(config_name="hardware/ur3/ur3_base")
    robot_cfg = cfg.hardware.ur3
    if args.host is not None:
        robot_cfg.host = args.host
    if not robot_cfg.get("host"):
        print("[fatal] ur3 host unset (pass --host)", file=sys.stderr)
        return 2

    if not args.skip_probe:
        host = str(robot_cfg.host)
        port = int(robot_cfg.get("rtde_port", 30004))
        print(f"[probe] checking {host}:{port} ...")
        found = detect_ur_robot([host], rtde_port=port)
        if found is None:
            print(f"[fatal] no RTDE handshake at {host}:{port}", file=sys.stderr)
            return 2
        print(f"[probe] OK: {found}")

    client = UR3Client(robot_cfg)
    try:
        client.start()
    except Exception as e:
        print(f"[fatal] UR3Client.start failed: {e}", file=sys.stderr)
        return 2

    interval = 1.0 / float(args.rate)
    samples: dict[str, list[np.ndarray]] = {
        "tcp_pose6": [], "gripper_width": [], "eef_force": [],
        "joint_state": [], "current": [], "control_mode": [],
    }
    t_start = time.monotonic()
    try:
        next_t = t_start
        while time.monotonic() - t_start < args.duration:
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(next_t - now, 0.05))
                continue
            next_t += interval
            out = client.get_latest()
            for k in samples:
                samples[k].append(np.asarray(out[k]))
            elapsed = now - t_start
            tcp = out["tcp_pose6"]; force = out["eef_force"]
            gw = float(out["gripper_width"][0])
            print(
                f"t={elapsed:6.2f}s  "
                f"tcp=[{tcp[0]:+.4f} {tcp[1]:+.4f} {tcp[2]:+.4f} "
                f"{tcp[3]:+.3f} {tcp[4]:+.3f} {tcp[5]:+.3f}]  "
                f"|F|={np.linalg.norm(force[:3]):5.2f}N  "
                f"|τ|={np.linalg.norm(force[3:]):5.2f}Nm  "
                f"gripper={gw:.4f}"
            )
    except KeyboardInterrupt:
        print("[interrupt] stopping")
    finally:
        client.stop()

    print()
    print("=== stationary stddev baseline ===")
    for k, vals in samples.items():
        if not vals:
            continue
        arr = np.stack(vals, axis=0)
        std = arr.std(axis=0)
        with np.printoptions(precision=5, suppress=True):
            print(f"  {k:14s} stddev = {std}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
