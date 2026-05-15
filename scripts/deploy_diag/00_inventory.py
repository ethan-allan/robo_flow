"""Stage S1 — hardware inventory probe (no sensors opened).

Composes a task yaml via hydra, walks the resolved cfg, runs each
discovery function in `trainflow.env.ur3_bir.discover`, and prints a
side-by-side table: what's plugged in vs what the cfg expects. Catches
"wrong serial" / "GelSight not power-cycled" / "robot offline" before
any streaming client is started — failure here means the cfg or the
rig is wrong, not the code.

Robot stays powered OFF (or e-stop pressed). Only RTDE handshake is
attempted; no motion commands are issued.

Usage:
    python -m scripts.deploy_diag.00_inventory --task peg_insertion_vrr_5fps

    # Skip the RTDE probe (cameras-only sanity check):
    python -m scripts.deploy_diag.00_inventory --task real_ee2_dice --skip-robot

Exit codes:
    0 = every configured-and-required device matched.
    1 = at least one mismatch / unreachable.

See scripts/deploy_diag/SENSOR_DEBUG_PLAN.md for go/no-go criteria.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import initialize_config_dir, compose

from trainflow.env.ur3_bir.discover import (
    detect_realsense_cameras,
    detect_ur_robot,
    discover_gelsight_devices,
    gelsight_serial_from_path,
)


REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "trainflow" / "config"


def _load_task(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg = compose(config_name=f"task/{name}")
    return cfg.task


def _check_realsense(task_cfg) -> tuple[bool, list[str]]:
    """For each obs whose client.kind == 'realsense', compare the
    configured serial against the detected device list. Camera obs
    keys with serial=null are flagged but don't fail the check
    (they're waiting to be populated per-rig)."""
    discovered = detect_realsense_cameras()
    by_serial = {c["serial"]: c["name"] for c in discovered}

    lines = [f"[realsense] discovered {len(discovered)} device(s):"]
    for c in discovered:
        lines.append(f"  serial={c['serial']}  name={c['name']}")

    ok = True
    cam_obs: list[tuple[str, object]] = []
    for key, attr in (task_cfg.shape_meta.get("obs", {}) or {}).items():
        client = attr.get("client")
        if client is not None and str(client.get("kind", "")) == "realsense":
            cam_obs.append((key, client))
    lines.append(f"[realsense] task obs uses {len(cam_obs)} camera(s):")
    for key, client in cam_obs:
        cfg_serial = client.get("serial", None)
        if cfg_serial is None:
            lines.append(f"  obs.{key}.client.serial=null  UNSET (populate per-rig)")
            continue
        matched = cfg_serial in by_serial
        status = "OK" if matched else "MISSING"
        lines.append(
            f"  obs.{key}.client.serial={cfg_serial}  {status}"
            + (f"  ({by_serial[cfg_serial]})" if matched else "")
        )
        if not matched:
            ok = False
    return ok, lines


def _check_gelsight(task_cfg) -> tuple[bool, list[str]]:
    """Discover GelSight devices, compare against the task's tactile
    mode + the slots actually referenced by obs."""
    paths = discover_gelsight_devices()
    lines = [f"[gelsight] discovered {len(paths)} device(s):"]
    for slot, p in enumerate(paths):
        lines.append(f"  slot {slot}  serial={gelsight_serial_from_path(p)}  {p}")

    tactile = task_cfg.hardware.get("tactile", {}) or {}
    mode = str(tactile.get("mode", "none"))
    lines.append(f"[gelsight] task.hardware.tactile.mode={mode}")

    gs_obs: list[tuple[str, int]] = []
    for key, attr in (task_cfg.shape_meta.get("obs", {}) or {}).items():
        client = attr.get("client")
        if client is not None and str(client.get("kind", "")) == "gelsight":
            gs_obs.append((key, int(client.get("slot", 0))))
    lines.append(f"[gelsight] task obs uses {len(gs_obs)} slot(s):")
    for key, slot in gs_obs:
        in_range = slot < len(paths)
        lines.append(
            f"  obs.{key}.client.slot={slot}  "
            f"{'OK' if in_range else 'NO DEVICE'}"
        )
    if gs_obs and mode != "gelsight":
        lines.append(f"  [warn] obs references gelsight but tactile.mode={mode}")
        return False, lines
    if gs_obs and any(slot >= len(paths) for _, slot in gs_obs):
        return False, lines
    return True, lines


def _check_robot(task_cfg) -> tuple[bool, list[str]]:
    robot_cfg = (task_cfg.hardware.get("clients", {})
                 .get("robot", {})
                 .get("ur3", {}) or {})
    host = robot_cfg.get("host", None)
    port = int(robot_cfg.get("rtde_port", 30004))
    lines = [f"[ur3] task.hardware.clients.robot.ur3.host={host!r}  rtde_port={port}"]
    if not host:
        lines.append("  host unset — skipping probe")
        return False, lines
    found = detect_ur_robot([str(host)], rtde_port=port)
    if found is None:
        lines.append(f"  RTDE handshake FAILED at {host}:{port}")
        return False, lines
    lines.append(f"  RTDE handshake OK at {found}")
    return True, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True,
                    help="task yaml name under config/task/, e.g. peg_insertion_vrr_5fps")
    ap.add_argument("--skip-robot", action="store_true",
                    help="Skip the RTDE handshake probe (cameras-only run).")
    args = ap.parse_args()

    try:
        task_cfg = _load_task(args.task)
    except Exception as e:
        print(f"[fatal] failed to compose task {args.task}: {e}", file=sys.stderr)
        return 2

    overall_ok = True

    rs_ok, rs_lines = _check_realsense(task_cfg)
    overall_ok &= rs_ok
    for line in rs_lines:
        print(line)
    print()

    gs_ok, gs_lines = _check_gelsight(task_cfg)
    overall_ok &= gs_ok
    for line in gs_lines:
        print(line)
    print()

    if args.skip_robot:
        print("[ur3] --skip-robot passed; not probing controller")
    else:
        ur_ok, ur_lines = _check_robot(task_cfg)
        overall_ok &= ur_ok
        for line in ur_lines:
            print(line)
    print()

    print(f"=== inventory: {'OK' if overall_ok else 'INCOMPLETE'} ===")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
