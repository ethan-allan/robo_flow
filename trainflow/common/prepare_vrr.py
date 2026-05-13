"""Per-episode VRR action preprocessor.

Reads each episode subdir under --src, computes the 19-D VRR action from
eef_state.npy (T, 7 axis-angle pose) and eef_force.npy (T, 6 F/T), and
writes vrr_action.npy (T, 19) alongside.

VRR action layout (per timestep):
    [tcp_xyz(3), tcp_rot6d(6), virtual_xyz(3), virtual_rot6d(6), stiffness(1)]

Virtual target = tcp - R_world_from_tcp @ (F_local / stiffness)
Stiffness is piecewise-linear in |F_xyz|: k_max in free space, k_min in contact.

Force is tared per episode (subtract mean of first N frames) to remove
between-episode sensor drift.

Example:
    python -m trainflow.common.prepare_vrr \\
        --src trainflow/dataset/ee2dice \\
        --task trainflow/config/task/real_dpt_vrr_10fps.yaml \\
        --dry-run

VRR parameters can be set in the task yaml under a top-level `vrr:` block;
otherwise upstream defaults are used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from trainflow.common.space_utils import pose_3d_9d_to_homo_matrix_batch


DEFAULT_VRR_PARAMS = {
    "k_max": 10000.0,
    "k_min": 200.0,
    "f_low": 0.5,
    "f_high": 5.0,
    "wrench_average_window": 1,
    "tare_frames": 5,
}


def axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
    R = Rotation.from_rotvec(axis_angle.reshape(-1, 3)).as_matrix()
    rot6d = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1)
    return rot6d.reshape(*axis_angle.shape[:-1], 6)


def pose7_to_pose9(arr: np.ndarray) -> np.ndarray:
    assert arr.ndim == 2 and arr.shape[1] == 7, f"expected (T, 7), got {arr.shape}"
    xyz = arr[:, 0:3]
    rot6d = axis_angle_to_rot6d(arr[:, 3:6])
    return np.concatenate([xyz, rot6d], axis=1).astype(np.float32)


def moving_average_1d(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return data.copy()
    kernel = np.ones(window) / window
    out = np.empty_like(data)
    for d in range(data.shape[1]):
        out[:, d] = np.convolve(data[:, d], kernel, mode="same")
    return out


def compute_stiffness(wrench_xyz: np.ndarray, k_max: float, k_min: float,
                      f_low: float, f_high: float) -> np.ndarray:
    mag = np.linalg.norm(wrench_xyz, axis=-1)
    interp = k_max - (k_max - k_min) * (mag - f_low) / (f_high - f_low)
    k = np.where(mag < f_low, k_max,
                 np.where(mag > f_high, k_min, interp))
    return k[:, None].astype(np.float32)


def compute_global_offset(local_offset: np.ndarray, tcp_pose_9d: np.ndarray) -> np.ndarray:
    H = pose_3d_9d_to_homo_matrix_batch(tcp_pose_9d.astype(np.float32))
    R = H[:, :3, :3]
    return np.einsum("tij,tj->ti", R, local_offset)


def tare_force(force: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    if n_frames <= 0:
        return force.copy(), np.zeros(force.shape[1], dtype=force.dtype)
    n = min(n_frames, len(force))
    baseline = force[:n].mean(axis=0)
    return force - baseline, baseline


def build_vrr_action(eef_state_7d: np.ndarray,
                     eef_force_6d: np.ndarray,
                     params: dict) -> tuple[np.ndarray, dict]:
    """Returns (action, stats). action shape (T, 19), action ≈ next(state)."""
    tcp_pose = pose7_to_pose9(eef_state_7d)
    wrench_avg = moving_average_1d(eef_force_6d[:, :3], params["wrench_average_window"])

    stiffness = compute_stiffness(
        wrench_avg, params["k_max"], params["k_min"],
        params["f_low"], params["f_high"],
    )
    local_offset = wrench_avg / stiffness
    global_offset = compute_global_offset(local_offset, tcp_pose)

    virtual_xyz = tcp_pose[:, :3] - global_offset
    virtual_rot = tcp_pose[:, 3:9]

    state = np.concatenate([tcp_pose, virtual_xyz, virtual_rot, stiffness],
                           axis=-1).astype(np.float32)
    action = np.concatenate([state[1:], state[-1:]], axis=0)

    stats = {
        "T": int(state.shape[0]),
        "force_mag_p50": float(np.median(np.linalg.norm(wrench_avg, axis=-1))),
        "force_mag_p95": float(np.quantile(np.linalg.norm(wrench_avg, axis=-1), 0.95)),
        "stiffness_min": float(stiffness.min()),
        "stiffness_max": float(stiffness.max()),
        "stiffness_mean": float(stiffness.mean()),
        "virtual_offset_max_m": float(np.linalg.norm(global_offset, axis=-1).max()),
    }
    return action, stats


def load_vrr_params(task_path: Path | None) -> dict:
    params = dict(DEFAULT_VRR_PARAMS)
    if task_path is None:
        return params
    cfg = OmegaConf.load(task_path)
    user = cfg.get("vrr", {}) or {}
    for k in params:
        if k in user:
            params[k] = user[k]
    return params


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True,
                   help="Root containing one subdir per episode.")
    p.add_argument("--task", type=Path, required=True,
                   help="Task yaml; reads top-level `vrr:` block for params.")
    p.add_argument("--tare-frames", type=int, default=None,
                   help="Override vrr.tare_frames from task cfg.")
    p.add_argument("--out-name", default="vrr_action.npy",
                   help="Output filename per episode.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print stats, do not write files.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N episodes (debug).")
    args = p.parse_args()

    params = load_vrr_params(args.task)
    if args.tare_frames is not None:
        params["tare_frames"] = args.tare_frames

    print("VRR parameters:")
    for k, v in params.items():
        print(f"  {k:24s} {v}")
    print()

    eps = sorted(d for d in args.src.iterdir() if d.is_dir())
    if args.limit is not None:
        eps = eps[:args.limit]
    if not eps:
        raise SystemExit(f"No episode subdirs in {args.src}")
    print(f"Processing {len(eps)} episodes from {args.src}\n")

    n_written = 0
    n_skipped = 0
    baselines = []
    for ep in tqdm(eps, desc="episodes", disable=args.dry_run):
        state_p = ep / "eef_state.npy"
        force_p = ep / "eef_force.npy"
        if not state_p.exists() or not force_p.exists():
            tqdm.write(f"  skip {ep.name}: missing eef_state.npy or eef_force.npy")
            n_skipped += 1
            continue
        state = np.load(state_p)
        force = np.load(force_p)
        if state.shape[0] != force.shape[0]:
            tqdm.write(f"  skip {ep.name}: T mismatch state={state.shape[0]} force={force.shape[0]}")
            n_skipped += 1
            continue

        force_t, baseline = tare_force(force, params["tare_frames"])
        baselines.append(baseline)
        action, stats = build_vrr_action(state, force_t, params)

        if args.dry_run:
            print(f"  {ep.name}  T={stats['T']:4d}  "
                  f"|F| p50={stats['force_mag_p50']:6.3f}  p95={stats['force_mag_p95']:6.3f}  "
                  f"K=[{stats['stiffness_min']:6.0f},{stats['stiffness_max']:6.0f}] mean={stats['stiffness_mean']:6.0f}  "
                  f"v_off_max={stats['virtual_offset_max_m']*1000:.2f}mm  "
                  f"baseline_F=[{baseline[0]:+6.3f},{baseline[1]:+6.3f},{baseline[2]:+6.3f}]")
        else:
            np.save(ep / args.out_name, action)
            n_written += 1

    if baselines:
        baselines = np.stack(baselines, axis=0)
        print("\nPer-episode force baselines (subtracted from each episode):")
        print(f"  mean across episodes: {baselines.mean(axis=0)}")
        print(f"  std  across episodes: {baselines.std(axis=0)}")
        print(f"  min/max per axis     min: {baselines.min(axis=0)}")
        print(f"                       max: {baselines.max(axis=0)}")

    if args.dry_run:
        print(f"\nDry run complete. Would have written {len(eps) - n_skipped} files.")
    else:
        print(f"\nWrote {n_written} files. Skipped {n_skipped}.")


if __name__ == "__main__":
    main()
