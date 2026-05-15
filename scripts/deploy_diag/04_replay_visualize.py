"""Phase A — offline replay viewer.

Walks a recorded episode through the deploy stack (sensor clients ->
obs_builder -> Ur3BirEnv.get_obs) and renders what the model would see.

Run BEFORE touching any hardware. If this looks wrong, no point hooking
up the robot. If it looks right, the deploy preprocessing chain matches
training; the next phases test live sensors and motion.

Usage:
    conda activate robo_flow
    cd trainflow
    python -m scripts.deploy_diag.04_replay_visualize \\
        --task peg_insertion_vrr_5fps \\
        --episode trainflow/dataset/peg_hole_tac/20260429115620 \\
        --n-obs-steps 2 \\
        --out /tmp/replay_viz \\
        [--legacy-rgb-ops]    # use bgr_to_rgb only (matches legacy zarr_writer)

Outputs:
    <out>/rgb_t<idx>.png        per-frame rgb obs reconstructed from CHW
    <out>/timeseries.png        eef_state / eef_force / action plots
    <out>/summary.txt           per-key shapes + dtypes + value ranges
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose

from trainflow.env.ur3_bir.ur3_bir_env import Ur3BirEnv

REPO = Path(__file__).resolve().parents[2]
CFG_DIR = REPO / "trainflow" / "config"


def _load_task(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg = compose(config_name=f"task/{name}")
    return cfg.task


def _mark_legacy_capture(task_cfg):
    """For legacy peg_hole_tac episodes, the capture script ran
    crop_x + resize before saving rgb.npy. Tell ObsBuilder via the
    sensor's `applied_at_capture` so those ops are skipped at deploy."""
    cfg = task_cfg.copy()
    cfg.hardware.cameras.platform_realsense.applied_at_capture = (
        OmegaConf.create(["crop_x", "resize"])
    )
    cfg.hardware.cameras.hand_realsense.applied_at_capture = (
        OmegaConf.create(["crop_x", "resize"])
    )
    return cfg


def _save_rgb_chw(arr_chw_float01: np.ndarray, path: Path) -> None:
    # arr is (C, H, W) float32 in [0,1] — convert to (H, W, C) uint8
    img = np.moveaxis(arr_chw_float01, 0, -1)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    try:
        import cv2
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except ImportError:
        from PIL import Image
        Image.fromarray(img).save(path)


def _save_timeseries(per_frame_obs: list[dict], action_seq: np.ndarray, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not installed — no timeseries plot")
        return
    keys = [k for k in per_frame_obs[0] if k != "rgb"]
    n_panels = len(keys) + (1 if action_seq is not None else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.5 * n_panels), squeeze=False)
    for i, key in enumerate(keys):
        ax = axes[i, 0]
        # each obs[key] is (T_obs_window, D); we plot the trailing frame per tick
        series = np.stack([o[key][-1] for o in per_frame_obs])  # (n_ticks, D)
        for d in range(series.shape[1]):
            ax.plot(series[:, d], label=f"{key}[{d}]", linewidth=0.8)
        ax.set_title(key)
        ax.legend(fontsize=6, ncol=4)
    if action_seq is not None:
        ax = axes[-1, 0]
        for d in range(action_seq.shape[1]):
            ax.plot(action_seq[:, d], label=f"action[{d}]", linewidth=0.8)
        ax.set_title("action (recorded)")
        ax.legend(fontsize=6, ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / "timeseries.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--episode", required=True, type=Path)
    p.add_argument("--n-obs-steps", type=int, default=2)
    p.add_argument("--obs-temporal-downsample-ratio", type=int, default=1)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--legacy-rgb-ops", action="store_true",
                   help="Use bgr_to_rgb only for rgb (matches what zarr_writer "
                        "applies to legacy capture-script output).")
    p.add_argument("--every", type=int, default=1, help="Save rgb every Nth frame.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    task_cfg = _load_task(args.task)
    if args.legacy_rgb_ops:
        task_cfg = _mark_legacy_capture(task_cfg)

    # Discover episode length from one npy file.
    ts = np.load(args.episode / "frame_timestamp.npy")
    T = len(ts)
    print(f"[info] episode {args.episode.name}  frames={T}  task={args.task}")

    env = Ur3BirEnv.from_npy_replay(
        task_cfg=task_cfg,
        episode_dir=args.episode,
        n_obs_steps=args.n_obs_steps,
        obs_temporal_downsample_ratio=args.obs_temporal_downsample_ratio,
    )

    per_frame_obs = []
    rgb_saved = 0
    summary_lines = []

    for cur in range(args.n_obs_steps - 1, T):
        env.seek_replay(cur)
        obs = env.get_obs()
        per_frame_obs.append({k: v for k, v in obs.items()})
        if "rgb" in obs and (cur - (args.n_obs_steps - 1)) % args.every == 0:
            # Most-recent frame in the obs window is index -1
            _save_rgb_chw(obs["rgb"][-1], args.out / f"rgb_t{cur:04d}.png")
            rgb_saved += 1

    # Per-key summary on the last tick.
    last = per_frame_obs[-1]
    for key, arr in last.items():
        line = (f"  {key:14s}  shape={tuple(arr.shape)}  dtype={arr.dtype}  "
                f"min={float(np.asarray(arr).min()):.4g}  "
                f"max={float(np.asarray(arr).max()):.4g}")
        summary_lines.append(line)

    # Recorded action for plotting (raw .npy, no decode — visual context).
    action_path = args.episode / "vrr_action.npy"
    if not action_path.exists():
        action_path = args.episode / "eef_action.npy"
    action_seq = np.load(action_path) if action_path.exists() else None

    _save_timeseries(per_frame_obs, action_seq, args.out)

    summary = (
        f"task={args.task}  episode={args.episode.name}  ticks={len(per_frame_obs)}\n"
        f"n_obs_steps={args.n_obs_steps}  obs_downsample_ratio={args.obs_downsample_ratio if hasattr(args,'obs_downsample_ratio') else args.obs_temporal_downsample_ratio}\n"
        f"rgb_frames_saved={rgb_saved}\n\n"
        "Per-key obs window (last tick):\n" + "\n".join(summary_lines) + "\n"
    )
    (args.out / "summary.txt").write_text(summary)
    print(summary)
    print(f"[done] artifacts in {args.out}")


if __name__ == "__main__":
    main()
