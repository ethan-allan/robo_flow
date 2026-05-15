"""Step-4 load-bearing test: env.get_obs() == RealImageTactileDataset.__getitem__'s
obs_dict, bit-for-bit, on the same frames.

Strategy:
  1. Load a real episode under trainflow/dataset/peg_hole_tac/.
  2. Compute the "ground truth" inline: load each .npy, run
     zarr_writer.load_key()'s per-frame ops, then __getitem__'s
     temporal+formatting transforms over a trailing window.
  3. Build a Ur3BirEnv in replay mode against the same episode at the
     same trailing-window endpoint; call env.get_obs().
  4. Assert per-key np.array_equal between the two.

Legacy-data caveat:
  The episodes under peg_hole_tac were captured by the legacy script
  (record_data_gui_new.py), which crops + resizes + BGR->RGB *before*
  saving rgb.npy. The deploy obs_sources op chain in the task yamls is
  written for raw frames (the future per-dataset capture script's
  output). For this test we override `obs_sources.rgb.ops` with just
  `bgr_to_rgb` so it matches what zarr_writer.load_key actually does
  for the on-disk rgb.npy. This documents the divergence; the
  framework path is fully exercised either way.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose

from trainflow.common import obs_processing as ops_mod
from trainflow.env.ur3_bir.ur3_bir_env import Ur3BirEnv


REPO = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO / "trainflow" / "dataset" / "peg_hole_tac"
CFG_DIR = REPO / "trainflow" / "config"


def _first_episode() -> Path:
    return sorted(p for p in DATASET_DIR.iterdir() if p.is_dir())[0]


def _load_task(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg = compose(config_name=f"task/{name}")
    return cfg.task


def _mark_legacy_capture(task_cfg):
    """Tell ObsBuilder which rgb ops were already applied at capture
    time for legacy peg_hole_tac episodes (crop_x + resize ran in the
    capture script before rgb.npy was saved). The bgr_to_rgb op stays
    in the chain because zarr_writer also calls bgr_to_rgb on load.

    This is the framework's intended mechanism — preferable to
    overriding obs_sources directly, which would hide the divergence
    from the reader of the task yaml.
    """
    cfg = task_cfg.copy()
    cfg.hardware.cameras.platform_realsense.applied_at_capture = (
        OmegaConf.create(["crop_x", "resize"])
    )
    cfg.hardware.cameras.hand_realsense.applied_at_capture = (
        OmegaConf.create(["crop_x", "resize"])
    )
    return cfg


def _ground_truth_obs(ep: Path, key: str, key_type: str, target_shape,
                      cur_idx: int, n_obs_steps: int, ratio: int) -> np.ndarray:
    """Reproduce zarr_writer.load_key + __getitem__ formatting inline."""
    fname = "eef_state.npy" if key == "eef_state" else f"{key}.npy"
    raw = np.load(ep / fname)

    # zarr_writer.load_key per-frame ops
    if key_type == "rgb":
        data = ops_mod.bgr_to_rgb(raw)
    elif key in ("eef_state", "eef_action", "action"):
        data = ops_mod.pose7_to_pose9(raw)
    else:
        data = raw.astype(np.float32, copy=False)

    # Trailing window of length n_obs_steps ending at cur_idx (inclusive).
    # The dataset's T_slice = slice(n_obs_steps) on a sample_sequence
    # output that starts at the matching idx; here we slice directly.
    start = cur_idx - (n_obs_steps - 1)
    assert start >= 0, "test idx too small for n_obs_steps"
    window = data[start : cur_idx + 1]               # (n_obs_steps, ...)

    # __getitem__ ops
    arr_t = window[::-ratio][::-1]
    if key_type == "rgb":
        return np.moveaxis(arr_t, -1, 1).astype(np.float32) / 255.0
    if len(target_shape) == 1:
        return arr_t[:, : target_shape[0]].astype(np.float32)
    if len(target_shape) == 2:
        return arr_t[:, : target_shape[0], : target_shape[1]].astype(np.float32)
    raise ValueError(target_shape)


def run_one(task_name: str, n_obs_steps: int, cur_idx: int) -> tuple[bool, list[str]]:
    ep = _first_episode()
    task_cfg = _load_task(task_name)
    task_cfg_overridden = _mark_legacy_capture(task_cfg)
    env = Ur3BirEnv.from_npy_replay(
        task_cfg=task_cfg_overridden,
        episode_dir=ep,
        n_obs_steps=n_obs_steps,
    )
    env.seek_replay(cur_idx)
    obs = env.get_obs()

    notes: list[str] = []
    ok = True
    for key, sm in task_cfg_overridden.shape_meta.obs.items():
        if "wrt" in key:
            continue
        gt = _ground_truth_obs(
            ep=ep,
            key=key,
            key_type=sm.type,
            target_shape=list(sm.shape),
            cur_idx=cur_idx,
            n_obs_steps=n_obs_steps,
            ratio=env.obs_downsample_ratio,
        )
        env_arr = obs[key]
        if env_arr.shape != gt.shape:
            ok = False
            notes.append(f"  [{task_name}/{key}] SHAPE: env={env_arr.shape} gt={gt.shape}")
            continue
        if env_arr.dtype != gt.dtype:
            ok = False
            notes.append(f"  [{task_name}/{key}] DTYPE: env={env_arr.dtype} gt={gt.dtype}")
            continue
        if not np.array_equal(env_arr, gt):
            ok = False
            mad = float(np.abs(env_arr - gt).max())
            notes.append(f"  [{task_name}/{key}] VALUE diff: max|d|={mad}")
            continue
        notes.append(f"  [{task_name}/{key}] OK shape={env_arr.shape} dtype={env_arr.dtype}")
    return ok, notes


def main():
    cases = [
        ("real_ee2_dice",                n := 2, 5),
        ("peg_insertion_vrr_5fps",       n := 2, 5),
        ("peg_insertion_reactive_5fps",  n := 2, 5),
    ]
    failed = False
    for task_name, n_obs_steps, cur_idx in cases:
        ok, notes = run_one(task_name, n_obs_steps, cur_idx)
        print(f"[{'PASS' if ok else 'FAIL'}] {task_name}  n_obs_steps={n_obs_steps}  cur_idx={cur_idx}")
        for line in notes:
            print(line)
        failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
