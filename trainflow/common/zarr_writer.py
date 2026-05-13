"""Config-driven replay-buffer zarr writer.

Reads a task YAML, walks `shape_meta.{obs,extended_obs,action}`, and writes
one zarr array per declared key from the matching `<key>.npy` file in each
episode directory.

Example:

    python -m trainflow.common.zarr_writer \
        --src  trainflow/dataset/ee2dice \
        --task trainflow/config/task/real_ee2_dice.yaml \
        --dst  data/ee2dice_zarr
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr
from numcodecs import Blosc
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from trainflow.common.replay_buffer import ReplayBuffer


# Builtin default: cfg key -> source filename when the two differ.
# A task yaml can override / extend these via a top-level `source_aliases:` block.
DEFAULT_SOURCE_ALIASES = {
    "action": "eef_action.npy",
}

# Builtin default: cfg keys whose source file holds (T, 7) axis-angle pose
# and needs 7 -> 9 ortho6d conversion (gripper channel dropped).
# A task yaml can override the set via a top-level `pose_keys:` list.
DEFAULT_POSE_KEYS = {"eef_state", "eef_action", "action"}

# Marker for VRR-style tasks: action is precomputed (19-D) by prepare_vrr.py.
VRR_ACTION_TYPE = "right_arm_6DOF_virtual_target_stiffness"
VRR_ACTION_FILE = "vrr_action.npy"


def resolve_source_aliases(task_cfg) -> dict[str, str]:
    """Merge builtin defaults with task-cfg overrides.

    VRR action type swaps the default `action` source to `vrr_action.npy`.
    A `source_aliases:` block in the task yaml further overrides anything.
    """
    aliases = dict(DEFAULT_SOURCE_ALIASES)
    if task_cfg.get("action_type") == VRR_ACTION_TYPE:
        aliases["action"] = VRR_ACTION_FILE
    user = task_cfg.get("source_aliases", {}) or {}
    aliases.update(user)
    return aliases


def resolve_pose_keys(task_cfg) -> set[str]:
    """Cfg keys whose source is (T, 7) axis-angle and needs 7 -> 9 conversion.

    In VRR mode the action is precomputed 19-D, so `action` is dropped from the
    default set. A `pose_keys:` list in the task yaml replaces the set entirely.
    """
    if "pose_keys" in task_cfg:
        return set(task_cfg.pose_keys)
    defaults = set(DEFAULT_POSE_KEYS)
    if task_cfg.get("action_type") == VRR_ACTION_TYPE:
        defaults.discard("action")
    return defaults


def axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
    R = Rotation.from_rotvec(axis_angle.reshape(-1, 3)).as_matrix()
    rot6d = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1)
    return rot6d.reshape(*axis_angle.shape[:-1], 6)


def pose7_to_pose9(arr: np.ndarray) -> np.ndarray:
    assert arr.ndim == 2 and arr.shape[1] == 7, f"expected (T, 7), got {arr.shape}"
    xyz = arr[:, 0:3]
    rot6d = axis_angle_to_rot6d(arr[:, 3:6])
    return np.concatenate([xyz, rot6d], axis=1).astype(np.float32)


def collect_keys(task_cfg) -> list[tuple[str, str]]:
    """Return [(key, type), ...] from shape_meta.obs, extended_obs, action."""
    sm = task_cfg.shape_meta
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(name: str, type_: str):
        if name in seen or "wrt" in name:
            return
        seen.add(name)
        out.append((name, type_))

    for section in ("obs", "extended_obs"):
        block = sm.get(section, {}) or {}
        for k, attr in block.items():
            add(k, (attr.get("type") or "low_dim"))

    if "action" in sm:
        add("action", "low_dim")

    return out


def source_file_for(key: str, aliases: dict[str, str]) -> str:
    return aliases.get(key, f"{key}.npy")


def load_key(ep_dir: Path, key: str, type_: str,
             aliases: dict[str, str], pose_keys: set[str]) -> np.ndarray:
    fname = source_file_for(key, aliases)
    path = ep_dir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"episode {ep_dir.name!r}: missing {fname} (required by cfg key {key!r})"
        )
    arr = np.load(path)
    if type_ == "rgb":
        if not (arr.ndim == 4 and arr.shape[-1] == 3 and arr.dtype == np.uint8):
            raise ValueError(
                f"{ep_dir.name}/{fname}: rgb expected (T,H,W,3) uint8, "
                f"got {arr.shape} {arr.dtype}"
            )
        arr = arr[..., ::-1]  # BGR -> RGB
        return arr
    if key in pose_keys:
        return pose7_to_pose9(arr)            # already float32
    return arr.astype(np.float32, copy=False)  # low_dim passthrough: enforce float32


def load_episode(ep_dir: Path, keys: Iterable[tuple[str, str]],
                 aliases: dict[str, str], pose_keys: set[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    ref_T: int | None = None
    ref_key: str | None = None
    for key, type_ in keys:
        arr = load_key(ep_dir, key, type_, aliases, pose_keys)
        T = arr.shape[0]
        if ref_T is None:
            ref_T, ref_key = T, key
        elif T != ref_T:
            raise ValueError(
                f"episode {ep_dir.name!r}: T mismatch — "
                f"{key}: {T} vs {ref_key}: {ref_T}"
            )
        out[key] = arr
    return out


def build_chunks_and_compressors(
    sample: dict[str, np.ndarray],
    key_types: dict[str, str],
    chunk_frames: int,
) -> tuple[dict[str, tuple], dict[str, Blosc]]:
    img_cpr = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    low_cpr = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    chunks: dict[str, tuple] = {}
    cprs: dict[str, Blosc] = {}
    for k, arr in sample.items():
        is_rgb = key_types[k] == "rgb"
        n = chunk_frames if is_rgb else chunk_frames * 16
        chunks[k] = (n,) + arr.shape[1:]
        cprs[k] = img_cpr if is_rgb else low_cpr
    return chunks, cprs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True,
                   help="Root containing one subdir per episode.")
    p.add_argument("--task", type=Path, required=True,
                   help="Task YAML with shape_meta.")
    p.add_argument("--dst", type=Path, required=False, default=Path("data/replay_buffers"),
                   help="Output dir; replay_buffer.zarr is written inside.")
    p.add_argument("--overwrite", action="store_true",
                   help="Delete dst/replay_buffer.zarr if it exists.")
    p.add_argument("--chunk-frames", type=int, default=128,
                   help="rgb chunk frames (low-dim uses 16x this).")
    p.add_argument("--no-derive", action="store_true",
                   help="Skip the prepare_vrr step in VRR mode (use vrr_action.npy as-is).")
    args = p.parse_args()

    task_cfg = OmegaConf.load(args.task)
    key_specs = collect_keys(task_cfg)
    key_types = {k: t for k, t in key_specs}
    if not key_specs:
        raise SystemExit(f"No keys found in {args.task} (shape_meta.{{obs,extended_obs,action}}).")
    aliases = resolve_source_aliases(task_cfg)
    pose_keys = resolve_pose_keys(task_cfg)
    action_type = task_cfg.get("action_type", None)
    print(f"action_type: {action_type or '(unset)'}")

    # In VRR mode, derive per-episode vrr_action.npy from raw signals first,
    # using the task cfg's `vrr:` block. Skip with --no-derive if you've
    # already run prepare_vrr externally (e.g. with non-default CLI flags).
    if action_type == VRR_ACTION_TYPE and not args.no_derive:
        from trainflow.common.prepare_vrr import load_vrr_params, run_for_src
        print(f"\n[VRR] auto-running prepare_vrr on {args.src} (use --no-derive to skip)")
        run_for_src(args.src, load_vrr_params(args.task))
        print("[VRR] prepare_vrr done\n")


    print("Keys to write:")
    for k, t in key_specs:
        flags = []
        if k in pose_keys: flags.append("pose7->9")
        if t == "rgb":     flags.append("BGR->RGB")
        flag_str = f"  [{','.join(flags)}]" if flags else ""
        print(f"  {k:24s}  type={t}  source={source_file_for(k, aliases)}{flag_str}")

    episode_dirs = sorted(d for d in args.src.iterdir() if d.is_dir())
    if not episode_dirs:
        raise SystemExit(f"No episode subdirectories found in {args.src}")
    print(f"Found {len(episode_dirs)} episodes in {args.src}")

    task_name = task_cfg.get("name", args.task.stem)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.dst / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = out_dir / "replay_buffer.zarr"
    if zarr_path.exists():
        if not args.overwrite:
            raise SystemExit(f"{zarr_path} already exists. Re-run with --overwrite.")
        shutil.rmtree(zarr_path)

    store = zarr.DirectoryStore(str(zarr_path))
    rb = ReplayBuffer.create_empty_zarr(storage=store)

    chunks: dict[str, tuple] | None = None
    cprs: dict[str, Blosc] | None = None
    for ep_dir in tqdm(episode_dirs, desc="episodes"):
        data = load_episode(ep_dir, key_specs, aliases, pose_keys)
        if chunks is None:
            chunks, cprs = build_chunks_and_compressors(
                data, key_types, args.chunk_frames
            )
        rb.add_episode(data, chunks=chunks, compressors=cprs)

    yaml = out_dir / "dataset_info.yaml"
    write_yaml(yaml, task_name, stamp, args, key_specs, rb, cprs, aliases, pose_keys)

    print(f"\nWrote {zarr_path}")
    print(f"  total frames: {rb.n_steps}")
    print(f"  episodes:     {rb.n_episodes}")
    for k in rb.data:
        arr = rb.data[k]
        print(f"  {k:24s}  {tuple(arr.shape)}  {arr.dtype}")
    print(f"  yaml file:      {yaml}")


def write_yaml(path, task_name, stamp, args, key_specs, rb, cprs, aliases, pose_keys):
    arrays = {}
    for k, t in key_specs:
        arr = rb.data[k]
        steps = []
        if t == "rgb":
            steps.append("BGR -> RGB channel swap")
        if k in pose_keys:
            steps.append("axis-angle (T,7) -> rot6d (T,9) float32 (gripper dropped)")
        cpr = cprs.get(k) if cprs else None
        arrays[k] = {
            "type": t,
            "source_file": source_file_for(k, aliases),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "chunks": list(arr.chunks),
            "compressor": (
                f"Blosc(cname={cpr.cname}, clevel={cpr.clevel}, shuffle={cpr.shuffle})"
                if cpr is not None else None
            ),
            "preprocessing": steps or ["passthrough (raw from .npy)"],
        }

    info = {
        "task_name": task_name,
        "created": stamp,
        "task_cfg": str(args.task.resolve()),
        "source_dir": str(args.src.resolve()),
        "n_episodes": int(rb.n_episodes),
        "n_steps": int(rb.n_steps),
        "structure": {
            "data/<key>": "per-modality array, leading dim = total frames",
            "meta/episode_ends": "cumulative frame index where each episode ends (int64)",
        },
        "arrays": arrays,
    }
    OmegaConf.save(OmegaConf.create(info), path)


if __name__ == "__main__":
    main()
