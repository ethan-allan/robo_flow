"""Convert an ee2dice episode-directory dump into the trainflow replay-buffer zarr.
Example usage:
    python convert_ee2dice_to_zarr.py \
        --src /path/to/ee2dice/dumps/2024-05-01 \
        --dst /path/to/trainflow/data/ee2dice


Source layout (one directory per episode, named by timestamp):

    <src_root>/<timestamp>/
        eef_state.npy   # (T, 7) float64  [xyz(3), axis_angle(3), gripper(1)]
        eef_action.npy  # (T, 7) float64  same layout
        rgb.npy    # (T, 256, 256, 3) uint8

Output layout:

    <dst>/replay_buffer.zarr/
        meta/episode_ends   int64
        data/
            rgb_hand        uint8   (N, 256, 256, 3)
            eef_state       float32 (N, 9)   xyz + rot6d (gripper dropped)
            action          float32 (N, 9)   xyz + rot6d (gripper dropped)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 6) first-two-columns rot6d (Zhou et al. 2019)."""
    R = Rotation.from_rotvec(axis_angle.reshape(-1, 3)).as_matrix()
    rot6d = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1)
    return rot6d.reshape(*axis_angle.shape[:-1], 6)


def pose7_to_pose9(arr: np.ndarray) -> np.ndarray:
    """(T, 7) [xyz, axis_angle, gripper] -> (T, 9) [xyz, rot6d] float32."""
    assert arr.ndim == 2 and arr.shape[1] == 7, f"expected (T, 7), got {arr.shape}"
    xyz = arr[:, 0:3]
    rot6d = axis_angle_to_rot6d(arr[:, 3:6])
    return np.concatenate([xyz, rot6d], axis=1).astype(np.float32)

def bgr_to_rgb(arr: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) uint8 BGR -> RGB."""
    assert arr.ndim == 4 and arr.shape[-1] == 3, f"expected (T,H,W,3), got {arr.shape}"
    return arr[..., ::-1]

def load_episode(ep_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eef_state = np.load(ep_dir / "eef_state.npy")
    eef_action = np.load(ep_dir / "eef_action.npy")
    rgb = np.load(ep_dir / "rgb.npy")

    T = eef_state.shape[0]
    assert eef_action.shape[0] == T and rgb.shape[0] == T, (
        f"{ep_dir.name}: T mismatch (state={eef_state.shape[0]} "
        f"action={eef_action.shape[0]} rgb={rgb.shape[0]})"
    )
    assert rgb.dtype == np.uint8 and rgb.ndim == 4 and rgb.shape[-1] == 3, (
        f"{ep_dir.name}: rgb expected (T,H,W,3) uint8, got {rgb.shape} {rgb.dtype}"
    )


    return pose7_to_pose9(eef_state), pose7_to_pose9(eef_action), rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True,
                   help="Root containing one subdirectory per episode.")
    p.add_argument("--dst", type=Path, required=True,
                   help="Output directory; replay_buffer.zarr will be written inside.")
    p.add_argument("--overwrite", action="store_true",
                   help="Delete dst/replay_buffer.zarr if it exists.")
    p.add_argument("--chunk-frames", type=int, default=128,
                   help="Frames per chunk for image array (default: 128).")
    args = p.parse_args()

    episode_dirs = sorted(d for d in args.src.iterdir() if d.is_dir())
    if not episode_dirs:
        raise SystemExit(f"No episode subdirectories found in {args.src}")
    print(f"Found {len(episode_dirs)} episodes in {args.src}")

    args.dst.mkdir(parents=True, exist_ok=True)
    zarr_path = args.dst / "replay_buffer.zarr"
    if zarr_path.exists():
        if not args.overwrite:
            raise SystemExit(f"{zarr_path} already exists. Re-run with --overwrite.")
        import shutil
        shutil.rmtree(zarr_path)

    # First pass: peek shapes from episode 0 so we can size arrays without
    # holding the whole dataset in memory.
    s0, a0, i0 = load_episode(episode_dirs[0])
    H, W = i0.shape[1], i0.shape[2]
    state_dim = s0.shape[1]
    action_dim = a0.shape[1]
    print(f"Image: {H}x{W}x3 uint8  state: {state_dim}  action: {action_dim}")

    root = zarr.open(str(zarr_path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")

    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    img_compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    rgb_arr = data.create_dataset(
        "rgb",
        shape=(0, H, W, 3),
        chunks=(args.chunk_frames, H, W, 3),
        dtype="uint8",
        compressor=img_compressor,
    )
    state_arr = data.create_dataset(
        "eef_state",
        shape=(0, state_dim),
        chunks=(args.chunk_frames * 16, state_dim),
        dtype="float32",
        compressor=compressor,
    )
    action_arr = data.create_dataset(
        "action",
        shape=(0, action_dim),
        chunks=(args.chunk_frames * 16, action_dim),
        dtype="float32",
        compressor=compressor,
    )

    episode_ends: list[int] = []
    total = 0
    for ep_dir in tqdm(episode_dirs, desc="episodes"):
        s, a, img = load_episode(ep_dir)
        rgb_arr.append(img)
        state_arr.append(s)
        action_arr.append(a)
        total += s.shape[0]
        episode_ends.append(total)

    meta.create_dataset(
        "episode_ends",
        data=np.asarray(episode_ends, dtype=np.int64),
        chunks=(max(len(episode_ends), 1),),
        dtype="int64",
    )

    print(f"\nWrote {zarr_path}")
    print(f"  total frames: {total}")
    print(f"  episodes:     {len(episode_ends)}")
    print(f"  rgb:     {rgb_arr.shape} {rgb_arr.dtype}")
    print(f"  eef_state:    {state_arr.shape} {state_arr.dtype}")
    print(f"  action:       {action_arr.shape} {action_arr.dtype}")


if __name__ == "__main__":
    main()
