"""Visualize one captured episode from a raw `.npy` episode directory.

Renders the per-frame rgb observation(s) into an mp4 and prints a
summary of every per-frame `.npy` (shape, dtype, value range). Useful as
a quick sanity check after recording with `capture_demo.py`.

Usage:
    # point at a single episode dir:
    python -m demo_capture.visualize_episode data/ee2dice_raw/episode_0000

    # or a raw root + episode index (-1 = most recent):
    python -m demo_capture.visualize_episode data/ee2dice_raw --episode -1 \\
        [--output /tmp/episode.mp4] [--fps 10] [--rgb-key rgb] [--no-bgr-swap]

If the episode contains multiple rgb arrays (e.g. workspace + wrist
camera, or tactile) they're tiled horizontally. On-disk rgb is BGR
(matching the capture stream + zarr_writer), so channels are swapped to
RGB for display by default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def _looks_like_rgb(arr: np.ndarray) -> bool:
    return arr.ndim == 4 and arr.shape[-1] == 3 and arr.dtype == np.uint8


def _resolve_episode_dir(path: Path, episode: int) -> Path:
    """`path` is either an episode dir (contains .npy files) or a raw root
    (contains episode_* subdirs). Returns the chosen episode dir."""
    if any(path.glob("*.npy")):
        return path
    subdirs = sorted(
        d for d in path.iterdir() if d.is_dir() and d.name.startswith("episode_")
    )
    if not subdirs:
        raise FileNotFoundError(
            f"{path} has no .npy files and no episode_* subdirectories"
        )
    idx = episode if episode >= 0 else len(subdirs) + episode
    if not 0 <= idx < len(subdirs):
        raise IndexError(
            f"episode {episode} out of range ({len(subdirs)} episodes in {path})"
        )
    return subdirs[idx]


def _load_episode(ep_dir: Path) -> dict[str, np.ndarray]:
    return {f.stem: np.load(f) for f in sorted(ep_dir.glob("*.npy"))}


def _print_summary(ep: dict) -> None:
    keys = sorted(ep.keys())
    Tset = {ep[k].shape[0] for k in keys}
    print(f"frames: {Tset.pop() if len(Tset) == 1 else sorted(Tset)}")
    for k in keys:
        v = ep[k]
        line = f"  {k:18s} shape={tuple(v.shape)} dtype={v.dtype}"
        if v.ndim == 2 and v.dtype.kind in "fi":
            mn = np.round(v.min(axis=0), 4).tolist()
            mx = np.round(v.max(axis=0), 4).tolist()
            line += f"  min={mn}  max={mx}"
        print(line)


def _episode_fps(ep_dir: Path, default: float) -> float:
    meta_path = ep_dir / "meta.yaml"
    if meta_path.exists():
        try:
            meta = OmegaConf.load(meta_path)
            return float(meta.get("fps", default))
        except Exception:
            pass
    return default


def _tile_horizontal(frames: list[np.ndarray]) -> np.ndarray:
    """All frames must share T. If they differ in H, pad to max with 0;
    if W differs, hstack as-is (output H stays uniform)."""
    H_max = max(f.shape[1] for f in frames)
    padded = []
    for f in frames:
        T, H, W, C = f.shape
        if H == H_max:
            padded.append(f)
            continue
        pad = np.zeros((T, H_max - H, W, C), dtype=f.dtype)
        padded.append(np.concatenate([f, pad], axis=1))
    return np.concatenate(padded, axis=2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("path", type=Path,
                    help="An episode dir (episode_NNNN/) or a raw root "
                         "containing episode_* subdirs.")
    ap.add_argument("--episode", type=int, default=-1,
                    help="Episode index when `path` is a raw root. Negative "
                         "counts from the end (default -1 = most recent).")
    ap.add_argument("--output", type=Path, default=None,
                    help="MP4 output path. Default: <episode_dir>/preview.mp4.")
    ap.add_argument("--fps", type=float, default=None,
                    help="Playback fps. Default: meta.yaml fps, else 10.")
    ap.add_argument("--rgb-key", default=None,
                    help="Render only this key (the .npy stem). Default: every "
                         "(T,H,W,3) uint8 array in the episode.")
    ap.add_argument("--no-bgr-swap", action="store_true",
                    help="Don't swap channels before encoding. By default the "
                         "on-disk rgb is BGR and we convert to RGB.")
    ap.add_argument("--summary-only", action="store_true",
                    help="Print the key/shape summary and exit; no mp4.")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"[fatal] path not found: {args.path}", file=sys.stderr)
        return 2

    try:
        ep_dir = _resolve_episode_dir(args.path, args.episode)
    except (FileNotFoundError, IndexError) as e:
        print(f"[fatal] {e}", file=sys.stderr)
        return 2

    ep = _load_episode(ep_dir)
    if not ep:
        print(f"[fatal] no .npy files in {ep_dir}", file=sys.stderr)
        return 2

    print(f"=== {ep_dir} ===")
    _print_summary(ep)

    if args.summary_only:
        return 0

    if args.rgb_key is not None:
        if args.rgb_key not in ep:
            print(f"[fatal] --rgb-key {args.rgb_key!r} not in episode "
                  f"(keys: {sorted(ep.keys())})", file=sys.stderr)
            return 2
        rgb_keys = [args.rgb_key]
    else:
        rgb_keys = [k for k, v in ep.items() if _looks_like_rgb(v)]

    if not rgb_keys:
        print("[info] no rgb arrays found; nothing to render. "
              "Re-run with --summary-only or pick --rgb-key.")
        return 0

    frames_per_key = []
    for k in rgb_keys:
        arr = ep[k]
        if not args.no_bgr_swap:
            arr = arr[..., ::-1]            # BGR -> RGB
        frames_per_key.append(np.ascontiguousarray(arr))
    video = _tile_horizontal(frames_per_key) if len(frames_per_key) > 1 \
        else frames_per_key[0]

    fps = args.fps if args.fps is not None else _episode_fps(ep_dir, 10.0)
    out_path = args.output or (ep_dir / "preview.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio                  # noqa: F401
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264",
        quality=8, macro_block_size=1,
    )
    try:
        for frame in video:
            writer.append_data(frame)
    finally:
        writer.close()

    H, W = video.shape[1], video.shape[2]
    print(f"[done] wrote {video.shape[0]} frames ({H}x{W}) @ {fps}fps -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
