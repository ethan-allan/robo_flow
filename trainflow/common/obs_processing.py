"""Shared per-key preprocessing ops.

Single source of truth for the deterministic transforms applied to raw
per-episode data, used by both:

  * the training-side writer (`trainflow.common.zarr_writer`, which converts
    per-episode `.npy` files into the bundled replay-buffer zarr), and
  * the training-side action preprocessor (`trainflow.common.prepare_vrr`,
    which derives the 19-D VRR action from raw `eef_state` + `eef_force`).

The deployment env will import the same ops so that the tensors fed to the
policy online are bit-identical to what the dataset returned during training.

Keep this module free of zarr / hydra / file-system concerns — every function
takes numpy arrays and returns numpy arrays.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from trainflow.common.space_utils import pose_3d_9d_to_homo_matrix_batch


# ---------------------------------------------------------------------------
# Pose conversions
# ---------------------------------------------------------------------------

def axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 6) ortho6d (first two columns of R)."""
    R = Rotation.from_rotvec(axis_angle.reshape(-1, 3)).as_matrix()
    rot6d = np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1)
    return rot6d.reshape(*axis_angle.shape[:-1], 6)


def pose7_to_pose9(arr: np.ndarray) -> np.ndarray:
    """(..., 7) [xyz, axis-angle] -> (..., 9) [xyz, ortho6d] float32.

    Drops col 6 (gripper) — it is not part of the 9-D pose representation.
    The deployment-side inverse is `pose9_to_pose6`; gripper is sourced
    separately by the action executor. Accepts either a single (7,) frame
    or a batch (T, 7) — needed because the deploy obs_processor calls this
    per-frame while zarr_writer calls it batched.
    """
    assert arr.shape[-1] == 7, f"expected last dim 7, got {arr.shape}"
    xyz = arr[..., 0:3]
    rot6d = axis_angle_to_rot6d(arr[..., 3:6])
    return np.concatenate([xyz, rot6d], axis=-1).astype(np.float32)


def rot6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) ortho6d -> (..., 3) axis-angle (rotvec). Inverse of axis_angle_to_rot6d."""
    flat = rot6d.reshape(-1, 6)
    a, b = flat[:, :3], flat[:, 3:]
    e1 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_proj = b - (e1 * b).sum(-1, keepdims=True) * e1
    e2 = b_proj / np.linalg.norm(b_proj, axis=-1, keepdims=True)
    e3 = np.cross(e1, e2)
    R = np.stack([e1, e2, e3], axis=-1)
    rotvec = Rotation.from_matrix(R).as_rotvec()
    return rotvec.reshape(*rot6d.shape[:-1], 3)


def pose9_to_pose6(arr: np.ndarray) -> np.ndarray:
    """(..., 9) [xyz, ortho6d] -> (..., 6) [xyz, axis-angle] float32.

    Inverse of `pose7_to_pose9`'s rotation half. The dropped gripper channel
    is not recoverable here; the action executor sources it separately.
    Accepts either (9,) per-frame or (T, 9) batched input.
    """
    assert arr.shape[-1] == 9, f"expected last dim 9, got {arr.shape}"
    xyz = arr[..., 0:3]
    axis_angle = rot6d_to_axis_angle(arr[..., 3:9])
    return np.concatenate([xyz, axis_angle], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

def bgr_to_rgb(arr: np.ndarray) -> np.ndarray:
    """(..., 3) BGR -> RGB via channel reverse. View, not copy."""
    return arr[..., ::-1]


def concat(arrs, axis: int = -1) -> np.ndarray:
    """Concatenate a sequence of arrays/scalars along `axis` (default -1,
    the feature axis). Each input is wrapped via `np.atleast_1d` and
    `np.asarray`, so a mix of shape `(D,)` arrays and python scalars
    fans into a single 1-D output.

    Used by `obs_sources` blocks whose `sensor:` field is a list of
    dotted paths (multi-input). The first op in the chain typically
    runs `concat` to fuse the inputs into a single array."""
    return np.concatenate(
        [np.atleast_1d(np.asarray(a)) for a in arrs], axis=axis
    )


def crop_x(img: np.ndarray, x0: int, x1: int) -> np.ndarray:
    """Slice horizontal pixel range and drop alpha if present.

    Matches `record_data_gui_new.py:1697,1705` — `img[:, x0:x1, :3]`.
    Accepts either a single (H, W, C) frame or batched (T, H, W, C).
    """
    return img[..., :, x0:x1, :3]


def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """cv2.resize with INTER_AREA, matching `record_data_gui_new.py:1702`.

    `size` is (width, height) per cv2 convention. Operates on a single
    (H, W, C) frame; for batched inputs, call per-frame.
    """
    h, w = size[1], size[0]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Force / wrench
# ---------------------------------------------------------------------------

def moving_average_1d(data: np.ndarray, window: int) -> np.ndarray:
    """Per-column same-mode moving average. window<=1 is a passthrough copy."""
    if window <= 1:
        return data.copy()
    kernel = np.ones(window) / window
    out = np.empty_like(data)
    for d in range(data.shape[1]):
        out[:, d] = np.convolve(data[:, d], kernel, mode="same")
    return out


def tare_force(force: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Subtract mean of first n_frames; returns (tared_force, baseline).

    n_frames<=0 is a passthrough with a zero baseline.
    """
    if n_frames <= 0:
        return force.copy(), np.zeros(force.shape[1], dtype=force.dtype)
    n = min(n_frames, len(force))
    baseline = force[:n].mean(axis=0)
    return force - baseline, baseline


# ---------------------------------------------------------------------------
# VRR action helpers
# ---------------------------------------------------------------------------

def compute_stiffness(
    wrench_xyz: np.ndarray,
    k_max: float,
    k_min: float,
    f_low: float,
    f_high: float,
) -> np.ndarray:
    """Piecewise-linear stiffness in |F_xyz|: k_max below f_low, k_min above f_high."""
    mag = np.linalg.norm(wrench_xyz, axis=-1)
    interp = k_max - (k_max - k_min) * (mag - f_low) / (f_high - f_low)
    k = np.where(mag < f_low, k_max,
                 np.where(mag > f_high, k_min, interp))
    return k[:, None].astype(np.float32)


def compute_global_offset(local_offset: np.ndarray, tcp_pose_9d: np.ndarray) -> np.ndarray:
    """Rotate local force/stiffness offset into the world frame using tcp pose9d."""
    H = pose_3d_9d_to_homo_matrix_batch(tcp_pose_9d.astype(np.float32))
    R = H[:, :3, :3]
    return np.einsum("tij,tj->ti", R, local_offset)
