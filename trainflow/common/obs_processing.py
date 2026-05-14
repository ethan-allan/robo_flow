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
    """(T, 7) [xyz, axis-angle] -> (T, 9) [xyz, ortho6d] float32."""
    assert arr.ndim == 2 and arr.shape[1] == 7, f"expected (T, 7), got {arr.shape}"
    xyz = arr[:, 0:3]
    rot6d = axis_angle_to_rot6d(arr[:, 3:6])
    return np.concatenate([xyz, rot6d], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

def bgr_to_rgb(arr: np.ndarray) -> np.ndarray:
    """(..., 3) BGR -> RGB via channel reverse. View, not copy."""
    return arr[..., ::-1]


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
