"""Shared per-window obs formatter.

Single source of truth for the temporal+formatting transforms applied
to a stacked obs window. Used by both:

  * `RealImageTactileDataset.__getitem__` (training, walks zarr-loaded
    arrays into batched tensors), and
  * `Ur3BirEnv.get_obs` (deployment, walks ring-buffered per-frame obs
    into the same shape).

Keeping this in one function means the dataset-side and deploy-side
final formatting cannot drift. Per-frame preprocessing (zarr_writer
ops at train time, ObsBuilder ops at deploy time) is upstream of this
function; this only handles the window-level work.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def format_obs_window(
    stacked: Mapping[str, np.ndarray],
    shape_meta_obs: Mapping[str, Mapping],
    n_obs_steps: int,
    obs_downsample_ratio: int,
) -> dict[str, np.ndarray]:
    """Apply T_slice + downsample + reverse + per-type formatting.

    Args:
        stacked: dict of (T_total, ...) per-key arrays. T_total must
            be >= n_obs_steps. The dataset side passes the sampler
            output directly (T_total = horizon); the deploy side
            passes a freshly-stacked obs window (T_total = n_obs_steps).
        shape_meta_obs: the `shape_meta.obs` block. Each entry must
            carry `shape: list[int]` and `type: "rgb" | "low_dim"`.
            Extra fields (`from`, `ops`, etc.) are ignored.
        n_obs_steps: number of trailing frames the model consumes.
        obs_downsample_ratio: stride through the trailing window.
            ratio=1 is identity.

    Returns:
        dict[key, ndarray]:
            - rgb keys -> (T, C, H, W) float32 in [0, 1]
            - low_dim keys -> (T, D) or (T, D1, D2) float32, truncated
              to shape_meta's declared shape.
        Keys containing 'wrt' are skipped (legacy convention).
    """
    out: dict[str, np.ndarray] = {}
    T_slice = slice(n_obs_steps)
    for key, attr in shape_meta_obs.items():
        if "wrt" in key:
            continue
        type_ = attr.get("type", "low_dim")
        target_shape = list(attr["shape"])
        arr = stacked[key]                                    # (T_total, ...)
        arr_t = arr[T_slice][::-obs_downsample_ratio][::-1]   # (n_obs_steps, ...)
        if type_ == "rgb":
            out[key] = np.moveaxis(arr_t, -1, 1).astype(np.float32) / 255.0
        elif type_ == "low_dim":
            if len(target_shape) == 1:
                out[key] = arr_t[:, : target_shape[0]].astype(np.float32)
            elif len(target_shape) == 2:
                out[key] = arr_t[
                    :, : target_shape[0], : target_shape[1]
                ].astype(np.float32)
            else:
                raise ValueError(
                    f"shape_meta.obs.{key}: shape {target_shape} not supported"
                )
        else:
            raise ValueError(
                f"shape_meta.obs.{key}: unknown type {type_!r}"
            )
    return out
