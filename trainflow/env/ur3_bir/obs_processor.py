"""Per-frame obs processor driven by the task yaml's `shape_meta.obs` block.

For each `shape_meta.obs.<key>`:
  1. Look up `from`. May be:
       - a single dotted path (e.g. `cameras.platform_realsense.rgb`),
         in which case the value is `clients[parent].get_latest()[leaf]`.
       - a list of dotted paths (multi-input fan-in, e.g.
         `[robot.ur3.tcp_pose6, robot.ur3.gripper_width]`), in which
         case the value is a python list of per-input arrays — typically
         consumed by a `concat` op as the first step.
  2. Run the declared op chain. Each op is looked up in OP_REGISTRY and
     called with the kwargs declared inline in the ops list. There is
     no hw-cfg fallback for op kwargs — bounds/sizes live next to the
     op in the task yaml so the dataflow is readable in one place.
  3. Result is the per-frame value the model would see IF this single
     frame were the dataset's per-frame array (i.e. matches
     `zarr_writer.load_key`'s output for that key).

Capture-time preprocessing skip:
    Each sensor block in the hw cfg may carry `applied_at_capture: [<op_name>, ...]`.
    When the obs_processor encounters an op whose name appears in that
    list for the primary sensor, the op is SKIPPED (input passes through
    unchanged). This is how the framework handles legacy datasets where
    some preprocessing ran in the capture script and is already present
    on disk.

Temporal stacking + final formatting (moveaxis/CHW, /255, shape
truncation, float32) lives in `Ur3BirEnv.get_obs` via the shared
`trainflow.common.obs_format.format_obs_window` — NOT here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig

from trainflow.common import obs_processing as ops_mod


OP_REGISTRY: dict[str, Any] = {
    "bgr_to_rgb": ops_mod.bgr_to_rgb,
    "concat": ops_mod.concat,
    "pose7_to_pose9": ops_mod.pose7_to_pose9,
    "pose9_to_pose6": ops_mod.pose9_to_pose6,
    "crop_x": ops_mod.crop_x,
    "resize": ops_mod.resize,
}


class ObsProcessor:
    """Stateless per-frame processor. Construct once per env, call
    `build_frame(sensor_outputs)` each tick."""

    def __init__(self, shape_meta_obs: DictConfig, hw_cfg: DictConfig):
        self._sm_obs = shape_meta_obs
        self._hw = hw_cfg

    # -- public ---------------------------------------------------------------

    def build_frame(self, sensor_outputs: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
        """sensor_outputs is keyed by sensor PARENT path (e.g.
        'cameras.platform_realsense'); values are that client's
        get_latest() dict. Returns {cfg_key: ndarray} per shape_meta.obs
        entry."""
        out: dict[str, np.ndarray] = {}
        for cfg_key, attr in self._sm_obs.items():
            if "wrt" in cfg_key:
                continue
            sensor_paths = self._normalise_from(attr["from"])
            primary_parent = sensor_paths[0].rsplit(".", 1)[0]

            if len(sensor_paths) == 1:
                value = self._read_one(sensor_paths[0], sensor_outputs, cfg_key)
            else:
                value = [self._read_one(p, sensor_outputs, cfg_key) for p in sensor_paths]

            for op_cfg in attr.get("ops", []):
                value = self._apply_op(value, op_cfg, primary_parent)
            out[cfg_key] = value
        return out

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _normalise_from(from_field) -> list[str]:
        if isinstance(from_field, str):
            return [from_field]
        if isinstance(from_field, (list, tuple, ListConfig)):
            return [str(s) for s in from_field]
        raise TypeError(f"shape_meta.obs.<key>.from must be str or list, got {type(from_field)}")

    @staticmethod
    def _read_one(path: str, sensor_outputs: dict, cfg_key: str):
        parent, leaf = path.rsplit(".", 1)
        try:
            return sensor_outputs[parent][leaf]
        except KeyError as e:
            raise KeyError(
                f"shape_meta.obs.{cfg_key}: source path {path!r} unresolved "
                f"(parent={parent!r}, leaf={leaf!r}); available parents="
                f"{list(sensor_outputs)}"
            ) from e

    def _apply_op(self, value, op_cfg: DictConfig, sensor_parent: str):
        op_name = op_cfg.name
        if op_name in self._applied_at_capture(sensor_parent):
            return value  # capture script / manifest declared this already done
        op_fn = OP_REGISTRY.get(op_name)
        if op_fn is None:
            raise KeyError(f"unknown op {op_name!r}; registered: {sorted(OP_REGISTRY)}")
        kwargs = {k: _to_py(v) for k, v in op_cfg.items() if k != "name"}
        return op_fn(value, **kwargs)

    def _applied_at_capture(self, sensor_parent: str) -> set[str]:
        sensor_node = self._lookup_dotted(sensor_parent)
        if sensor_node is None or "applied_at_capture" not in sensor_node:
            return set()
        return {str(name) for name in sensor_node.applied_at_capture}

    def _lookup_dotted(self, dotted_path: str):
        node = self._hw
        for seg in dotted_path.split("."):
            try:
                node = node[seg]
            except Exception:
                return None
        return node


def _to_py(v: Any) -> Any:
    """Recursively convert omegaconf containers to plain python."""
    if isinstance(v, ListConfig):
        return [_to_py(x) for x in v]
    if isinstance(v, DictConfig):
        return {k: _to_py(x) for k, x in v.items()}
    return v
