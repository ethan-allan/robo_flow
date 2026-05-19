"""Action executor: decodes a model-output action vector into per-sink
actuator commands. Reads `shape_meta.action.out` from the task cfg.

`ActionExecutor` is an ABC. Stage 5 ships `NoOpExecutor` which decodes
each sink's op chain and *records* the result but never touches the
robot. Stage 6 will add `RtdeExecutor` whose `dispatch` swaps the log
for `RTDEControlInterface.servoL(...)` + safety clips.

Both executors share the parsed sink table (slice ranges, op chains,
target hw path) so the runner doesn't need to care which one is wired.
"""
from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig

from trainflow.env.ur3_bir.obs_builder import OP_REGISTRY


# ---------------------------------------------------------------------------
# Sink table
# ---------------------------------------------------------------------------

@dataclass
class _Sink:
    name: str
    slice_lo: int
    slice_hi: int
    to: str
    ops: list[dict] = field(default_factory=list)


def _to_py(v: Any) -> Any:
    """Recursively convert omegaconf containers to plain python."""
    if isinstance(v, ListConfig):
        return [_to_py(x) for x in v]
    if isinstance(v, DictConfig):
        return {k: _to_py(x) for k, x in v.items()}
    return v


def _parse_sinks(action_block: DictConfig) -> list[_Sink]:
    sinks: list[_Sink] = []
    out = action_block.get("out", {}) or {}
    for name, snk in out.items():
        lo, hi = int(snk.slice[0]), int(snk.slice[1])
        ops = [_to_py(op) for op in snk.get("ops", [])]
        sinks.append(_Sink(name=str(name), slice_lo=lo, slice_hi=hi,
                           to=str(snk.to), ops=ops))
    return sinks


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class ActionExecutor(ABC):
    """Reads `task_cfg.shape_meta.action.out` to know how to slice +
    decode + dispatch an action vector. Subclasses override `dispatch`."""

    def __init__(self, task_cfg: DictConfig, hw_cfg: DictConfig):
        self.task_cfg = task_cfg
        self.hw_cfg = hw_cfg
        self._sinks = _parse_sinks(task_cfg.shape_meta.action)
        self.action_dim = int(task_cfg.shape_meta.action.shape[0])

    @property
    def sinks(self) -> list[_Sink]:
        return list(self._sinks)

    def _decode(self, action_vec: np.ndarray) -> dict[str, np.ndarray]:
        """Run each sink's op chain on its slice. Returns
        {sink_name: decoded_array}. Used by both NoOpExecutor and the
        future RtdeExecutor."""
        if action_vec.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_vec last-dim {action_vec.shape[-1]} != "
                f"declared action_dim {self.action_dim}"
            )
        decoded: dict[str, np.ndarray] = {}
        for sink in self._sinks:
            value: Any = action_vec[..., sink.slice_lo:sink.slice_hi]
            for op_cfg in sink.ops:
                op_name = op_cfg["name"]
                op_fn = OP_REGISTRY.get(op_name)
                if op_fn is None:
                    raise KeyError(
                        f"unknown op {op_name!r}; registered: {sorted(OP_REGISTRY)}"
                    )
                kwargs = {k: v for k, v in op_cfg.items() if k != "name"}
                value = op_fn(value, **kwargs)
            decoded[sink.name] = value
        return decoded

    @abstractmethod
    def dispatch(self, action_vec: np.ndarray) -> dict[str, np.ndarray]:
        """Decode + dispatch a (action_dim,) action vector. Returns the
        decoded per-sink dict so callers can introspect."""

    def stop(self) -> None:
        return None


# ---------------------------------------------------------------------------
# NoOp
# ---------------------------------------------------------------------------

class NoOpExecutor(ActionExecutor):
    """Decode each sink and record the result; do NOT command the robot.

    Used in stage 5 to validate the data + inference loop end-to-end.
    Stage 6's `RtdeExecutor` will replace this with `rtde_c.servoL(...)`
    + safety clips behind the same interface.

    Optional `log_path` writes one JSON line per dispatch — useful for
    `03_get_obs_live` and the final stage-5 verification script.
    """

    def __init__(
        self,
        task_cfg: DictConfig,
        hw_cfg: DictConfig,
        log_path: str | None = None,
    ):
        super().__init__(task_cfg, hw_cfg)
        self.last_dispatch: dict[str, np.ndarray] = {}
        self.dispatch_count: int = 0
        self._log_fp = None
        self._log_lock = threading.Lock()
        if log_path is not None:
            self._log_fp = open(log_path, "w", buffering=1)

    def dispatch(self, action_vec: np.ndarray) -> dict[str, np.ndarray]:
        decoded = self._decode(np.asarray(action_vec))
        self.last_dispatch = decoded
        self.dispatch_count += 1
        if self._log_fp is not None:
            row = {
                "t": time.time(),
                "step": self.dispatch_count,
                "sinks": {k: np.asarray(v).tolist() for k, v in decoded.items()},
            }
            with self._log_lock:
                self._log_fp.write(json.dumps(row) + "\n")
        return decoded

    def stop(self) -> None:
        if self._log_fp is not None:
            try:
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None
