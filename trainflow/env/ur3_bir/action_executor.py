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

from trainflow.env.ur3_bir.obs_processor import OP_REGISTRY


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


# ---------------------------------------------------------------------------
# RTDE
# ---------------------------------------------------------------------------

class RtdeExecutor(ActionExecutor):
    """Decode each sink, apply safety clips, and dispatch motion via
    `RTDEControlInterface.servoL`. Today only sinks routed to
    `robot.ur3.tcp_command` are dispatched — other sinks (e.g. VRR's
    virtual_target / stiffness) are decoded for logging but consumed
    by impedance-control logic that is out of scope here.

    Safety clips applied per tick, before any servoL call:
      * `workspace_bbox`: hard-clip xyz into the box (or pass through)
      * `max_step_m` / `max_step_rad`: cap delta vs the last commanded
        target so a model glitch can't warp-jump
      * `force_abort_n`: if the optional `rtde_receive` injected at
        __init__ reports |F_xyz| above this, the executor aborts and
        refuses further dispatch until a new instance is constructed

    Cfg read:
      * `hw_cfg.clients.robot.ur3.host` — RTDE control endpoint
      * `hw_cfg.clients.robot.ur3.tcp_command.{kind,lookahead_time,gain,
         speed,acceleration,time}` — servoL kwargs
      * `hw_cfg.clients.robot.ur3.safety.*` — clip thresholds
    """

    def __init__(
        self,
        task_cfg: DictConfig,
        hw_cfg: DictConfig,
        rtde_receive: Any = None,
    ):
        super().__init__(task_cfg, hw_cfg)
        from rtde_control import RTDEControlInterface  # heavy

        ur3_cfg = hw_cfg.clients.robot.ur3
        tcp_cmd = ur3_cfg.tcp_command
        kind = str(tcp_cmd.get("kind", "rtde_servoL"))
        if kind != "rtde_servoL":
            raise ValueError(
                f"RtdeExecutor: only 'rtde_servoL' is supported; cfg "
                f"tcp_command.kind = {kind!r}"
            )

        self._rtde_c = RTDEControlInterface(str(ur3_cfg.host))
        self._rtde_r = rtde_receive

        self._lookahead_time = float(tcp_cmd.get("lookahead_time", 0.1))
        self._gain = float(tcp_cmd.get("gain", 300))
        self._speed = float(tcp_cmd.get("speed", 0.05))
        self._acceleration = float(tcp_cmd.get("acceleration", 0.5))
        self._servo_time = float(tcp_cmd.get("time", 0.2))

        safety = ur3_cfg.get("safety", {}) or {}
        self._max_step_m = float(safety.get("max_step_m", 0.01))
        self._max_step_rad = float(safety.get("max_step_rad", 0.05))
        bbox = safety.get("workspace_bbox", None)
        self._workspace_bbox = (
            None if bbox is None else [float(v) for v in bbox]
        )
        self._force_abort_n = float(safety.get("force_abort_n", 30.0))

        self._last_target: np.ndarray | None = None
        self._aborted = False

    def dispatch(self, action_vec: np.ndarray) -> dict[str, np.ndarray]:
        if self._aborted:
            raise RuntimeError(
                "RtdeExecutor is aborted; construct a new instance to retry"
            )

        # Force-abort check (cheap; do before decode so a runaway force
        # spike kills dispatch even if action decode is slow).
        if self._rtde_r is not None:
            f = np.asarray(self._rtde_r.getActualTCPForce(), dtype=np.float64)
            fmag = float(np.linalg.norm(f[:3]))
            if fmag > self._force_abort_n:
                self._abort_locally()
                raise RuntimeError(
                    f"RtdeExecutor: |F_xyz|={fmag:.2f}N exceeded "
                    f"force_abort_n={self._force_abort_n:.2f}N — aborted"
                )

        decoded = self._decode(np.asarray(action_vec))
        for sink in self._sinks:
            if sink.to != "robot.ur3.tcp_command":
                continue
            target = np.asarray(decoded[sink.name], dtype=np.float64)
            if target.ndim != 1 or target.shape[0] != 6:
                raise ValueError(
                    f"sink {sink.name!r} produced shape {target.shape}; "
                    f"expected (6,) pose6 after op chain"
                )
            target = self._apply_safety(target)
            self._rtde_c.servoL(
                target.tolist(),
                self._speed,
                self._acceleration,
                self._servo_time,
                self._lookahead_time,
                self._gain,
            )
            self._last_target = target.copy()
        return decoded

    def _apply_safety(self, target: np.ndarray) -> np.ndarray:
        out = target.copy()
        if self._workspace_bbox is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = self._workspace_bbox
            out[0] = float(np.clip(out[0], xmin, xmax))
            out[1] = float(np.clip(out[1], ymin, ymax))
            out[2] = float(np.clip(out[2], zmin, zmax))
        if self._last_target is not None:
            dxyz = out[:3] - self._last_target[:3]
            drot = out[3:6] - self._last_target[3:6]
            nxyz = float(np.linalg.norm(dxyz))
            nrot = float(np.linalg.norm(drot))
            if nxyz > self._max_step_m:
                dxyz = dxyz * (self._max_step_m / nxyz)
            if nrot > self._max_step_rad:
                drot = drot * (self._max_step_rad / nrot)
            out[:3] = self._last_target[:3] + dxyz
            out[3:6] = self._last_target[3:6] + drot
        return out

    def _abort_locally(self) -> None:
        self._aborted = True
        try:
            self._rtde_c.servoStop()
        except Exception:
            pass

    def stop(self) -> None:
        try:
            self._rtde_c.servoStop()
        except Exception:
            pass
        try:
            self._rtde_c.stopScript()
        except Exception:
            pass
        try:
            self._rtde_c.disconnect()
        except Exception:
            pass
