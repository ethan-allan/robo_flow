"""UR3 robot client — RTDE state read + Robotiq gripper width.

Live mode wraps `rtde_receive.RTDEReceiveInterface` (RTDE port, default 30004).
Gripper width is read from the same JSON file the capture script writes
(`record_data_gui_new.py:read_gripper_state`, line 770). This keeps live
deploy bit-comparable to recorded `.npy` episodes without introducing a
parallel wire-level protocol.

Replay mode reads `eef_state.npy` (T, 7) [xyz, axis-angle, gripper] and
adjacent state arrays from one episode dir; `idx_ref[0]` selects frame.

The client publishes raw fields only (`tcp_pose6`, `gripper_width`,
`eef_force`, ...). Concats are expressed declaratively in the task
yaml via `obs_sources.<key>.sensor: [robot.ur3.tcp_pose6,
robot.ur3.gripper_width]` plus a `concat` op — no per-task virtual
keys live in this client.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from .base_client import BaseSensorClient


class UR3Client(BaseSensorClient):
    # -- replay -------------------------------------------------------------

    @classmethod
    def from_npy_replay(
        cls,
        cfg: DictConfig,
        episode_dir: Path,
        idx_ref: list[int],
        **kwargs: Any,
    ) -> "UR3Client":
        self = cls.__new__(cls)
        BaseSensorClient.__init__(self, cfg)
        self._replay_mode = True
        self._idx_ref = idx_ref
        ep = Path(episode_dir)

        self._eef_state = np.load(ep / "eef_state.npy")           # (T, 7)
        self._eef_force = np.load(ep / "eef_force.npy")           # (T, 6)
        self._joint_state = np.load(ep / "joint_state.npy")       # (T, 7)
        self._current = np.load(ep / "current.npy")               # (T, 6)
        self._control_mode = np.load(ep / "control_mode.npy")     # (T, 1)
        self._frame_ts = np.load(ep / "frame_timestamp.npy")      # (T,)
        if self._eef_state.shape[1] != 7:
            raise ValueError(
                f"{ep.name}: eef_state expected (T,7), got {self._eef_state.shape}"
            )
        return self

    # -- live ---------------------------------------------------------------

    def start(self) -> None:
        """Open RTDE receive interface + record gripper-JSON path. Live mode only."""
        if self._replay_mode:
            return
        from rtde_receive import RTDEReceiveInterface  # heavy import

        host = str(self.cfg.host)
        port = int(self.cfg.get("rtde_port", 30004))
        # rtde_receive's RTDEReceiveInterface uses a fixed port internally;
        # the cfg port is exposed for diagnostics scripts that may probe
        # connectivity at a custom port before this call.
        self._rtde_r = RTDEReceiveInterface(host)
        self._control_mode_pin = int(self.cfg.get("control_mode_pin", 7))
        self._gripper_state_path = str(
            self.cfg.get("gripper_state_path", "/tmp/ur_gripper_state.json")
        )

    def stop(self) -> None:
        if self._replay_mode:
            return
        try:
            self._rtde_r.disconnect()
        except Exception:
            pass

    def _read_gripper(self) -> float:
        """Mirror record_data_gui_new.py:read_gripper_state. Returns 0.0
        on any error so a missing JSON file doesn't break the obs path."""
        try:
            with open(self._gripper_state_path, "r") as f:
                return float(json.load(f)["position"])
        except Exception:
            return 0.0

    # -- public -------------------------------------------------------------

    def get_latest(self) -> dict[str, Any]:
        if self._replay_mode:
            i = self._idx_ref[0]
            eef7 = self._eef_state[i]
            # Pass through raw on-disk dtype (float64 for legacy episodes).
            # Downstream ops (pose7_to_pose9 etc.) cast to float32 at the
            # end. Casting here would break bit-equality with zarr_writer's
            # batched path.
            return {
                "tcp_pose6": eef7[:6],
                "gripper_width": np.atleast_1d(eef7[6]),
                "eef_force": self._eef_force[i],
                "joint_state": self._joint_state[i],
                "current": self._current[i],
                "control_mode": self._control_mode[i],
                "ts": float(self._frame_ts[i]),
            }

        # Live. Keep raw float64 — same reasoning as replay.
        tcp_pose6 = np.asarray(self._rtde_r.getActualTCPPose(), dtype=np.float64)
        eef_force = np.asarray(self._rtde_r.getActualTCPForce(), dtype=np.float64)
        joint6 = np.asarray(self._rtde_r.getActualQ(), dtype=np.float64)
        current = np.asarray(self._rtde_r.getActualCurrent(), dtype=np.float64)
        gripper_width = np.atleast_1d(np.float64(self._read_gripper()))
        joint_state = np.concatenate([joint6, gripper_width])     # (7,)
        try:
            cm_bool = bool(self._rtde_r.getDigitalOutState(self._control_mode_pin))
        except Exception:
            cm_bool = False
        control_mode = np.atleast_1d(np.float64(cm_bool))         # (1,)
        return {
            "tcp_pose6": tcp_pose6,
            "gripper_width": gripper_width,
            "eef_force": eef_force,
            "joint_state": joint_state,
            "current": current,
            "control_mode": control_mode,
            "ts": time.time(),
        }
