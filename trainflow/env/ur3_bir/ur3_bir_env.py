"""Single-arm UR3 BIR deployment env.

Wires sensor clients to obs_processor and exposes `get_obs()` whose
output matches `RealImageTactileDataset.__getitem__` bit-for-bit on the
same frames. Action execution lives in step 6 (`action_executor.py`);
this module is read-only at deploy time.

This module also contains:
  * `ObsRingBuffer` — small enough that a separate file would be churn
  * `validate_task_cfg` — runs at __init__ to catch typos/gaps early

Window-level formatting (T_slice + downsample + reverse + per-type
moveaxis/divide/truncate) is delegated to
`trainflow.common.obs_format.format_obs_window` so train and deploy
cannot drift.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig
from omegaconf.errors import ConfigKeyError, ConfigAttributeError

from trainflow.common.obs_format import format_obs_window
from trainflow.env.ur3_bir.action_executor import ActionExecutor
from trainflow.env.ur3_bir.obs_processor import ObsProcessor
from trainflow.env.ur3_bir.sensor_clients import (
    GelsightClient,
    RealsenseClient,
    UR3Client,
)


# ---------------------------------------------------------------------------
# Obs ring buffer
# ---------------------------------------------------------------------------

class ObsRingBuffer:
    """Circular buffer of (ts, per-key obs dict).

    A producer pushes per-frame entries ; `Ur3BirEnv.get_obs()` reads the
    trailing `n_obs_steps` entries via `tail()`.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._frames: deque = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._frames)

    def push(self, ts: float, frame: dict[str, Any]) -> None:
        self._frames.append((ts, frame))

    def latest_ts(self) -> float | None:
        return self._frames[-1][0] if self._frames else None

    def clear(self) -> None:
        self._frames.clear()

    def tail(self, n: int) -> list[tuple[float, dict[str, Any]]]:
        """Most-recent n frames, chronological order. Pads by repeating
        the oldest entry if the buffer holds < n frames."""
        if not self._frames:
            raise RuntimeError("ObsRingBuffer is empty; nothing to return")
        frames = list(self._frames)
        if len(frames) < n:
            return [frames[0]] * (n - len(frames)) + frames
        return frames[-n:]


# ---------------------------------------------------------------------------
# Cfg validation
# ---------------------------------------------------------------------------

def _from_paths(from_field) -> list[str]:
    if isinstance(from_field, str):
        return [from_field]
    if isinstance(from_field, (list, tuple, ListConfig)):
        return [str(s) for s in from_field]
    return []


def _resolve_sensor_cfg(task_cfg: DictConfig, parent: str) -> DictConfig | None:
    """Return the cfg block for a sensor `parent` dotted path, or None.

    Two routes depending on prefix:
      * `robot.ur3` — shared connection cfg at `hardware.clients.robot.ur3`
        (composed once per env_cfg; every obs that reads `robot.ur3.*`
        uses the same client).
      * `cameras.<name>` / `tactile.gelsight_<n>` — per-sensor cfg lives
        on the matching `shape_meta.obs.<key>` entry. The entry's
        `client:` sub-block carries connection params; the surrounding
        fields (shape/type/from/ops) belong to obs processing. When
        multiple obs keys share a parent (e.g. rgb + depth on one
        camera), the first match wins.

    Returns the resolved cfg block, or None if the parent is unknown
    or the expected cfg is missing.
    """
    if parent == "robot.ur3":
        try:
            return task_cfg.hardware.clients.robot.ur3
        except (KeyError, ConfigKeyError, ConfigAttributeError):
            return None
    if parent.startswith("cameras.") or parent.startswith("tactile.gelsight_"):
        for cfg_key, attr in task_cfg.shape_meta.obs.items():
            if "wrt" in cfg_key:
                continue
            from_field = attr.get("from", None) if hasattr(attr, "get") else None
            if from_field is None:
                continue
            for path in _from_paths(from_field):
                if str(path).rsplit(".", 1)[0] == parent:
                    return attr
        return None
    return None


def validate_task_cfg(task_cfg: DictConfig) -> list[str]:
    """Return a list of human-readable error strings. Empty list = ok.

    Checks:
      * every `shape_meta.obs.<key>` declares `from` + `ops`; every
        `from` path resolves into hw cfg.
      * if `shape_meta.action.out` is present, every named sink declares
        `slice` + `to`, slices cover [0, action_dim) with no overlap or
        gap. `to` path resolution is deferred to the executor at runtime.
    """
    errors: list[str] = []
    sm = task_cfg.shape_meta

    # --- obs side ----------------------------------------------------------
    for cfg_key, attr in sm.obs.items():
        if "wrt" in cfg_key:
            continue
        if "from" not in attr:
            errors.append(f"shape_meta.obs.{cfg_key} missing required field `from`")
            continue
        if "ops" not in attr:
            errors.append(f"shape_meta.obs.{cfg_key} missing required field `ops` (use [] for none)")
        for path in _from_paths(attr["from"]):
            parent = str(path).rsplit(".", 1)[0]
            sensor_cfg = _resolve_sensor_cfg(task_cfg, parent)
            if sensor_cfg is None:
                errors.append(
                    f"shape_meta.obs.{cfg_key}: source parent {parent!r} (from path "
                    f"{path!r}) is not a recognised sensor parent or its cfg is "
                    f"unreachable (cameras/tactile expect a sibling shape_meta.obs "
                    f"entry; robot.ur3 expects hardware.clients.robot.ur3)"
                )
            elif (parent.startswith("cameras.") or parent.startswith("tactile.gelsight_")) \
                    and "client" not in sensor_cfg:
                errors.append(
                    f"shape_meta.obs.{cfg_key}: parent {parent!r} resolves but the "
                    f"shape_meta.obs entry has no `client:` sub-block (required to "
                    f"instantiate the sensor client)"
                )

    # --- action side -------------------------------------------------------
    action_block = sm.get("action", {}) or {}
    sinks = action_block.get("out", {}) or {}
    if sinks:
        if "shape" not in action_block:
            errors.append("shape_meta.action.shape missing; cannot validate action.out")
        else:
            action_dim = int(action_block.shape[0])
            covered: list[str | None] = [None] * action_dim
            for name, snk in sinks.items():
                if "slice" not in snk:
                    errors.append(f"shape_meta.action.out.{name}: missing slice")
                    continue
                lo, hi = int(snk.slice[0]), int(snk.slice[1])
                if lo < 0 or hi > action_dim or lo >= hi:
                    errors.append(
                        f"shape_meta.action.out.{name}: slice [{lo},{hi}) invalid for "
                        f"action dim {action_dim}"
                    )
                    continue
                for i in range(lo, hi):
                    if covered[i] is not None:
                        errors.append(
                            f"shape_meta.action.out.{name}: slice [{lo},{hi}) overlaps "
                            f"sink {covered[i]!r} at index {i}"
                        )
                        break
                    covered[i] = name
            # Note: full coverage is NOT required. Some action dims are
            # training-only targets (e.g. VRR's virtual_target + stiffness
            # at indices [9, 19) are consumed by the impedance-control
            # logic and never dispatched as actuator commands). Missing
            # coverage is fine; overlap and out-of-range slices are not.
            for name, snk in sinks.items():
                if "to" not in snk:
                    errors.append(f"shape_meta.action.out.{name}: missing `to` path")

    return errors


def assert_task_cfg(task_cfg: DictConfig) -> None:
    errs = validate_task_cfg(task_cfg)
    if errs:
        bullet = "\n  - ".join(errs)
        raise ValueError(f"task cfg validation failed:\n  - {bullet}")


# ---------------------------------------------------------------------------
# Sensor client factory
# ---------------------------------------------------------------------------


def _start_order(parents: list[str]) -> list[str]:
    """Stable client startup order. DM-Tac (when wired) must precede
    RealSense per capture-script note record_data_gui_new.py:375-379 —
    its closed-source SDK probes /dev/video* sequentially and can hang on
    a RealSense sub-node. Today no task references tactile.dmtac, so
    this is forward-compatibility only."""
    def key(p: str) -> tuple[int, str]:
        if p.startswith("tactile.dmtac"):
            return (0, p)
        return (1, p)
    return sorted(parents, key=key)


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class Ur3BirEnv:

    def __init__(
        self,
        task_cfg: DictConfig,
        n_obs_steps: int,
        obs_temporal_downsample_ratio: int = 1,
        buffer_capacity: int | None = None,
        executor: ActionExecutor | None = None,
    ):
        assert_task_cfg(task_cfg)

        self.task_cfg = task_cfg
        self.hw = task_cfg.hardware
        self.shape_meta = task_cfg.shape_meta
        self.n_obs_steps = int(n_obs_steps)
        self.obs_downsample_ratio = int(obs_temporal_downsample_ratio)

        self._clients = self._build_clients()
        self._obs_processor = ObsProcessor(self.shape_meta.obs, self.hw)
        self._executor = executor

        cap = buffer_capacity if buffer_capacity is not None else max(
            self.n_obs_steps * max(self.obs_downsample_ratio, 1) + 4, 32
        )
        self._buffer = ObsRingBuffer(capacity=cap)

        self._started = False
        self._producer_stop: threading.Event | None = None
        self._producer_thread: threading.Thread | None = None
        self._producer_errors: list[str] = []

    # -- construction helpers -------------------------------------------------

    def _build_clients(self) -> dict[str, Any]:
        """Instantiate one client per sensor parent referenced by
        `shape_meta.obs.<key>.from`. Returns {parent: client}; no
        hardware is touched here — that happens in `start()` via each
        client's `.start()`."""
        parents: set[str] = set()
        for cfg_key, attr in self.shape_meta.obs.items():
            if "wrt" in cfg_key:
                continue
            for path in _from_paths(attr["from"]):
                parents.add(str(path).rsplit(".", 1)[0])

        clients: dict[str, Any] = {}
        for parent in sorted(parents):
            cfg = _resolve_sensor_cfg(self.task_cfg, parent)
            if cfg is None:
                # validate_task_cfg should have caught this in __init__;
                # raise here is defensive.
                raise RuntimeError(f"no cfg for sensor parent {parent!r}")
            if parent.startswith("cameras."):
                clients[parent] = RealsenseClient(cfg.client)
            elif parent.startswith("tactile.gelsight_"):
                slot = int(parent.rsplit("_", 1)[1])
                clients[parent] = GelsightClient(cfg.client, slot=slot)
            elif parent == "robot.ur3":
                clients[parent] = UR3Client(cfg)
            else:
                raise ValueError(
                    f"unknown sensor parent {parent!r} — known prefixes: "
                    f"cameras.*, tactile.gelsight_*, robot.ur3"
                )
        return clients

    # -- public API -----------------------------------------------------------

    def reset(self) -> None:
        """Clear the obs buffer. Jog-to-home is the action executor's
        job (stage 6) — this method does NOT move the robot."""
        self._buffer.clear()

    # -- live mode ----------------------------------------------------------

    def start(self) -> None:
        """Bring sensor clients up and spawn the producer thread.

        Idempotent: a second start() is a no-op. Pairs with `stop()`."""
        if self._started:
            return

        for parent in _start_order(list(self._clients.keys())):
            self._clients[parent].start()

        self._producer_stop = threading.Event()
        self._producer_errors = []
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name="ur3bir-producer",
            daemon=True,
        )
        self._producer_thread.start()
        self._started = True

    def stop(self) -> None:
        """Stop the producer thread, the executor (if any), then each
        client in reverse startup order. Safe to call when not started."""
        if not self._started:
            return
        if self._producer_stop is not None:
            self._producer_stop.set()
        if self._producer_thread is not None:
            self._producer_thread.join(timeout=2.0)
        if self._executor is not None:
            try:
                self._executor.stop()
            except Exception:
                pass
        for parent in reversed(_start_order(list(self._clients.keys()))):
            try:
                self._clients[parent].stop()
            except Exception:
                pass
        self._started = False

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        """Dispatch a model-output action vector via the wired executor.
        Returns the per-sink decoded dict. Raises if no executor was
        passed to __init__."""
        if self._executor is None:
            raise RuntimeError(
                "Ur3BirEnv.step requires an executor; pass one to __init__ "
                "(e.g. NoOpExecutor for dry-run, RtdeExecutor for motion)"
            )
        return self._executor.dispatch(action)

    def _producer_loop(self) -> None:
        """Snapshot all clients at control_fps, build one frame, push to
        the ring buffer. Runs in its own thread; transient errors get
        appended to `self._producer_errors` instead of crashing the loop."""
        interval = 1.0 / float(self.hw.get("control_fps", 5))
        assert self._producer_stop is not None
        while not self._producer_stop.is_set():
            t0 = time.monotonic()
            try:
                sensor_outputs = {p: c.get_latest() for p, c in self._clients.items()}
                frame = self._obs_processor.build_frame(sensor_outputs)
                ts = self._infer_ts(sensor_outputs)
                self._buffer.push(ts, frame)
            except Exception as e:
                # Keep the loop alive — a sensor stall shouldn't kill the env.
                self._producer_errors.append(f"{type(e).__name__}: {e}")
            elapsed = time.monotonic() - t0
            sleep_s = interval - elapsed
            if sleep_s > 0:
                self._producer_stop.wait(timeout=sleep_s)

    @property
    def producer_errors(self) -> list[str]:
        """Most recent non-fatal errors from the producer thread.
        Diagnostic — empty in the happy path."""
        return list(self._producer_errors)

    def get_obs(self) -> dict[str, np.ndarray]:
        """Return the trailing `n_obs_steps` frames as a dict of
        (T, ...) arrays formatted to match
        `RealImageTactileDataset.__getitem__`'s `obs_dict` exactly.

        Reads from the obs ring buffer. The buffer must have been
        populated by sensor threads. The window-level formatting is
        the same shared function the dataset uses.
        """
        if len(self._buffer) == 0:
            raise RuntimeError(
                "ObsRingBuffer is empty — call start() to spawn the "
                "producer thread before get_obs."
            )

        n = self.n_obs_steps
        tail = self._buffer.tail(n)
        per_frame = [frame for (_ts, frame) in tail]
        keys = list(per_frame[0].keys())
        stacked = {k: np.stack([f[k] for f in per_frame], axis=0) for k in keys}

        return format_obs_window(
            stacked=stacked,
            shape_meta_obs=self.shape_meta.obs,
            n_obs_steps=n,
            obs_downsample_ratio=self.obs_downsample_ratio,
        )

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _infer_ts(sensor_outputs: dict[str, dict]) -> float:
        if "robot.ur3" in sensor_outputs and "ts" in sensor_outputs["robot.ur3"]:
            return float(sensor_outputs["robot.ur3"]["ts"])
        for parent, out in sensor_outputs.items():
            if "ts" in out:
                return float(out["ts"])
        return 0.0
