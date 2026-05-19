"""Single-arm UR3 BIR deployment env.

Wires sensor clients to obs_builder and exposes `get_obs()` whose
output matches `RealImageTactileDataset.__getitem__` bit-for-bit on the
same frames. Action execution lives in step 6 (`action_executor.py`);
this module is read-only at deploy time.

Step 4 only commits to the replay-mode path
(`Ur3BirEnv.from_npy_replay`). Live mode raises until step 5+, but the
ring-buffer architecture is shared: in live mode, sensor threads will
push to the same `ObsRingBuffer` that the replay producer pushes to
here. `get_obs()` reads the buffer tail in either case.

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
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig
from omegaconf.errors import ConfigKeyError, ConfigAttributeError

from trainflow.common.obs_format import format_obs_window
from trainflow.env.ur3_bir.obs_builder import ObsBuilder
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

    A producer pushes per-frame entries (replay loop in offline mode,
    sensor threads in live mode); `Ur3BirEnv.get_obs()` reads the
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

def _normalise_parent(parent: str) -> str:
    """Mirror env._lookup's special cases. Keep in sync with
    `Ur3BirEnv._lookup`."""
    if parent.startswith("tactile.gelsight_"):
        return "tactile.gelsight"
    return parent


def _resolves(hw_cfg, dotted: str) -> bool:
    node = hw_cfg
    for seg in dotted.split("."):
        try:
            node = node[seg]
        except (KeyError, ConfigKeyError, ConfigAttributeError, TypeError):
            return False
    return True


def _from_paths(from_field) -> list[str]:
    if isinstance(from_field, str):
        return [from_field]
    if isinstance(from_field, (list, tuple, ListConfig)):
        return [str(s) for s in from_field]
    return []


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
    hw = task_cfg.hardware

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
            parent = _normalise_parent(path.rsplit(".", 1)[0])
            if not _resolves(hw, parent):
                errors.append(
                    f"shape_meta.obs.{cfg_key}: source parent {parent!r} (from path "
                    f"{path!r}) does not resolve into hardware cfg"
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
            missing = [i for i, v in enumerate(covered) if v is None]
            if missing:
                errors.append(
                    f"shape_meta.action.out: action indices {missing} not covered by any sink"
                )
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

def _client_factory(parent: str, cfg: DictConfig, ep_dir: Path | None,
                    idx_ref: list[int]) -> Any:
    """Map a sensor parent path to a client instance. Auditable in one place."""
    if parent == "robot.ur3":
        return UR3Client.from_npy_replay(cfg, ep_dir, idx_ref) if ep_dir else UR3Client(cfg)
    if parent == "cameras.platform_realsense":
        return (RealsenseClient.from_npy_replay(cfg, ep_dir, idx_ref, replay_key="rgb")
                if ep_dir else RealsenseClient(cfg))
    if parent == "cameras.hand_realsense":
        return (RealsenseClient.from_npy_replay(cfg, ep_dir, idx_ref, replay_key="rgb_hand")
                if ep_dir else RealsenseClient(cfg))
    if parent.startswith("tactile.gelsight_"):
        slot = int(parent.rsplit("_", 1)[1])
        return (GelsightClient.from_npy_replay(cfg, ep_dir, idx_ref, slot=slot)
                if ep_dir else GelsightClient(cfg, slot=slot))
    raise NotImplementedError(f"sensor parent {parent!r} not handled")


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
    """Replay-mode read-only env. Step 5/6 will add live mode + step()."""

    def __init__(
        self,
        task_cfg: DictConfig,
        n_obs_steps: int,
        obs_temporal_downsample_ratio: int = 1,
        episode_dir: Path | None = None,
        idx_ref: list[int] | None = None,
        buffer_capacity: int | None = None,
    ):
        assert_task_cfg(task_cfg)

        self.task_cfg = task_cfg
        self.hw = task_cfg.hardware
        self.shape_meta = task_cfg.shape_meta
        self.n_obs_steps = int(n_obs_steps)
        self.obs_downsample_ratio = int(obs_temporal_downsample_ratio)
        self.idx_ref = idx_ref if idx_ref is not None else [0]

        self._ep_dir = Path(episode_dir) if episode_dir is not None else None
        self._clients = self._build_clients()
        self._obs_builder = ObsBuilder(self.shape_meta.obs, self.hw)

        cap = buffer_capacity if buffer_capacity is not None else max(
            self.n_obs_steps * max(self.obs_downsample_ratio, 1) + 4, 32
        )
        self._buffer = ObsRingBuffer(capacity=cap)

        # Live-mode producer thread state. Replay mode never touches these.
        self._started = False
        self._producer_stop: threading.Event | None = None
        self._producer_thread: threading.Thread | None = None
        self._producer_errors: list[str] = []

    # -- construction helpers -------------------------------------------------

    def _build_clients(self) -> dict[str, Any]:
        parents: set[str] = set()
        for cfg_key, attr in self.shape_meta.obs.items():
            if "wrt" in cfg_key:
                continue
            from_field = attr["from"]
            paths = [from_field] if isinstance(from_field, str) else list(from_field)
            for p in paths:
                parents.add(str(p).rsplit(".", 1)[0])
        clients: dict[str, Any] = {}
        for parent in sorted(parents):
            cfg = self._lookup(parent)
            clients[parent] = _client_factory(parent, cfg, self._ep_dir, self.idx_ref)
        return clients

    def _lookup(self, dotted_path: str) -> Any:
        if dotted_path.startswith("tactile.gelsight_"):
            return self.hw.tactile.gelsight
        node = self.hw
        for seg in dotted_path.split("."):
            node = node[seg]
        return node

    # -- public API -----------------------------------------------------------

    @classmethod
    def from_npy_replay(
        cls,
        task_cfg: DictConfig,
        episode_dir: Path,
        n_obs_steps: int,
        obs_temporal_downsample_ratio: int = 1,
        idx_ref: list[int] | None = None,
    ) -> "Ur3BirEnv":
        return cls(
            task_cfg=task_cfg,
            n_obs_steps=n_obs_steps,
            obs_temporal_downsample_ratio=obs_temporal_downsample_ratio,
            episode_dir=Path(episode_dir),
            idx_ref=idx_ref,
        )

    def reset(self) -> None:
        """Clear the obs buffer. In replay mode also rewinds `idx_ref`
        to 0. In live mode, jog-to-home is the action executor's job
        (stage 6) — this method does NOT move the robot."""
        self._buffer.clear()
        if self._ep_dir is not None:
            self.idx_ref[0] = 0

    # -- live mode ----------------------------------------------------------

    def start(self) -> None:
        """Bring sensor clients up and spawn the producer thread. Live
        mode only — a replay env reads through `seek_replay/tick_replay`.

        Idempotent: a second start() is a no-op. Pairs with `stop()`."""
        if self._ep_dir is not None:
            raise RuntimeError(
                "Ur3BirEnv.start() is live-mode only; replay envs use "
                "tick_replay/seek_replay."
            )
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
        """Stop the producer thread, then stop each client in reverse
        startup order. Safe to call when not started."""
        if not self._started:
            return
        if self._producer_stop is not None:
            self._producer_stop.set()
        if self._producer_thread is not None:
            self._producer_thread.join(timeout=2.0)
        for parent in reversed(_start_order(list(self._clients.keys()))):
            try:
                self._clients[parent].stop()
            except Exception:
                pass
        self._started = False

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
                frame = self._obs_builder.build_frame(sensor_outputs)
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

    # -- replay mode --------------------------------------------------------

    def tick_replay(self) -> None:
        """Read each client at the current `idx_ref`, build one frame
        via `obs_builder`, push into the ring buffer. Replay-only."""
        if self._ep_dir is None:
            raise RuntimeError(
                "tick_replay is replay-only; live mode uses the producer "
                "thread started by start()."
            )
        sensor_outputs = {p: c.get_latest() for p, c in self._clients.items()}
        frame = self._obs_builder.build_frame(sensor_outputs)
        ts = self._infer_ts(sensor_outputs)
        self._buffer.push(ts, frame)

    def seek_replay(self, target_idx: int) -> None:
        """Bring the buffer up to date so its tail ends at `target_idx`."""
        if self._ep_dir is None:
            raise RuntimeError("seek_replay only valid in replay mode")
        self._buffer.clear()
        for i in range(target_idx + 1):
            self.idx_ref[0] = i
            self.tick_replay()

    def get_obs(self) -> dict[str, np.ndarray]:
        """Return the trailing `n_obs_steps` frames as a dict of
        (T, ...) arrays formatted to match
        `RealImageTactileDataset.__getitem__`'s `obs_dict` exactly.

        Reads from the obs ring buffer. The buffer must have been
        populated by `tick_replay` / `seek_replay` (replay) or by
        sensor threads (live, step 5+). The window-level formatting is
        the same shared function the dataset uses.
        """
        if len(self._buffer) == 0:
            raise RuntimeError(
                "ObsRingBuffer is empty — call seek_replay/tick_replay "
                "(replay) or start sensor threads (live) before get_obs."
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
