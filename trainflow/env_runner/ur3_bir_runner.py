"""Stage-5 UR3 BIR deployment runner.

Two threads:

  * **Inference** — every 1/inference_fps:
      1. `env.get_obs()` (ring-buffer tail, formatted to match training)
      2. policy.predict_action  (the policy handles its own normalizer)
      3. take `action_pred[latency_step : latency_step + n_action_steps]`
      4. replace the action queue with those rows.

  * **Control** — every 1/control_fps: pop one action from the queue
    and `executor.dispatch(action)`.

Deliberately *no* `EnsembleBuffer`: at our control_fps == inference_fps
(both 5 Hz in the lab default) the multi-rate split degenerates, and
upstream `_weighted_average_action` doesn't handle the 19-D VRR action
anyway. Newest-prediction-wins is enough; if jitter shows up at
Phase H we'll add per-sink low-pass filtering instead.

Action vector reaches the executor as a 1-D numpy array of length
`task_cfg.shape_meta.action.shape[0]`. The executor handles per-sink
slicing + op chain; the runner stays sink-agnostic.

Out of scope for stage 5:
  * `use_latent_action_with_rnn_decoder` / `use_reactive_transformer`
    (reactive task is composed but the runner path is deferred).
  * `rpy_for_rotation` re-expansion (model emits 13D for VRR; we
    currently smoke-test against real_ee2_dice which emits 9-D direct).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from trainflow.env.ur3_bir.action_executor import ActionExecutor
from trainflow.env.ur3_bir.ur3_bir_env import Ur3BirEnv


# ---------------------------------------------------------------------------
# Action queue
# ---------------------------------------------------------------------------

class _ActionQueue:
    """Lock-protected FIFO with newest-wins replacement. Inference
    thread calls `replace(actions)`; control thread calls `pop()`."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._q: deque = deque()

    def replace(self, actions: np.ndarray) -> None:
        with self._lock:
            self._q.clear()
            for a in actions:
                self._q.append(np.asarray(a))

    def pop(self) -> np.ndarray | None:
        with self._lock:
            if not self._q:
                return None
            return self._q.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._q)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class RunnerStats:
    inference_count: int = 0
    dispatch_count: int = 0
    starved_count: int = 0      # control ticks with empty queue
    inference_latency_s: float = 0.0   # last
    dispatch_latency_s: float = 0.0    # last
    errors: list[tuple[str, str]] = field(default_factory=list)


class Ur3BirRunner:
    """Drives a live `Ur3BirEnv` against a policy + executor.

    The runner does NOT own the env's sensor lifecycle in a "build for
    me" sense — the caller passes a constructed env, and `run()` calls
    `env.start()` / `env.stop()` for the duration. Policy + executor
    are similarly caller-owned (the runner only calls `.predict_action`,
    `.dispatch`, and `.stop`).
    """

    def __init__(
        self,
        env: Ur3BirEnv,
        policy: Any,
        executor: ActionExecutor,
        *,
        control_fps: float,
        inference_fps: float,
        n_action_steps: int,
        latency_step: int = 0,
        device: str = "cuda",
        warmup_timeout_s: float = 5.0,
    ):
        if control_fps <= 0 or inference_fps <= 0:
            raise ValueError("control_fps / inference_fps must be > 0")
        # Allow non-integer fps but require integer ratio so steps_per_inference is clean.
        ratio = control_fps / inference_fps
        if not float(ratio).is_integer():
            raise ValueError(
                f"control_fps ({control_fps}) must be an integer multiple of "
                f"inference_fps ({inference_fps})"
            )
        if n_action_steps < 1:
            raise ValueError("n_action_steps must be >= 1")
        if latency_step < 0:
            raise ValueError("latency_step must be >= 0")

        self.env = env
        self.policy = policy
        self.executor = executor
        self.control_fps = float(control_fps)
        self.inference_fps = float(inference_fps)
        self.n_action_steps = int(n_action_steps)
        self.latency_step = int(latency_step)
        self.device = device
        self.warmup_timeout_s = float(warmup_timeout_s)

        self._queue = _ActionQueue()
        self._stop_event = threading.Event()
        self.stats = RunnerStats()

    # -- main loop ----------------------------------------------------------

    def run(
        self,
        duration_s: float | None = None,
        on_tick: Callable[[RunnerStats], None] | None = None,
    ) -> RunnerStats:
        """Block until `duration_s` elapses, KeyboardInterrupt, or
        external `stop()`. Returns final stats. `on_tick` (optional)
        is called from the main thread at ~10 Hz with a stats snapshot
        — useful for diag scripts that want a live HUD."""
        self.env.start()
        self.env.reset()

        # Wait for the producer to put at least n_obs_steps frames into the buffer.
        n_needed = self.env.n_obs_steps
        t_warm = time.monotonic()
        while len(self.env._buffer) < n_needed:
            if time.monotonic() - t_warm > self.warmup_timeout_s:
                self.env.stop()
                raise RuntimeError(
                    f"Ur3BirRunner: buffer warmup timed out after "
                    f"{self.warmup_timeout_s}s (have {len(self.env._buffer)} "
                    f"frames, need {n_needed}). Producer errors: "
                    f"{self.env.producer_errors[-5:]}"
                )
            time.sleep(0.01)

        infer_t = threading.Thread(target=self._inference_loop,
                                   name="runner-inference", daemon=True)
        ctl_t = threading.Thread(target=self._control_loop,
                                 name="runner-control", daemon=True)

        try:
            infer_t.start()
            ctl_t.start()
            t_start = time.monotonic()
            try:
                while not self._stop_event.is_set():
                    if duration_s is not None and (time.monotonic() - t_start) > duration_s:
                        break
                    if on_tick is not None:
                        on_tick(self.stats)
                    self._stop_event.wait(timeout=0.1)
            except KeyboardInterrupt:
                pass
        finally:
            self._stop_event.set()
            infer_t.join(timeout=2.0)
            ctl_t.join(timeout=2.0)
            self.env.stop()
            self.executor.stop()
        return self.stats

    def stop(self) -> None:
        self._stop_event.set()

    # -- threads ------------------------------------------------------------

    def _inference_loop(self) -> None:
        interval = 1.0 / self.inference_fps
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self._step_inference()
                self.stats.inference_count += 1
            except Exception as e:
                self.stats.errors.append(("inference", f"{type(e).__name__}: {e}"))
            self.stats.inference_latency_s = time.monotonic() - t0
            sleep_s = interval - self.stats.inference_latency_s
            if sleep_s > 0:
                self._stop_event.wait(timeout=sleep_s)

    def _control_loop(self) -> None:
        interval = 1.0 / self.control_fps
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            action = self._queue.pop()
            if action is None:
                self.stats.starved_count += 1
            else:
                try:
                    self.executor.dispatch(action)
                    self.stats.dispatch_count += 1
                except Exception as e:
                    self.stats.errors.append(("control", f"{type(e).__name__}: {e}"))
            self.stats.dispatch_latency_s = time.monotonic() - t0
            sleep_s = interval - self.stats.dispatch_latency_s
            if sleep_s > 0:
                self._stop_event.wait(timeout=sleep_s)

    # -- per-tick -----------------------------------------------------------

    def _step_inference(self) -> None:
        import torch  # heavy import; keep local

        obs_np = self.env.get_obs()
        # All keys are (T, ...) float32; policies expect (B, T, ...).
        obs_t = {
            k: torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(0).to(self.device)
            for k, v in obs_np.items()
        }
        with torch.no_grad():
            result = self.policy.predict_action(obs_t)
        action = result["action"]                         # (1, horizon, action_dim)
        action_np = action[0].detach().to("cpu").numpy()  # (horizon, action_dim)

        ks = self.latency_step
        ke = ks + self.n_action_steps
        slice_ = action_np[ks:ke]
        if slice_.shape[0] == 0:
            return
        self._queue.replace(slice_)
