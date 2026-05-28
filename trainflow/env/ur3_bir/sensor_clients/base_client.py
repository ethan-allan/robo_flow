"""Sensor-client base contract for the UR3 BIR deployment env.

A sensor client is a thin wrapper around one piece of hardware (robot,
camera, tactile sensor). It exposes a uniform `get_latest()` returning
the most-recent reading as a `dict[str, Any]`. Live mode wraps the
vendor SDK; replay mode reads from an episode's `.npy` files indexed by
a shared `idx_ref` so all clients return the same frame on each tick.

Live-mode `__init__` may raise `NotImplementedError` — step-3 only
commits to the replay path; live wiring lands in step 5+.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


class BaseSensorClient(ABC):
    """Uniform contract for live + replay sensor clients."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._replay_mode = False

    def start(self) -> None:
        """Begin streaming. No-op in replay mode."""
        return None

    def stop(self) -> None:
        """Release the device. No-op in replay mode."""
        return None

    @abstractmethod
    def get_latest(self) -> dict[str, Any]:
        """Return the most-recent reading.

        Keys are sensor-specific; values are numpy arrays or scalars.
        Always includes a `ts` float (seconds, monotonic in live mode;
        per-frame timestamp in replay mode where available).
        """
        ...

    @classmethod
    def from_npy_replay(
        cls,
        cfg: DictConfig,
        episode_dir: Path,
        idx_ref: list[int],
        **kwargs: Any,
    ) -> "BaseSensorClient":
        """Construct a replay-mode client.

        `idx_ref` is a single-element list shared by every client and
        the env runner; mutating it advances all clients to the same
        frame on the next `get_latest()` call.

        Default: not supported. Subclasses that need replay (currently
        only `GelsightClient`) override this.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement from_npy_replay"
        )
