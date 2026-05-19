"""GelSight Mini client — one instance per physical sensor (slot 0, 1, ...).

Live mode wraps `cv2.VideoCapture(/dev/v4l/by-id/...)` with MJPG fourcc,
fps from cfg (default 25), buffersize=1, and a warmup of `cfg.warmup_frames`
(default 30). Mirrors `record_data_gui_new.py:initialize_gelsight_cameras`
(line 543) and `preview_loop_gelsight` (line 587). A dedicated capture
thread per slot reads frames into a lock-protected slot.

Raw output is BGR uint8 at native resolution (~3280x2464); the
capture-script resize + BGR→RGB are deliberately NOT applied in the
client — the obs_sources op chain handles them (a future task that
declares gelsight in shape_meta.obs will carry `resize` + `bgr_to_rgb`
ops). This is symmetric with `RealsenseClient`.

Replay mode reads `gelsight_rgb_<slot>.npy` from one episode dir;
`idx_ref[0]` selects frame. On-disk replay frames are already
(240, 320, 3) RGB (post-processed by the legacy capture script).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from .base_client import BaseSensorClient


class GelsightClient(BaseSensorClient):
    def __init__(self, cfg: DictConfig, slot: int = 0):
        super().__init__(cfg)
        self._slot = int(slot)

    # -- replay -------------------------------------------------------------

    @classmethod
    def from_npy_replay(
        cls,
        cfg: DictConfig,
        episode_dir: Path,
        idx_ref: list[int],
        slot: int = 0,
        **kwargs: Any,
    ) -> "GelsightClient":
        self = cls.__new__(cls)
        BaseSensorClient.__init__(self, cfg)
        self._replay_mode = True
        self._idx_ref = idx_ref
        self._slot = int(slot)
        ep = Path(episode_dir)
        rgb_path = ep / f"gelsight_rgb_{slot}.npy"
        if not rgb_path.exists():
            raise FileNotFoundError(
                f"{ep.name}: missing gelsight_rgb_{slot}.npy (slot={slot})"
            )
        self._rgb = np.load(rgb_path)                              # (T, 240, 320, 3)
        ts_path = ep / f"gelsight_ts_{slot}.npy"
        self._gs_ts = np.load(ts_path) if ts_path.exists() else None
        # Two-clock domain: gelsight runs at ~25 Hz native vs the 5 Hz
        # synchronized loop. Index by the synchronized frame index against
        # the per-slot timestamp series, picking the latest gelsight
        # sample whose ts <= frame_timestamp[idx_ref].
        self._frame_ts = np.load(ep / "frame_timestamp.npy")
        return self

    # -- live ---------------------------------------------------------------

    def start(self) -> None:
        if self._replay_mode:
            return
        import cv2  # heavy import

        # Resolve the path for this slot. Prefer cfg.paths if pinned;
        # otherwise auto-discover under /dev/v4l/by-id.
        cfg_paths = self.cfg.get("paths", None)
        if cfg_paths:
            paths = [str(p) for p in cfg_paths]
        else:
            from trainflow.env.ur3_bir.discover import discover_gelsight_devices
            paths = discover_gelsight_devices()
        if self._slot >= len(paths):
            raise RuntimeError(
                f"GelsightClient: requested slot={self._slot} but only "
                f"{len(paths)} GelSight sensor(s) discovered: {paths}"
            )
        path = paths[self._slot]

        fps = int(self.cfg.get("fps", 25))
        warmup_frames = int(self.cfg.get("warmup_frames", 30))
        fourcc_str = str(self.cfg.get("fourcc", "MJPG"))

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"GelsightClient slot={self._slot}: cannot open {path}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Discard warmup frames — the sensor outputs zeros for ~30 frames on connect.
        for _ in range(warmup_frames):
            cap.grab()
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise RuntimeError(
                f"GelsightClient slot={self._slot}: no frame after {warmup_frames}-frame warmup"
            )

        self._cap = cap
        self._path = path
        self._latest_lock = threading.Lock()
        self._latest: dict[str, Any] = {"rgb": frame, "ts": time.time()}
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"gelsight-{self._slot}",
            daemon=True,
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue
            ts = time.time()
            with self._latest_lock:
                self._latest = {"rgb": frame, "ts": ts}

    def stop(self) -> None:
        if self._replay_mode:
            return
        try:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._cap.release()
        except Exception:
            pass

    # -- public -------------------------------------------------------------

    def get_latest(self) -> dict[str, Any]:
        if self._replay_mode:
            i_frame = self._idx_ref[0]
            if self._gs_ts is not None:
                target_ts = float(self._frame_ts[i_frame])
                j = int(np.searchsorted(self._gs_ts, target_ts, side="right")) - 1
                j = max(0, min(j, len(self._rgb) - 1))
            else:
                j = min(i_frame, len(self._rgb) - 1)
            return {
                "rgb": self._rgb[j],
                "ts": float(self._gs_ts[j]) if self._gs_ts is not None else float(self._frame_ts[i_frame]),
            }

        with self._latest_lock:
            return dict(self._latest)
