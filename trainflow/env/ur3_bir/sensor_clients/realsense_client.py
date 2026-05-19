"""RealSense client — one instance per camera serial.

Live mode wraps `pyrealsense2.pipeline` per the capture script's
`start_realsense_by_serial` (record_data_gui_new.py:732) and
`preview_loop_realsense` (:1337). Frames are read off the SDK in a
dedicated capture thread (`poll_for_frames`, non-blocking) and stashed
in a lock-protected slot; `get_latest()` snapshots that slot.

Raw output is BGR uint8 (rs.format.bgr8). The capture script applies
crop + resize + BGR→RGB inside its synchronized loop before saving;
in this framework that work belongs to the per-key op chain
(`bgr_to_rgb`, `crop_x`, `resize`). See plan hand-off note #2.

Replay mode reads `<key>.npy` from one episode dir, where `<key>` is
configured via the `replay_key` constructor kwarg (defaults to "rgb"
for the platform camera; pass "rgb_hand" for the hand camera).

Shape divergence between live and replay (live = native 480x640 BGR;
replay = on-disk post-processed RGB) is bridged by the
`applied_at_capture` mechanism declared per-sensor in the hw cfg.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from .base_client import BaseSensorClient


class RealsenseClient(BaseSensorClient):

    # -- live ---------------------------------------------------------------

    def start(self) -> None:
        if self._replay_mode:
            return
        import pyrealsense2 as rs  # heavy import

        serial = self.cfg.get("serial", None)
        if serial in (None, "", "null"):
            raise RuntimeError(
                "RealsenseClient: serial is null. Populate it in the hw "
                "cfg or run discover.detect_realsense_cameras() to list "
                "connected devices."
            )
        width = int(self.cfg.get("width", 640))
        height = int(self.cfg.get("height", 480))
        fps = int(self.cfg.get("fps", 15))
        d_w = int(self.cfg.get("depth_width", width))
        d_h = int(self.cfg.get("depth_height", height))

        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(str(serial))
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, d_w, d_h, rs.format.z16, fps)
        profile = pipe.start(cfg)

        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self._intrinsics = depth_profile.get_intrinsics()
        self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        self._pipeline = pipe
        self._latest_lock = threading.Lock()
        self._latest: dict[str, Any] = {}
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"realsense-{serial}",
            daemon=True,
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        import pyrealsense2 as rs  # noqa: F401 — keep local for thread safety
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.poll_for_frames()
            except Exception:
                frames = None
            if not frames:
                time.sleep(0.001)
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            ts = time.time()
            update: dict[str, Any] = {"ts": ts}
            if color_frame:
                try:
                    update["rgb"] = np.asarray(color_frame.get_data())
                except Exception:
                    pass
            if depth_frame:
                try:
                    update["depth"] = np.asarray(depth_frame.get_data()).copy()
                except Exception:
                    pass
            if "rgb" in update:
                with self._latest_lock:
                    self._latest.update(update)

    def stop(self) -> None:
        if self._replay_mode:
            return
        try:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._pipeline.stop()
        except Exception:
            pass

    # -- public -------------------------------------------------------------

    def get_latest(self) -> dict[str, Any]:
        if self._replay_mode:
            i = self._idx_ref[0]
            out: dict[str, Any] = {
                "rgb": self._rgb[i],
                "ts": float(self._frame_ts[i]),
            }
            if self._depth is not None:
                out["depth"] = self._depth[i]
            return out

        with self._latest_lock:
            if "rgb" not in self._latest:
                raise RuntimeError(
                    "RealsenseClient.get_latest: no frame yet; call "
                    "start() and wait for the capture thread to fill."
                )
            return dict(self._latest)  # shallow copy under lock
