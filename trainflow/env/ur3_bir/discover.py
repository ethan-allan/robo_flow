"""Small device-discovery helpers, vendored from the capture-script repo.

Keeps the trainflow runtime independent of
`/media/ethan/SANDISK128/capture/data_recording/` (a removable mount). The
originals are:

  * `detect_ur_robot`   — `auto_detect_devices.py:37`
  * `discover_gelsight_devices` and `_gelsight_serial_from_path`
                        — `record_data_gui_new.py:230` / `:246`
  * `detect_realsense_cameras` — the RealSense branch of
                          `auto_detect_devices.py:detect_all_cameras` at `:62`

These do no I/O against the robot/cameras beyond a ping or a v4l directory
listing. The full pyrealsense2 / RTDE connections happen inside the sensor
clients.
"""
from __future__ import annotations

import os
import socket
import subprocess
from typing import Iterable


# ---------------------------------------------------------------------------
# UR robot
# ---------------------------------------------------------------------------

def _ping(ip: str, timeout_s: int = 1) -> bool:
    try:
        subprocess.check_output(
            ["ping", "-c", "1", "-W", str(timeout_s), ip],
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _is_port_open(ip: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout_s):
            return True
    except Exception:
        return False


def detect_ur_robot(ip_list: Iterable[str], rtde_port: int = 30004) -> str | None:
    """Probe each IP via ping → RTDE port → handshake. Return the first
    IP whose `getActualQ()` returns non-None, else None. Mirrors
    auto_detect_devices.detect_ur_robot but lets the caller pass the
    RTDE port so the hw cfg drives it.
    """
    from rtde_receive import RTDEReceiveInterface  # local import: heavy

    for ip in ip_list:
        if not _ping(ip):
            continue
        if not _is_port_open(ip, rtde_port):
            continue
        try:
            rtde_r = RTDEReceiveInterface(ip)
            try:
                if rtde_r.getActualQ() is not None:
                    return ip
            finally:
                try:
                    rtde_r.disconnect()
                except Exception:
                    pass
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# RealSense
# ---------------------------------------------------------------------------

def detect_realsense_cameras() -> list[dict]:
    """Enumerate connected RealSense devices. Each entry is
    {name, serial}. Empty list if pyrealsense2 is missing or no devices.
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        return []
    try:
        ctx = rs.context()
        out = []
        for dev in ctx.query_devices():
            out.append({
                "name": dev.get_info(rs.camera_info.name),
                "serial": dev.get_info(rs.camera_info.serial_number),
            })
        return out
    except Exception:
        return []


# ---------------------------------------------------------------------------
# GelSight Mini
# ---------------------------------------------------------------------------

GELSIGHT_BY_ID_DIR = "/dev/v4l/by-id"


def discover_gelsight_devices(by_id_dir: str = GELSIGHT_BY_ID_DIR) -> list[str]:
    """Return sorted list of GelSight Mini capture-node paths under
    /dev/v4l/by-id. Each sensor exposes two video nodes; we keep only
    the `-video-index0` capture node. Sorting by filename → sorting by
    serial, so slot 0/1 are stable across reboots.
    """
    if not os.path.isdir(by_id_dir):
        return []
    out = []
    for name in sorted(os.listdir(by_id_dir)):
        if "GelSight_Mini" in name and name.endswith("-video-index0"):
            out.append(os.path.join(by_id_dir, name))
    return out


def gelsight_serial_from_path(path: str) -> str:
    """Extract the sensor serial from a /dev/v4l/by-id path, or '' if
    unknown. Path layout: ...GelSight_Mini_R0B_<MODEL>_<SERIAL>-video-index0
    """
    if not path:
        return ""
    base = os.path.basename(path)
    marker = "GelSight_Mini_R0B_"
    i = base.find(marker)
    if i < 0:
        return ""
    rest = base[i + len(marker):]
    rest = rest.split("-video-")[0]
    return rest.split("_")[0]
