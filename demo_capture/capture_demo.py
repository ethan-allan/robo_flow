"""Demo capture — record human teleoperated demonstrations into raw
per-episode `.npy` directories that `trainflow.common.zarr_writer` can
turn into a training zarr on demand.

Three modes:

  --mode freedrive
      Robot is put in gravity-comp via RTDE `teachMode()`. The operator
      moves the arm by hand. action[t] = eef_state[t+1] (drops the last
      frame). No robot commands are issued by the script.

  --mode spacemouse
      Operator drives a 3Dconnexion SpaceMouse. Each tick reads the 6-DOF
      deflection, integrates it into a target TCP pose, dispatches it via
      `RtdeExecutor.servoL`, and records the commanded 6-DOF pose (plus the
      observed gripper width) as the action label.
      Install once: `pip install pyspacemouse`.

  --mode gello
      Operator drives a GELLO leader arm. Each tick reads the leader's
      6-DOF joint vector, runs UR3 forward kinematics, dispatches the TCP
      target, and records the commanded pose (+ observed gripper) as the
      action label.
      Install gello_software from https://github.com/wuphilipp/gello_software
      and pass the leader's serial port via --gello-port.

Episode boundaries are interactive: press ENTER to start, ENTER again to
end. After each episode you're prompted to save (Y/n). 'q' ENTER quits.

Output:
  Each saved episode becomes a directory `<output>/episode_NNNN/` holding
  one `.npy` per modality plus a `meta.yaml`. All signals the instantiated
  sensor clients emit are stored RAW (pre op-chain) so the layout matches
  what `zarr_writer` + the replay clients read:

    eef_state.npy        (T, 7)  [xyz, axis-angle, gripper]
    eef_action.npy       (T, 7)  recorded action, same layout
    eef_force.npy        (T, 6)
    joint_state.npy      (T, 7)  [6 joints, gripper]
    current.npy          (T, 6)
    control_mode.npy     (T, 1)
    frame_timestamp.npy  (T,)
    rgb.npy              (T, H, W, 3) uint8 BGR (cropped/resized)
    depth.npy            (T, H, W)    uint16   (if the camera streams it)
    meta.yaml

  Build the training zarr later with:
    python -m trainflow.common.zarr_writer \\
        --task trainflow/config/task/<task>.yaml --src <output>

  Output dir resolution: --output > task_cfg.raw_episodes > data/<name>_raw.
  Sessions append: numbering continues past existing episode_* dirs.

Preconditions (all modes):
  * Robot is in remote-control mode on Polyscope, no e-stop.
  * Cameras and (if shape_meta uses tactile) GelSight are reachable.

Usage:
  python -m demo_capture.capture_demo --task real_ee2_dice --mode freedrive
  python -m demo_capture.capture_demo --task real_ee2_dice --mode spacemouse \\
      --spacemouse-trans-gain 0.02 --spacemouse-rot-gain 0.3
  python -m demo_capture.capture_demo --task real_ee2_dice --mode gello \\
      --gello-port /dev/ttyUSB0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from trainflow.common.obs_processing import crop_x, resize
from trainflow.env.ur3_bir.action_executor import RtdeExecutor
from trainflow.env.ur3_bir.ur3_bir_env import Ur3BirEnv


REPO = Path(__file__).resolve().parents[1]
CFG_DIR = REPO / "trainflow" / "config"


# Match phase_h_dry_run.py — training cfgs use ${eval:...} / ${now:...}.
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "now",
    lambda fmt="%Y%m%d-%H%M%S": datetime.now().strftime(fmt),
    replace=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_task(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        return compose(config_name=f"task/{name}")


def pose6_to_pose9(p6: np.ndarray) -> np.ndarray:
    """[x, y, z, rx, ry, rz] (axis-angle) -> [x, y, z, ortho6d].
    ortho6d is the first two columns of the rotation matrix, flattened
    column-major (the inverse of OP_REGISTRY['pose9_to_pose6'])."""
    R = Rotation.from_rotvec(p6[3:6]).as_matrix()       # (3, 3)
    return np.concatenate([p6[:3], R[:, :2].T.reshape(6)])


class _StdinSignal:
    """Background reader: blocks on stdin so the capture loop ticks
    without sitting in input(). Each line read sets `event`; consumers
    poll `event.is_set()`, read `last_line`, and `event.clear()`."""

    def __init__(self):
        self.event = threading.Event()
        self.last_line: str = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                return
            self.last_line = line.rstrip("\n")
            self.event.set()

    def consume(self) -> Optional[str]:
        if not self.event.is_set():
            return None
        line = self.last_line
        self.event.clear()
        return line


# ---------------------------------------------------------------------------
# Mode adapters
# ---------------------------------------------------------------------------

class CaptureMode:
    name: str = "base"

    def setup(self) -> None: ...
    def episode_start(self) -> None: ...
    def tick(self, raw: dict) -> Optional[np.ndarray]:
        """Dispatch one control step (if any). `raw` is the per-tick
        sweep {client_parent: get_latest() dict}. Returns the recorded
        action label (7-D [xyz, axis-angle, gripper]) or None."""
        ...
    def episode_end(self) -> None: ...
    def finalize_action(
        self, recorded: list, eef_state: Optional[np.ndarray]
    ) -> np.ndarray:
        raise NotImplementedError
    def teardown(self) -> None: ...


class FreedriveMode(CaptureMode):
    name = "freedrive"

    def __init__(self, host: str):
        self.host = host
        self._rtde_c = None

    def setup(self) -> None:
        from rtde_control import RTDEControlInterface
        self._rtde_c = RTDEControlInterface(self.host)

    def episode_start(self) -> None:
        self._rtde_c.teachMode()

    def tick(self, raw: dict) -> Optional[np.ndarray]:
        return None

    def episode_end(self) -> None:
        try:
            self._rtde_c.endTeachMode()
        except Exception:
            pass

    def finalize_action(
        self, recorded: list, eef_state: Optional[np.ndarray]
    ) -> np.ndarray:
        # action[t] = eef_state[t+1]; drop the last frame. eef_state is
        # the raw (T, 7) [xyz, axis-angle, gripper] — store as-is, no
        # ortho6d conversion (zarr_writer's pose7_to_pose9 does that).
        if eef_state is None:
            raise RuntimeError(
                "FreedriveMode needs a robot client to synthesise actions "
                "from eef_state; none was instantiated"
            )
        return eef_state[1:].astype(np.float32)

    def teardown(self) -> None:
        if self._rtde_c is not None:
            try:
                self._rtde_c.endTeachMode()
            except Exception:
                pass
            try:
                self._rtde_c.disconnect()
            except Exception:
                pass


class SpacemouseMode(CaptureMode):
    name = "spacemouse"

    def __init__(self, env: Ur3BirEnv, trans_gain: float, rot_gain: float):
        self.env = env
        self.trans_gain = float(trans_gain)
        self.rot_gain = float(rot_gain)
        self._pyspacemouse = None
        self.target_pose6: Optional[np.ndarray] = None
        self.last_t: float = 0.0

    def setup(self) -> None:
        try:
            import pyspacemouse
        except ImportError:
            print(
                "[fatal] pyspacemouse is required for --mode spacemouse.\n"
                "        Install with: pip install pyspacemouse\n"
                "        (Linux also needs libhidapi-dev installed.)",
                file=sys.stderr,
            )
            raise
        if not pyspacemouse.open():
            raise RuntimeError("pyspacemouse.open() returned False")
        self._pyspacemouse = pyspacemouse

    def episode_start(self) -> None:
        ur3 = self.env._clients["robot.ur3"]
        self.target_pose6 = np.asarray(
            ur3.get_latest()["tcp_pose6"], dtype=np.float64
        ).copy()
        self.last_t = time.monotonic()
        # Drain any queued spacemouse state so a button held during
        # the prompt doesn't translate into a huge first delta.
        for _ in range(5):
            self._pyspacemouse.read()

    def tick(self, raw: dict) -> Optional[np.ndarray]:
        st = self._pyspacemouse.read()
        now = time.monotonic()
        dt = now - self.last_t
        self.last_t = now

        dxyz = np.array([st.x, st.y, st.z]) * self.trans_gain * dt
        drpy = np.array([st.roll, st.pitch, st.yaw]) * self.rot_gain * dt

        R_new = (
            Rotation.from_rotvec(drpy)
            * Rotation.from_rotvec(self.target_pose6[3:6])
        )
        self.target_pose6[:3] += dxyz
        self.target_pose6[3:6] = R_new.as_rotvec()

        # Dispatch the 9-D (xyz + ortho6d) target; record the 7-D form
        # (commanded pose6 + observed gripper) for eef_action.npy.
        action_9d = pose6_to_pose9(self.target_pose6).astype(np.float32)
        self.env.step(action_9d)
        return _pose6_plus_gripper(self.target_pose6, raw)

    def episode_end(self) -> None:
        return None

    def finalize_action(
        self, recorded: list, eef_state: Optional[np.ndarray]
    ) -> np.ndarray:
        return np.stack(recorded).astype(np.float32)

    def teardown(self) -> None:
        if self._pyspacemouse is not None:
            try:
                self._pyspacemouse.close()
            except Exception:
                pass


class GelloMode(CaptureMode):
    """GELLO leader-arm teleop. Reads leader joints, FK on the UR3, and
    dispatches the resulting TCP target via env.step.

    The script doesn't ship a GELLO driver — bring your own. By default
    we lazily import `gello.agents.gello_agent.GelloAgent` from the
    upstream gello_software repo. Override `--gello-agent-class` if your
    leader uses a different driver class (must expose `.get_joint_state()`
    -> ndarray of 6 joint angles in radians, in UR3 joint order).
    """
    name = "gello"

    def __init__(
        self,
        host: str,
        port: str,
        agent_class: str = "gello.agents.gello_agent.GelloAgent",
    ):
        self.host = host
        self.port = port
        self.agent_class = agent_class
        self._agent = None
        self._rtde_c = None     # side-band for FK only

    def setup(self) -> None:
        # FK side-band: we only use getForwardKinematics here. The main
        # motion commands flow through env.step -> RtdeExecutor on its
        # own RTDEControlInterface (the UR3 accepts multiple control
        # connections; FK is read-only).
        from rtde_control import RTDEControlInterface
        self._rtde_c = RTDEControlInterface(self.host)

        try:
            module_path, cls_name = self.agent_class.rsplit(".", 1)
            mod = __import__(module_path, fromlist=[cls_name])
            AgentCls = getattr(mod, cls_name)
        except (ImportError, AttributeError) as e:
            print(
                f"[fatal] could not import {self.agent_class!r} for "
                f"--mode gello.\n"
                f"        Install gello_software and ensure it's on "
                f"PYTHONPATH, or pass a different\n"
                f"        --gello-agent-class. Underlying error: {e}",
                file=sys.stderr,
            )
            raise

        try:
            self._agent = AgentCls(port=self.port)
        except TypeError:
            # Some GELLO drivers want `(port, start_joints, ...)` —
            # fall back to a port-only constructor.
            self._agent = AgentCls(self.port)

    def _read_joints(self) -> np.ndarray:
        a = self._agent
        for attr in ("get_joint_state", "get_action", "act"):
            fn = getattr(a, attr, None)
            if callable(fn):
                try:
                    out = fn() if attr != "act" else fn({})
                except TypeError:
                    out = fn()
                j = np.asarray(out, dtype=np.float64).reshape(-1)
                return j[:6]
        raise RuntimeError(
            f"gello agent {type(self._agent).__name__} exposes none of "
            f"get_joint_state / get_action / act — cannot read joints"
        )

    def episode_start(self) -> None:
        return None

    def tick(self, raw: dict) -> Optional[np.ndarray]:
        joints = self._read_joints()
        tcp_pose6 = np.asarray(
            self._rtde_c.getForwardKinematics(joints.tolist()),
            dtype=np.float64,
        )
        action_9d = pose6_to_pose9(tcp_pose6).astype(np.float32)
        self.env.step(action_9d)
        return _pose6_plus_gripper(tcp_pose6, raw)

    def episode_end(self) -> None:
        return None

    def finalize_action(
        self, recorded: list, eef_state: Optional[np.ndarray]
    ) -> np.ndarray:
        return np.stack(recorded).astype(np.float32)

    def teardown(self) -> None:
        for thing in (self._agent, self._rtde_c):
            if thing is None:
                continue
            for method in ("close", "stop", "disconnect"):
                fn = getattr(thing, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass

    @property
    def env(self) -> Ur3BirEnv:
        return self._env

    @env.setter
    def env(self, value: Ur3BirEnv) -> None:
        self._env = value


def _pose6_plus_gripper(pose6: np.ndarray, raw: dict) -> np.ndarray:
    """Build a 7-D action label [xyz, axis-angle, gripper]. The gripper
    channel is the *observed* width (these modes don't command it); it is
    dropped by zarr_writer's pose7_to_pose9, so it only documents state."""
    grip = raw.get("robot.ur3", {}).get("gripper_width", np.zeros(1))
    return np.concatenate(
        [np.asarray(pose6, dtype=np.float32).reshape(6),
         np.atleast_1d(np.asarray(grip, dtype=np.float32)).reshape(-1)[:1]]
    )


# ---------------------------------------------------------------------------
# Save / format helpers
# ---------------------------------------------------------------------------

# Raw robot leaves -> canonical on-disk filenames (passthrough, no transform).
# `eef_state` is built separately (concat of tcp_pose6 + gripper_width).
_ROBOT_PASSTHROUGH = {
    "eef_force": "eef_force.npy",
    "joint_state": "joint_state.npy",
    "current": "current.npy",
    "control_mode": "control_mode.npy",
}


def _rgb_spatial_transform(ops) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Compose the spatial image ops (crop_x, resize) from an obs key's op
    chain, skipping the ops zarr_writer re-applies on load (bgr_to_rgb) and
    non-image ops (concat / pose conversions). Returns a per-frame callable
    or None if there are no spatial ops."""
    spatial = {"crop_x": crop_x, "resize": resize}
    try:
        ops_py = OmegaConf.to_container(ops, resolve=True)
    except Exception:
        ops_py = list(ops) if ops else []
    steps: list[tuple[Callable, dict]] = []
    for op in ops_py or []:
        name = op.get("name")
        if name in spatial:
            steps.append((spatial[name], {k: v for k, v in op.items() if k != "name"}))
    if not steps:
        return None

    def transform(frame: np.ndarray) -> np.ndarray:
        for fn, kw in steps:
            frame = fn(frame, **kw)
        return frame

    return transform


def _image_obs_specs(task_cfg) -> list[dict]:
    """One spec per rgb-typed obs key: which raw leaf feeds it, the
    capture-stage spatial transform, and the on-disk filenames. The image
    file is named after the obs key (matches zarr_writer's source_file_for);
    the depth sibling follows the `depth[/_<suffix>]` convention."""
    specs: list[dict] = []
    for key, attr in task_cfg.shape_meta.obs.items():
        if "wrt" in key:
            continue
        if attr.get("type", "low_dim") != "rgb":
            continue
        frm = attr.get("from")
        if not isinstance(frm, str):
            continue
        parent, leaf = frm.rsplit(".", 1)
        depth_name = ("depth" + key[3:]) if key.startswith("rgb") else f"depth_{key}"
        specs.append({
            "filename": f"{key}.npy",
            "depth_filename": f"{depth_name}.npy",
            "parent": parent,
            "leaf": leaf,
            "transform": _rgb_spatial_transform(attr.get("ops", [])),
        })
    return specs


def _wait_clients_ready(env: Ur3BirEnv, timeout_s: float = 10.0) -> None:
    """Poll every client's get_latest() until one full sweep succeeds
    (the RealSense client raises until its capture thread fills)."""
    t0 = time.monotonic()
    while True:
        try:
            for c in env._clients.values():
                c.get_latest()
            return
        except Exception as e:
            if time.monotonic() - t0 > timeout_s:
                errs = env.producer_errors[:3]
                raise RuntimeError(
                    f"sensor clients didn't become ready in {timeout_s:.0f}s "
                    f"({type(e).__name__}: {e}); producer errors: {errs}"
                )
            time.sleep(0.1)


def _next_episode_index(out_root: Path) -> int:
    if not out_root.exists():
        return 0
    idxs: list[int] = []
    for d in out_root.iterdir():
        if d.is_dir() and d.name.startswith("episode_"):
            try:
                idxs.append(int(d.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                pass
    return max(idxs) + 1 if idxs else 0


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO), capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------

def _record_one_episode(
    env: Ur3BirEnv,
    mode: CaptureMode,
    stdin_sig: _StdinSignal,
    fps: float,
) -> Optional[dict]:
    """Run the inner tick loop until a second ENTER. Each tick sweeps every
    client's raw get_latest() and buffers every array leaf. Returns the
    in-memory buffers dict, or None if the user aborted with no frames."""
    interval = 1.0 / float(fps)
    leaves: dict[str, list] = defaultdict(list)
    ts_list: list[float] = []
    action_list: list = []

    print("  recording... press ENTER to stop")
    mode.episode_start()
    start_dt = datetime.now()
    t_start = time.monotonic()
    next_tick = t_start
    last_log = t_start
    tick_count = 0
    try:
        while True:
            now = time.monotonic()
            if stdin_sig.consume() is not None:
                break
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.02))
                continue
            next_tick = now + interval

            raw = {p: c.get_latest() for p, c in env._clients.items()}
            ts = float(raw.get("robot.ur3", {}).get("ts", time.time()))
            for parent, out in raw.items():
                for leaf, val in out.items():
                    if leaf == "ts":
                        continue
                    leaves[f"{parent}/{leaf}"].append(np.asarray(val))
            ts_list.append(ts)
            action_list.append(mode.tick(raw))
            tick_count += 1

            if now - last_log >= 1.0:
                elapsed = now - t_start
                tcp = leaves.get("robot.ur3/tcp_pose6")
                xyz = None if not tcp else np.round(tcp[-1][:3], 3).tolist()
                print(
                    f"  [t={elapsed:5.1f}s] frames={tick_count:4d} xyz={xyz}",
                    flush=True,
                )
                last_log = now
    finally:
        mode.episode_end()

    if tick_count == 0:
        return None
    buffers: dict = dict(leaves)
    buffers["_ts"] = ts_list
    buffers["_action"] = action_list
    buffers["_start_dt"] = start_dt
    buffers["_end_dt"] = datetime.now()
    return buffers


def _save_episode_npy(
    ep_dir: Path,
    buffers: dict,
    mode: CaptureMode,
    image_specs: list[dict],
    fps: float,
    task_cfg,
    args,
) -> int:
    """Stack raw buffers, attach the action via the mode, write one .npy per
    modality plus meta.yaml. Returns the saved frame count T."""
    n_ticks = len(buffers["_ts"])
    stacked = {
        k: np.stack(v) for k, v in buffers.items()
        if not k.startswith("_") and len(v) == n_ticks
    }

    eef_state_7d = None
    if "robot.ur3/tcp_pose6" in stacked and "robot.ur3/gripper_width" in stacked:
        eef_state_7d = np.concatenate(
            [stacked["robot.ur3/tcp_pose6"], stacked["robot.ur3/gripper_width"]],
            axis=-1,
        )

    action = mode.finalize_action(buffers["_action"], eef_state_7d)
    T = int(action.shape[0])

    ep_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, np.ndarray] = {}

    def _save(name: str, arr: np.ndarray) -> None:
        arr = arr[:T]
        np.save(ep_dir / name, arr)
        saved[name] = arr

    # Robot signals (canonical names; eef_state is the concat).
    if eef_state_7d is not None:
        _save("eef_state.npy", eef_state_7d)
    for leaf, fname in _ROBOT_PASSTHROUGH.items():
        key = f"robot.ur3/{leaf}"
        if key in stacked:
            _save(fname, stacked[key])

    _save("eef_action.npy", action)

    # Camera / tactile images (apply spatial ops, keep BGR) + depth sibling.
    for spec in image_specs:
        key = f"{spec['parent']}/{spec['leaf']}"
        if key not in stacked:
            continue
        frames = stacked[key]
        if spec["transform"] is not None:
            frames = np.stack([spec["transform"](f) for f in frames])
        _save(spec["filename"], np.ascontiguousarray(frames).astype(np.uint8))
        dkey = f"{spec['parent']}/depth"
        if dkey in stacked:
            _save(spec["depth_filename"], stacked[dkey])

    _save("frame_timestamp.npy", np.asarray(buffers["_ts"], dtype=np.float64))

    _write_meta_yaml(ep_dir, buffers, saved, mode, fps, task_cfg, args, T)
    return T


def _write_meta_yaml(
    ep_dir: Path, buffers: dict, saved: dict, mode: CaptureMode,
    fps: float, task_cfg, args, T: int,
) -> None:
    ts = np.asarray(buffers["_ts"], dtype=np.float64)[:T]
    duration = float(ts[-1] - ts[0]) if T > 1 else 0.0

    mode_params: dict = {}
    if mode.name == "spacemouse":
        mode_params = {
            "spacemouse_trans_gain": float(args.spacemouse_trans_gain),
            "spacemouse_rot_gain": float(args.spacemouse_rot_gain),
        }
    elif mode.name == "gello":
        mode_params = {"gello_port": str(args.gello_port)}

    try:
        host = str(task_cfg.hardware.clients.robot.ur3.host)
    except Exception:
        host = None
    action_type = None
    if "action" in task_cfg.shape_meta:
        action_type = task_cfg.shape_meta.action.get("type", None)

    meta = {
        "task_name": str(task_cfg.get("name", "")),
        "teleop_mode": mode.name,
        "fps": float(fps),
        "control_fps": float(task_cfg.hardware.get("control_fps", fps)),
        "episode_index": int(ep_dir.name.split("_", 1)[1])
        if "_" in ep_dir.name else None,
        "start_datetime": buffers["_start_dt"].isoformat(timespec="seconds"),
        "end_datetime": buffers["_end_dt"].isoformat(timespec="seconds"),
        "n_frames": T,
        "duration_s": round(duration, 3),
        "robot_host": host,
        "action_type": action_type,
        "git_commit": _git_commit(),
        "mode_params": mode_params,
        "files": {
            name: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            for name, arr in sorted(saved.items())
        },
    }
    OmegaConf.save(OmegaConf.create(meta), ep_dir / "meta.yaml")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--task", required=True,
                    help="Task cfg name under trainflow/config/task/.")
    ap.add_argument("--mode", required=True,
                    choices=["freedrive", "spacemouse", "gello"],
                    help="Demonstration source.")
    ap.add_argument("--fps", type=float, default=None,
                    help="Capture rate. Defaults to hardware.control_fps, or 5.")
    ap.add_argument("--output", default=None,
                    help="Override output dir. Default: task_cfg.raw_episodes "
                         "or data/<name>_raw. Episodes go in <output>/episode_NNNN/.")
    ap.add_argument("--spacemouse-trans-gain", type=float, default=0.02,
                    help="m/s at full deflection (spacemouse mode only).")
    ap.add_argument("--spacemouse-rot-gain", type=float, default=0.3,
                    help="rad/s at full deflection (spacemouse mode only).")
    ap.add_argument("--gello-port", default="/dev/ttyUSB0",
                    help="Serial port for the GELLO leader (gello mode only).")
    ap.add_argument("--gello-agent-class",
                    default="gello.agents.gello_agent.GelloAgent",
                    help="Dotted path to a GELLO agent class that exposes "
                         ".get_joint_state() (or .get_action/.act).")
    args = ap.parse_args()

    task_cfg = _load_task(args.task).task

    # This recorder writes a 7-D eef_action.npy (xyz + axis-angle + gripper);
    # zarr_writer converts 7->9. A non-9-D trained action means a different
    # action mode than tcp_absolute_9d, which we can't faithfully record.
    action_shape = list(task_cfg.shape_meta.action.shape)
    if action_shape != [9]:
        print(f"[warn] shape_meta.action.shape={action_shape}, expected [9] "
              f"(tcp_absolute_9d). Recording the commanded TCP pose anyway.",
              file=sys.stderr)

    # FPS resolution: CLI > hardware.control_fps > 5.
    if args.fps is not None:
        fps = float(args.fps)
    else:
        fps = float(task_cfg.hardware.get("control_fps", 5))

    if args.output:
        out_root = Path(args.output)
    elif "raw_episodes" in task_cfg:
        out_root = Path(task_cfg.raw_episodes)
    else:
        out_root = Path("data") / f"{task_cfg.get('name', 'task')}_raw"
    if not out_root.is_absolute():
        out_root = REPO / out_root

    start_idx = _next_episode_index(out_root)
    print(f"[info] task     : {args.task}")
    print(f"[info] mode     : {args.mode}")
    print(f"[info] fps      : {fps}")
    print(f"[info] output   : {out_root}")
    print(f"[info] starting : {start_idx} episode(s) already on disk")

    # Build env. Spacemouse/gello need an executor that actually moves
    # the arm; freedrive must not have one.
    if args.mode in ("spacemouse", "gello"):
        executor = RtdeExecutor(task_cfg, task_cfg.hardware)
        env = Ur3BirEnv(task_cfg, n_obs_steps=1, executor=executor)
    else:
        env = Ur3BirEnv(task_cfg, n_obs_steps=1, executor=None)

    env.start()
    try:
        _wait_clients_ready(env)
        print(f"[info] sensor clients ready: {sorted(env._clients)}")
        image_specs = _image_obs_specs(task_cfg)

        host = str(task_cfg.hardware.clients.robot.ur3.host)
        if args.mode == "spacemouse":
            mode: CaptureMode = SpacemouseMode(
                env=env,
                trans_gain=args.spacemouse_trans_gain,
                rot_gain=args.spacemouse_rot_gain,
            )
        elif args.mode == "gello":
            mode = GelloMode(
                host=host,
                port=args.gello_port,
                agent_class=args.gello_agent_class,
            )
            mode.env = env
        else:
            mode = FreedriveMode(host=host)
        mode.setup()

        next_idx = start_idx
        stdin_sig = _StdinSignal()
        try:
            while True:
                print()
                print(
                    f"--- ready for episode {next_idx} "
                    f"(ENTER to start, 'q' ENTER to quit) ---",
                    flush=True,
                )
                # Block until the user gives us a line.
                while True:
                    line = stdin_sig.consume()
                    if line is not None:
                        break
                    time.sleep(0.05)
                if line.strip().lower() == "q":
                    break

                buffers = _record_one_episode(
                    env=env, mode=mode, stdin_sig=stdin_sig, fps=fps,
                )
                if buffers is None:
                    print("  (no frames captured; nothing to save)")
                    continue

                duration = buffers["_ts"][-1] - buffers["_ts"][0]
                T_raw = len(buffers["_ts"])
                print(
                    f"  captured {T_raw} frames over {duration:.2f}s. "
                    f"Save? [Y/n]: ",
                    end="", flush=True,
                )
                ans = ""
                while True:
                    line = stdin_sig.consume()
                    if line is not None:
                        ans = line.strip().lower()
                        break
                    time.sleep(0.05)
                if ans in ("n", "no"):
                    print("  discarded.")
                    continue

                ep_dir = out_root / f"episode_{next_idx:04d}"
                T = _save_episode_npy(
                    ep_dir, buffers, mode, image_specs, fps, task_cfg, args,
                )
                print(f"  saved {T} frames -> {ep_dir}")
                next_idx += 1
        finally:
            mode.teardown()
    finally:
        env.stop()

    print(f"[done] episodes in {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
