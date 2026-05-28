# `demo_capture` — record UR3 teleop demos into raw `.npy` episodes

This package records human teleoperated demonstrations on the UR3-BiR
platform as raw per-episode `.npy` directories. Each episode is one
folder of one `.npy` per modality plus a `meta.yaml`. The layout matches
what `trainflow.common.zarr_writer` consumes, so you build the training
zarr on demand:

```
python -m trainflow.common.zarr_writer \
    --task trainflow/config/task/<task>.yaml --src <output>
```

If can't connect to UR then sudo ip addr add 192.168.201.100/24 dev enp3s0

Three teleop modes are supported: **spacemouse**, **gello**, and
**freedrive**.

```
demo_capture/
├── capture_demo.py        # record episodes -> raw .npy dirs
└── visualize_episode.py   # render a captured episode dir as an mp4
```

---

## 1. Prerequisites — task config

A capture run is keyed by a **task config** at
`trainflow/config/task/<your_task>.yaml`. The task config drives:

- which sensors are opened (cameras, GelSight, robot),
- which obs the model will see (`shape_meta.obs` — drives image crop/resize),
- which action encoding is recorded (`shape_meta.action`), and
- where raw episodes are written (`raw_episodes`, else `data/<name>_raw`).

`capture_demo.py` expects `shape_meta.action.shape == [9]`
(`tcp_absolute_9d`); a different action shape warns but still records the
commanded TCP pose.

Capture stores **all** signals each instantiated client emits (TCP pose,
gripper, force, joint state, current, control mode, every camera's
rgb/depth, tactile), not just the keys in `shape_meta.obs`. Image crop +
resize come from the matching `shape_meta.obs.<key>.ops` chain; images
are stored BGR (the `zarr_writer` does the BGR→RGB swap), and poses are
stored as `(T,7)` axis-angle (the `zarr_writer` does the 7→9 ortho6d
conversion).

### Minimal task config

Copy an existing one and edit the name + dataset_path. Look at
`trainflow/config/task/real_ee2_dice.yaml` for the simplest example —
it composes pre-built hardware bundles:

```yaml
# trainflow/config/task/my_task.yaml
defaults:
  - _task_base
  - /hardware/env_cfgs@hardware:                          ur3_bir_default
  - /hardware/sensor_cfgs@shape_meta.obs.rgb:             platform_realsense_rgb
  - /hardware/ur3/sensor_cfgs@shape_meta.obs.eef_state:   ur3_eef_state
  - /hardware/ur3/action_modes@shape_meta.action:         tcp_absolute_9d
  - _self_

name: my_task
image_shape: [3, 256, 256]
image_resize_shape: null
dataset_path: data/my_task        # zarr_writer output: data/my_task/replay_buffer.zarr
raw_episodes: data/my_task_raw    # capture output: data/my_task_raw/episode_NNNN/
```

### What to check before your first capture

1. `trainflow/config/hardware/env_cfgs/ur3_bir_default.yaml` — confirm
   `control_fps`, robot host IP, and the camera sensors selected.
2. `trainflow/config/hardware/ur3/ur3_base.yaml` — robot host (default
   `192.168.201.101`), gripper-state JSON path, RTDE port.
3. `trainflow/config/hardware/sensor_cfgs/platform_realsense_rgb.yaml`
   — camera serial / resolution / crop.
4. Add tactile (`gelsight_<slot>`) or extra cameras only by adding
   another `shape_meta.obs.<key>` entry mapped to the corresponding
   sensor cfg — `capture_demo` records anything in `shape_meta.obs`
   automatically; you don't need to touch the script.

If you change a hardware file mid-session, restart `capture_demo` so
the env rebuilds with the new clients.

---

## 2. Common prerequisites at runtime

Before any capture mode:

- **Robot**: power on; on Polyscope load and run the program that
  enables **Remote Control** mode; no e-stop; you can hear the brakes
  release.
- **Cameras**: RealSense (and GelSight, if your task uses tactile) are
  plugged into USB. A quick `lsusb` should list them.
- **Network**: this PC can reach the robot IP from
  `trainflow/config/hardware/ur3/ur3_base.yaml`.
- **Python**: `source venv/bin/activate` from the repo root.

Then test that everything connects without recording anything:

```bash
python -m demo_capture.capture_demo --task my_task --mode freedrive
# At the first "ready for episode 0" prompt, type:  q  ENTER
# Expect a clean exit with no episodes written.
```

---

## 3. Capturing with the SpaceMouse

The script reads a 3Dconnexion SpaceMouse, integrates the 6-DOF
deflection into a target TCP pose at every tick, and dispatches that
pose via `RtdeExecutor.servoL`. The recorded action label is the
**commanded** vector — no post-hoc shift.

### One-off install

```bash
source venv/bin/activate
pip install pyspacemouse
# Linux also needs libhidapi-dev system-wide. If pyspacemouse can't
# open the device on Linux, install via:
sudo apt install libhidapi-dev libhidapi-libusb0
```

If the device requires non-root access, add a udev rule for vendor
`256f` (3Dconnexion).

### Run

```bash
python -m demo_capture.capture_demo \
    --task my_task \
    --mode spacemouse \
    --spacemouse-trans-gain 0.02 \
    --spacemouse-rot-gain 0.3
```

- `--spacemouse-trans-gain` (m/s at full deflection). Start at `0.01`
  for the first run, then raise.
- `--spacemouse-rot-gain` (rad/s at full deflection). Start at `0.15`.
- `--fps` (defaults to `task_cfg.hardware.control_fps`).

`RtdeExecutor` clamps the per-tick TCP step using
`hardware.clients.robot.ur3.safety.max_step_m` / `max_step_rad` and the
optional `workspace_bbox`. Picking unsafe gains is then clipped, not
crashed — but tune to the smallest setting that feels responsive.

### Per-episode UI

1. Script prints `--- ready for episode N ---`.
2. Press **ENTER** → tick loop starts, robot tracks the SpaceMouse.
3. Press **ENTER** again → recording stops.
4. `Save episode (T frames, X s)? [Y/n]:` → ENTER to save, `n` to
   discard.
5. Returns to step 1. Type `q` ENTER at the start prompt to quit.

---

## 4. Capturing with GELLO

GELLO is the open-source 3D-printable leader arm
(<https://github.com/wuphilipp/gello_software>). On each tick the
script reads the leader's joint vector, runs UR3 forward kinematics
via `RTDEControlInterface.getForwardKinematics`, encodes the resulting
TCP pose to 9-D (xyz + ortho6d), and dispatches it via
`RtdeExecutor.servoL`.

### One-off install

`demo_capture` does not bundle the GELLO driver. Install upstream's
`gello_software` and ensure it's on `PYTHONPATH`:

```bash
source venv/bin/activate
git clone https://github.com/wuphilipp/gello_software.git
pip install -e gello_software/
```

You'll also need Dynamixel SDK access to the GELLO USB serial port —
typically `/dev/ttyUSB0`. If you don't have read/write permission, add
your user to `dialout`:

```bash
sudo usermod -aG dialout $USER     # log out / in to take effect
```

### Run

```bash
python -m demo_capture.capture_demo \
    --task my_task \
    --mode gello \
    --gello-port /dev/ttyUSB0
```

- `--gello-port` defaults to `/dev/ttyUSB0`.
- `--gello-agent-class` defaults to `gello.agents.gello_agent.GelloAgent`.
  Override only if you use a custom driver class. The class must expose
  one of:
  - `get_joint_state()` → ndarray of 6 joint angles in radians
    (UR3 joint order), **or**
  - `get_action()` / `act(obs)` — the script falls back to those.

### Calibration

GELLO assumes the leader joints map 1:1 onto the UR3's
`getActualQ()`. Calibrate per upstream instructions before the first
session — otherwise the FK step computes a pose for the wrong arm
configuration and the robot will jump on episode start.

The first tick reads the *current* leader joints and immediately
servos to the corresponding TCP pose. If the leader is far from the
robot's current state at episode start, that first command is large
and `RtdeExecutor`'s `max_step_m` / `max_step_rad` clip will fight it.
**Always start an episode with the leader near the follower's pose.**

### Per-episode UI

Identical to the SpaceMouse flow above.

---

## 5. Capturing with freedrive (no teleop hardware)

The script enables UR3 `teachMode()`; you move the arm by hand.
`action[t] = eef_state[t+1]` (drops the last frame).

```bash
python -m demo_capture.capture_demo --task my_task --mode freedrive
```

Useful for quickly building a dataset when you don't have a SpaceMouse
or GELLO available. Action labels are derived from the observed pose,
which is good enough for absolute-pose policies but slightly lossy
compared to true commanded actions.

---

## 6. Verifying a capture

After saving at least one episode, inspect + visualize the raw dir:

```bash
# summary of every modality (shape/dtype/range) + render an mp4:
python -m demo_capture.visualize_episode data/my_task_raw --episode -1
# (writes data/my_task_raw/episode_NNNN/preview.mp4)

# summary only, no mp4:
python -m demo_capture.visualize_episode data/my_task_raw --summary-only
```

On disk (raw, pre op-chain) you should see, all sharing the same `T`:

- `rgb.npy`: `(T, H, W, 3) uint8` **BGR** (cropped/resized)
- `eef_state.npy` / `eef_action.npy`: `(T, 7)` `[xyz, axis-angle, gripper]`
- `eef_force.npy` `(T,6)`, `joint_state.npy` `(T,7)`, `current.npy` `(T,6)`,
  `control_mode.npy` `(T,1)`, `frame_timestamp.npy` `(T,)`
- `meta.yaml` — task, mode, fps, timestamps, per-file shapes, git commit

`--episode -1` picks the most recent episode; you can also pass a single
`episode_NNNN/` dir. See `visualize_episode.py --help` for layout options
when the task has more than one rgb key.

---

## 7. End-to-end check

Build the training zarr from the raw episodes, then train one step:

```bash
# raw .npy dirs -> replay_buffer.zarr (does BGR->RGB + pose 7->9):
python -m trainflow.common.zarr_writer \
    --task trainflow/config/task/my_task.yaml --src data/my_task_raw --overwrite

python train.py --config-name=train_diffusion_unet_real_image_workspace \
    task=my_task training.max_train_steps=10
```

If `zarr_writer` complains about a missing file or a `T` mismatch, the
raw episode is malformed; if the dataset loader complains about a shape
or key, the capture config and the model config disagree — start from
the task yaml.

---

## Notes

- Multiple capture sessions **append**: numbering continues past any
  existing `episode_*` dirs in the output. To start fresh, `rm -rf` the
  raw output dir before running.
- `Ctrl+C` during recording exits abruptly without saving the current
  episode (episodes already written to disk are preserved).
- The current spacemouse and gello modes pass `rtde_receive=None` to
  `RtdeExecutor`, so the force-abort safety is **off**. If your task
  involves contact, plumb the receiver through before relying on
  force-abort.
