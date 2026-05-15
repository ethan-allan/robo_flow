# Sensor-only debug plan

Scope: bring up the **deployment pipeline with no model attached** and
visualise sensor data. This is the precondition for Phase D in
`scripts/deploy_diag/README.md`; treat that doc as the umbrella plan
and this as the focused, incremental slice that gets sensors talking
end-to-end without touching the policy/executor halves.

The robot stays powered OFF (or e-stop pressed) throughout. No motion
commands are issued in any stage of this plan.

---

## 0. What we know is broken

The deployment commit (`273f002 deplotment pipeline added - NEEDS DEBUG`)
adds the scripts + configs but **not** the runtime modules they import:

| Module | Imported by | Status |
| --- | --- | --- |
| `trainflow.env.ur3_bir.discover` | `01_visualize_sensors.py`, `02_robot_readonly.py` | missing |
| `trainflow.env.ur3_bir.sensor_clients` | `01..04`, `test_replay_clients.py` | missing |
| `trainflow.env.ur3_bir.ur3_bir_env` | `03,04`, `test_obs_builder_bitequality.py`, `env_runner/ur3_bir_runner.py` | missing |
| `trainflow.env.ur3_bir.action_executor` | `env_runner/ur3_bir_runner.py` | missing |

Until that namespace exists, **every diag script fails on import** —
that's our #1 debug target. The deploy_diag/README at-a-glance table
marks Phases B/C/D as "implemented" because the script files exist;
in fact none of them can be exec'd today. We'll fix that table when
the namespace lands.

Other gating items:

- `robo_flow` conda env not present on this host; `pyrealsense2`,
  `rtde_control`, `rtde_receive`, `cv2` not installed.
- `trainflow/dataset/peg_hole_tac/` has no episodes checked in, so
  the `from_npy_replay` path can't be smoke-tested against real data
  until we either record some or copy a sample episode in.

---

## 1. Staged plan

Each stage gates the next. Same fail-closed discipline as the
umbrella README, just compressed to the sensor-only slice. Every
stage writes its output to `/tmp/sensor_debug/<stage>/` so failures
are reproducible.

| Stage | What | Hw needed | Outcome |
| --- | --- | --- | --- |
| S0 | env + deps install | none | `python -m scripts.deploy_diag.01_visualize_sensors --help` runs |
| S1 | hardware inventory probe (no clients started) | sensors plugged in, robot OFF | discovered devices vs cfg printed; cfg holes flagged |
| S2 | per-sensor live smoke (one at a time) | per stage | each stream proven independently |
| S3 | full mosaic (existing `01_visualize_sensors.py`) | all sensors, robot OFF | 4-tile mosaic for 30 s, fps + interval histograms |
| S4 | live `env.get_obs()` loop, no model | all sensors + robot (e-stop pressed) | post-ObsBuilder rgb + lowdim plotted; matches Phase D dry run |

Stage S4 *is* "the deployment pipeline with no model attached" — it
constructs `Ur3BirEnv` and pumps frames through the full obs-source op
chain into the ring buffer, calls `env.get_obs()` at `control_fps`,
and renders / logs. No `Ur3BirRunner`, no policy, no executor.

---

## 2. Stage detail

### S0 — environment + dependency install

**Goal:** baseline that `python -m scripts.deploy_diag.01_visualize_sensors`
fails on a real import (e.g. `trainflow.env.ur3_bir`), not on
`cv2`/`pyrealsense2`/`rtde_*`.

Actions:
1. `conda create -n robo_flow python=3.10 -y && conda activate robo_flow`
2. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
3. `pip install -r requirements.txt` (adds the train-side deps the
   diag scripts also need: `omegaconf`, `hydra-core`, `cv2`, `zarr`,
   `loguru`, `scipy`).
4. `pip install pyrealsense2 ur-rtde` (deploy-only, not in
   `requirements.txt`; add them there as part of this stage so a
   second laptop can reproduce).
5. `python -c "import pyrealsense2, rtde_receive, cv2; print('ok')"` —
   gate.

Pass: `import pyrealsense2, rtde_receive, cv2` all succeed.

Fail modes worth calling out:
- `pyrealsense2` wheel is x86-64 only on linux; sanity check `uname -m`
  before debugging mysterious install errors.
- `ur-rtde` ships as `rtde_control` / `rtde_receive` modules; the pip
  name doesn't match the import name.

### S1 — hardware inventory probe

**Goal:** without opening any device, print exactly what's connected,
exactly what the cfg expects, and where they disagree. Catches "wrong
serial in cfg" / "gelsight not power-cycled" / "robot offline" in
under 5 seconds, before we waste time on streaming failures.

Surface to write: `trainflow/env/ur3_bir/discover.py` — port the
upstream functions from `tactile_data_recording/data_recording/`:

| Function | Source | Returns |
| --- | --- | --- |
| `detect_realsense_cameras()` | `auto_detect_devices.py:detect_all_cameras` | `list[{name, serial}]` |
| `detect_ur_robot(ip_list, rtde_port=30004)` | `auto_detect_devices.py:detect_ur_robot` | matched ip or None |
| `discover_gelsight_devices()` | `record_data_gui_new.py:230` | sorted list of `/dev/v4l/by-id/...-video-index0` paths |
| `gelsight_serial_from_path(path)` | `record_data_gui_new.py:_gelsight_serial_from_path` | serial string |

Script to write: `scripts/deploy_diag/00_inventory.py`. No clients
opened, no frames captured. Prints a 4-section table:

```
[realsense]    cfg.cameras.platform_realsense.serial=<x>  matched=<bool>
[realsense]    cfg.cameras.hand_realsense.serial=<x>      matched=<bool>
[gelsight]     slot 0 -> /dev/v4l/by-id/...               serial=<x>
[gelsight]     slot 1 -> /dev/v4l/by-id/...               serial=<x>
[ur3]          cfg.robot.ur3.host=<x>                     reachable=<bool> rtde=<bool>
```

Pass: every "matched" / "reachable" column is True OR the report is
plausible (e.g. one gelsight slot empty if we only have one
sensor — operator-decided).

On fail: hand-edit `trainflow/config/hardware/ur3_bir_default.yaml`
so serials/IP match. Rerun. Do NOT skip ahead with mismatched cfg —
S2/S3 failures will be much harder to diagnose downstream.

### S2 — per-sensor live smoke

**Goal:** prove each stream works in isolation. One client, one
window, 10 seconds, then close. Catches per-sensor permission /
USB-bandwidth / firmware quirks without confusing them with the
mosaic's start-order races.

Surface to write: `trainflow/env/ur3_bir/sensor_clients.py` with
three classes in **live mode** (replay mode + `from_npy_replay`
classmethod can come later for Phase A bit-equality reuse):

```python
class RealsenseClient:
    def __init__(self, cam_cfg): ...
    def start(self): ...                # spin up rs.pipeline thread
    def stop(self): ...
    def get_latest(self) -> dict: ...   # {'rgb': (H,W,3) uint8 BGR, 'ts': float}

class GelsightClient:
    def __init__(self, gs_cfg, slot: int): ...
    # same surface; cv2.VideoCapture on by-id path

class UR3Client:
    def __init__(self, robot_cfg): ...
    # get_latest returns {'tcp_pose6', 'gripper_width', 'eef_force',
    #                     'joint_state', 'current', 'control_mode', 'ts'}
```

Each client owns its own producer thread + a single-slot mailbox the
main thread reads with `get_latest()`. No shared state, no torch.

Scripts to write (or fold into one with `--only <name>`):

- `scripts/deploy_diag/01a_smoke_realsense.py --cam platform|hand --duration 10`
- `scripts/deploy_diag/01b_smoke_gelsight.py --slot 0|1 --duration 10`
- `scripts/deploy_diag/01c_smoke_ur3.py --duration 10` (read-only, RTDE
  only — same idea as `02_robot_readonly.py` but tied to S2 ordering)

Pass per sensor:
- RealSense: ≥10 fps achieved at configured resolution; no all-black
  frames after warmup; native size matches cfg `width`/`height`.
- GelSight: ≥20 fps; image isn't full-white (sensor not seated).
- UR3: TCP pose updates each tick; matches teach-pendant within
  ~1 mm/0.5°; force baseline within ±2 N stationary.

On fail: localise to that sensor before continuing. Don't progress
to S3 with a half-working RealSense — the mosaic's "no frame" tile
will mask intermittent stalls.

### S3 — full mosaic

**Goal:** the existing `01_visualize_sensors.py`, now actually
runnable. All sensors open simultaneously; 2x2 mosaic for 30 s; fps
+ interval histogram summary printed at the end. This is the
canonical Phase B from the umbrella README.

Reuses the surface from S2. New code: none — script already exists,
just needs S1+S2 surface present.

Pass criteria: as in the umbrella README Phase B (every stream ~native
rate, no stalls, serials match). Plus the additional bar that the 30-s
window finishes with no thread errors raised by any client (a quiet
`producer_errors` list).

On fail: see umbrella README Phase B fail modes. Don't continue to S4
until 10 minutes of mosaic streaming is stall-free.

### S4 — live `env.get_obs()`, no model

**Goal:** the headline ask — run the deployment pipeline end-to-end
on the live sensors, without a policy or executor in the loop, and
render the resulting per-key obs window the model *would* have seen.

Surface to write: `trainflow/env/ur3_bir/ur3_bir_env.py`. Three
responsibilities:

1. **Producer thread** — owns the sensor clients, ticks at
   `control_fps`, pulls one frame from each client, runs the per-key
   `obs_sources` op chain (the `from` + `ops` blocks composed under
   `shape_meta.obs.<key>` via the `sensor_pipelines/*.yaml`),
   appends a single-frame dict to a ring buffer of size
   `n_obs_steps * obs_temporal_downsample_ratio` (with headroom).
2. **`get_obs()`** — stacks the trailing window from the ring buffer
   into `(T_total, ...)` arrays per key, then defers to
   `trainflow.common.obs_format.format_obs_window` for the
   T_slice + downsample + reverse + cast/moveaxis/truncate. This
   second half is the bit-equality-shared path with
   `RealImageTactileDataset.__getitem__`.
3. **Lifecycle** — `start()` / `stop()` / `reset()` plus the public
   surface the tests assume (`_buffer`, `producer_errors`,
   `n_obs_steps`, `obs_downsample_ratio`, `seek_replay` for replay
   mode).

Op-chain dispatch: `obs_processing.py` already exposes
`bgr_to_rgb`, `pose7_to_pose9`, `pose9_to_pose6`, `crop_x`, `resize`,
`concat`, `moving_average_1d`, `tare_force`. The ObsBuilder needs a
small dispatcher that:
- reads `shape_meta.obs.<key>.from` (a dotted path or list thereof
  into the per-frame raw obs dict produced by clients),
- skips any op in `attr.ops` whose `name` is also in
  `hw_cfg.cameras.<cam>.applied_at_capture` (when sourcing from a
  camera),
- calls the named op with its kwargs from the cfg.

This is the ONLY new logic. The list of ops is fixed and small; a
`OP_REGISTRY = {"bgr_to_rgb": obs_processing.bgr_to_rgb, ...}` is
enough — no plugin system needed.

Script to wire: `scripts/deploy_diag/03_get_obs_live.py` (already
written; will run once `ur3_bir_env.py` exists).

Pass criteria (per umbrella README Phase D):
- 30 s @ `control_fps`, ≥99 % of ticks within the control period.
- rgb obs visually identical to a recorded episode's frame (framing,
  colour, no double-crop, no missing channel swap).
- `get_obs` per-tick latency p99 < 50 ms.
- `eef_state` follows manual teach-jog smoothly during the run; no
  dropped frames.

Out of scope at S4 (deferred to subsequent debug stages):
- `Ur3BirRunner` (inference thread + control thread).
- `ActionExecutor` (action dispatch).
- Bit-equality test (`tests/test_obs_builder_bitequality.py`) —
  needs replay mode + at least one recorded episode.

---

## 3. Minimum surface to author

Pulled out of S1–S4 so it's easy to estimate / split:

```
trainflow/env/__init__.py                 (empty)
trainflow/env/ur3_bir/__init__.py         (re-exports)
trainflow/env/ur3_bir/discover.py         (S1)
trainflow/env/ur3_bir/sensor_clients.py   (S2; live mode only here)
trainflow/env/ur3_bir/obs_builder.py      (S4; op-chain dispatch)
trainflow/env/ur3_bir/ur3_bir_env.py      (S4; producer + ring buffer + get_obs)
```

Replay-mode classmethods on the sensor clients + `from_npy_replay`
on the env can land in a follow-up; they unblock Phase A bit-equality
but aren't required to debug sensors live.

All four diag scripts (`01..04`) are deliberately not modified by
this plan — they'll start working once the surface above exists.
Same for `tests/test_replay_clients.py` and
`tests/test_obs_builder_bitequality.py` (gated on replay mode).

---

## 4. Open questions to confirm before S1 starts

1. Which physical sensors will be plugged in for S1's inventory? The
   `ur3_bir_default.yaml` schema assumes 2 RealSense + 2 GelSight; if
   we only have 1 GelSight or no hand camera, S1's pass criteria
   loosen accordingly and we should mark those cameras `enabled: false`
   in a per-deploy override yaml rather than the lab default.
2. Robot host IP — `192.168.20.25` is the cfg default. Confirm on the
   actual controller before running S1's `detect_ur_robot` probe.
3. Gripper backend — cfg references the JSON file at
   `/tmp/ur_gripper_state.json` that the capture script writes. For
   sensor-only stages we can skip gripper reads (treat it as missing
   in `UR3Client.get_latest` and let the cfg-driven `concat` op
   either omit it or default to NaN); decide before S2's UR3 client
   lands.
