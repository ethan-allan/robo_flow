# Deployment hardware test plan

Safety-first sequence to validate the deploy stack. Each phase MUST
fully pass before moving to the next. Two principles:

1. **Validate the data path before commanding any motion.** Sensors,
   parsing, and obs_builder must be proven against recorded data, then
   against live sensors with the robot powered down, before the robot
   is asked to move.
2. **Fail closed.** Every motion phase has an e-stop drill, a max-speed
   override, and a pre-flight workspace check. No motion phase runs
   without an operator with their hand on the e-stop.

Phases A–C are independent of step 5 (live sensor mode) and step 6
(action executor). Phases D–I depend on those landing.

| Phase | What | Hw needed | Step dep |
| --- | --- | --- | --- |
| A | Offline replay + viz | none | step 4 (DONE) |
| B | Sensors-only live | cameras + gelsight; **robot OFF** | step 5 |
| C | Robot read-only via RTDE | robot ON, e-stop pressed | step 5 |
| D | Live `get_obs()` dry run | all sensors + robot, manual jog | step 5 |
| E | Action decode dry run (no execute) | none | step 6 |
| F | Single jog-to-start | robot ON, e-stop ready | step 6 |
| G | Replay-as-action with execution, slow | full stack | step 6 |
| H | Trained policy, inference only (no execute) | full stack | step 6 |
| I | Trained policy, execute, bounded + slow | full stack | step 6 |

---

## Phase A — offline replay + visualisation

**Goal:** confirm the deploy preprocessing chain (replay clients ->
obs_builder -> env.get_obs) produces the same per-key tensors the
model saw during training. Bit-equality is verified by a unit test;
this phase additionally renders frames + timeseries so a human can
eyeball whether obs look right.

**Pre-flight:** none.

**Run:**
```bash
conda activate robo_flow
cd trainflow

# Bit-equality test — must pass for all 3 task yamls.
python -m tests.test_obs_builder_bitequality

# Replay viewer — pick any episode under trainflow/dataset/peg_hole_tac.
python -m scripts.deploy_diag.04_replay_visualize \
    --task peg_insertion_vrr_5fps \
    --episode trainflow/dataset/peg_hole_tac/<ep_id> \
    --n-obs-steps 2 \
    --out /tmp/replay_viz \
    --legacy-rgb-ops              # legacy data was already crop+resized at capture
```

**Pass criteria:**
- bit-equality test: 3/3 PASS.
- `/tmp/replay_viz/rgb_t*.png` look correct (not channel-swapped, not
  cropped wrong, no artefacts).
- `timeseries.png` `eef_state` xyz/rotation tracks recorded motion
  smoothly. `eef_force` shows expected contact spikes.
- `summary.txt` value ranges plausible (rgb in [0,1], force in
  Newtons, state in metres+radians).

**On fail:** the issue is in obs_processing ops, the obs_sources op
chain, or the env's temporal+formatting. Fix offline. Do NOT proceed.

---

## Phase B — sensors-only live (robot powered OFF)

**Goal:** confirm every camera and tactile sensor opens, produces
sane frames at the configured rate, and that the serials in
`config/hardware/ur3_bir_default.yaml` actually point at the sensors
you have plugged in.

**Pre-flight:**
- Robot controller powered off OR e-stop pressed.
- Lab clear of personnel near the workspace.
- Cameras + gelsights plugged in.

**Run** (script TBD when step 5 lands):
```bash
python -m scripts.deploy_diag.01_visualize_sensors \
    --hw config/hardware/ur3_bir_default.yaml \
    --duration 30
```

The script will:
1. Open `cameras.platform_realsense` and `cameras.hand_realsense` by
   serial. List discovered serials side-by-side with the cfg.
2. Spawn the GelSight v4l auto-discovery (`discover_gelsight_devices`
   from the capture-script lineage). List by-id paths discovered.
3. For 30 s, render an OpenCV mosaic: platform RGB | hand RGB |
   gelsight 0 | gelsight 1.
4. Print fps achieved per stream, native resolution, and a histogram
   of frame intervals (catches stalls).

**Pass criteria:**
- Configured serials matched discovered serials. If null, paste them
  back into the hw cfg before continuing.
- All four streams visible, ~15 fps cameras, ~25 fps gelsight, no
  black frames after warmup, no stall warnings.

**On fail:** wrong serial → fix cfg. Stalled stream → check USB
bandwidth / cable / port. v4l discovery returns nothing → check
`/dev/v4l/by-id` for `usb-GelSight_*` entries; permissions.

**Stop condition before B is "passed":** drain a 10-min recording
through this script with no stalls. Sensor stalls during deployment
are the #1 cause of subtle bit-inequality.

---

## Phase C — robot read-only (RTDE)

**Goal:** confirm RTDE connection, gripper socket connection, force
sensor reads. **No motion commands sent.**

**Pre-flight:**
- Robot powered ON, in remote mode.
- E-stop within reach.
- Operator visually confirms robot is in a known, safe pose before
  starting.
- Network: `ping <robot_host>` succeeds.

**Run** (script TBD step 5):
```bash
python -m scripts.deploy_diag.02_robot_readonly \
    --hw config/hardware/ur3_bir_default.yaml \
    --duration 60 --rate 5
```

The script will:
1. Open `RTDEReceiveInterface(host)` per `auto_detect_devices.py`
   (ping + port + RTDE handshake before assuming connection).
2. Open the Robotiq socket (`port 63352`).
3. For 60 s at 5 Hz, print: tcp_pose6 (m, rad), gripper_width (m),
   eef_force6 (N, Nm), joint_state7. Compare tcp_pose against the
   teach-pendant readout — they must match.
4. Compute a "no-motion baseline": stddev of each scalar over 60 s
   while the robot is stationary. Anything > 1e-3 m or 1e-3 rad is
   suspicious (cable noise, driver bug).

**Pass criteria:**
- Connection established cleanly, no `RTDE handshake failed`.
- TCP pose readout matches teach-pendant within ~1 mm, 0.5°.
- Gripper socket reads a sane width (0..max) at the resting state.
- 60-s stationary stddev within tolerance.
- Force baseline within ±2 N of zero on each axis (apply tare in
  obs_processing if needed — `tare_force` already exists).

**On fail:** wrong IP → fix hw cfg `robot.ur3.host`. Wrong port →
check controller URCap. Wrong gripper port → confirm Robotiq URCap
exposes 63352. Excessive noise → check ground / cable.

---

## Phase D — live `get_obs()` dry run

**Goal:** confirm the env's full obs assembly works in live mode at
the control rate, with all sensors + robot connected. Robot may be
manually jogged via teach pendant during this; **the script never
sends a command.**

**Pre-flight:** B + C passed.

**Run** (script TBD step 5):
```bash
python -m scripts.deploy_diag.03_get_obs_live \
    --task peg_insertion_vrr_5fps \
    --duration 30 \
    --out /tmp/obs_live
```

The script will:
1. Build `Ur3BirEnv` in live mode with `n_obs_steps=2`,
   `obs_temporal_downsample_ratio=1`.
2. For 30 s at the cfg `control_fps`, call `env.get_obs()`. Render
   the trailing rgb obs in a window, plot eef_state/eef_force as a
   rolling timeseries, log per-tick latency.
3. Save same artifact bundle as Phase A (so a side-by-side with the
   recorded episode is possible — first sanity check that live ==
   replay-shaped).

**Pass criteria:**
- 30 s with ≥99% of ticks completing within the control period.
- rgb obs visually identical in framing/color to a recorded episode.
- Per-tick `get_obs` latency p99 < 50 ms (well under 200 ms control
  budget at 5 Hz).
- During manual jog, obs `eef_state` follows the motion smoothly with
  no dropped frames.

**On fail:** latency spikes → profile per-sensor get_latest. Frame
drops → check sensor thread health (one per camera, gelsight, force).
rgb framing wrong → cfg crop_x bounds, camera mount.

---

## Phase E — action decode dry run (no execute)

**Goal:** validate the `action_sinks` dispatcher decodes a recorded
action vector into the same actuator commands the robot received
when the episode was collected. Pure offline, no hardware.

**Pre-flight:** step 6 (`action_executor.py`) implemented.

**Run** (script TBD step 6):
```bash
python -m scripts.deploy_diag.05_action_dry_run \
    --task peg_insertion_vrr_5fps \
    --episode trainflow/dataset/peg_hole_tac/<ep_id>
```

The script will:
1. Load `vrr_action.npy` (or `eef_action.npy` for non-VRR).
2. For each frame, push the action vector through the `action_sinks`
   dispatcher to produce per-sink decoded values.
3. For `tcp_target`, compare decoded pose6 against
   `eef_action.npy` (the recorded RTDE target). For VRR
   `virtual_target`, compare against… a recomputed one via the
   inverse of `prepare_vrr.build_vrr_action`.
4. Per-frame max abs deviation; warn if > 1e-5.

**Pass criteria:**
- All sinks decode without exception.
- `tcp_target` vs recorded `eef_action.npy` matches within float32
  precision.
- VRR `stiffness` decoded scalars within 1e-5 of recomputed values.

**On fail:** the bug is in `pose9_to_pose6`, the action_sinks slice
ranges, or the action executor's dispatch. Fix before any phase
sends a command.

---

## Phase F — single jog-to-start

**Goal:** the FIRST commanded motion. One slow `moveL` to a known-safe
"home" pose well above the workspace.

**Pre-flight:**
- Phases A–E passed.
- Workspace clear, fixture removed if it could be hit.
- Speed override on teach pendant set to 25%.
- Operator standing at e-stop, second person observing.
- "Home" pose chosen and entered into the script — **far from any
  surface, gripper open, joints in a safe configuration.**

**Run** (script TBD step 6):
```bash
python -m scripts.deploy_diag.06_jog_to_start \
    --hw config/hardware/ur3_bir_default.yaml \
    --target-pose <x y z rx ry rz>   # TCP pose, metres + axis-angle
    --speed 0.05                      # m/s — very slow
    --accel 0.05                      # m/s^2
    --confirm                         # require interactive yes/no
```

The script will:
1. Connect RTDE control + receive.
2. Print current pose vs target pose.
3. Wait for keyboard "y" to proceed.
4. Issue `moveL(target_pose, speed=0.05, accel=0.05)`.
5. Block until completion or 30 s timeout. On timeout, abort.

**Pass criteria:**
- Robot reaches the target pose within tolerance.
- No protective stop, no joint limit hit, no force spike.
- Operator did not touch e-stop.

**On fail:** connection or kinematics issue. **DO NOT** rerun without
diagnosing why — repeated abort attempts can confuse the controller.

---

## Phase G — replay-as-action, executed slowly

**Goal:** play a recorded episode's actions through the executor with
the robot actually moving, at reduced speed, to validate that the
full command path produces motion that matches what was recorded.

**Pre-flight:** F passed. Operator at e-stop. Speed override 25%.

**Run** (script TBD step 6):
```bash
python -m scripts.deploy_diag.07_replay_execute \
    --task peg_insertion_vrr_5fps \
    --episode trainflow/dataset/peg_hole_tac/<ep_id> \
    --speed-scale 0.25 \             # 1/4 of recorded velocities
    --max-step-m 0.01 \              # per-tick clip
    --max-step-rad 0.05 \
    --workspace-bbox <xmin xmax ymin ymax zmin zmax> \
    --confirm
```

Safety wrapper applied to every action before execution:
- TCP target clipped into the workspace bbox.
- Per-tick max delta clipped (no warp jumps).
- Force monitor: abort + protective-stop if `||eef_force[xyz]|| > 30 N`
  or `||eef_force[rot]|| > 5 Nm`.
- Inference rate halved vs recording (so velocity scales down 1/2; with
  speed-scale 0.25 the effective velocity is ~1/8 of recorded).

**Pass criteria:**
- Episode completes without abort.
- Final pose within 5 mm / 1° of recorded final pose.
- No stall in the executor's command thread.

**On fail (and ANY abort):** stop. Diagnose force trace, abort
reason, executor logs. Do not move on.

---

## Phase H — trained policy, inference only (NO execution)

**Goal:** run the trained checkpoint against live obs, log what it
would have commanded, but do not send anything to the robot.

**Pre-flight:** G passed. Robot at home pose, idle.

**Run** (script TBD step 6):
```bash
python -m scripts.deploy_diag.08_policy_dry_run \
    --task peg_insertion_vrr_5fps \
    --ckpt path/to/checkpoint.ckpt \
    --duration 60 \
    --out /tmp/policy_dry_run
```

The script will:
1. Build env in live mode (no executor wired).
2. Load checkpoint, run inference at the configured `inference_fps`.
3. For each step, log `action_pred`, decode via `action_sinks`, plot
   the would-have-commanded TCP target overlaid on the current pose
   on a 3D plot.
4. Compute residual: `||tcp_target - tcp_actual||` per tick. A
   well-behaved policy near-stationary should produce near-zero
   residual when the robot isn't moving.

**Pass criteria:**
- Inference latency p99 < `1/inference_fps - 50ms`.
- Residual analysis: `tcp_target` is within plausible bounds (not
  jumping outside workspace, not spitting NaNs).
- VRR stiffness in plausible range (`k_min`–`k_max`).

**On fail:** model/cfg mismatch (shape_meta vs checkpoint), wrong
normalizer, or model emitting garbage. Do NOT proceed.

---

## Phase I — trained policy, executed, bounded + slow

**Goal:** the first end-to-end run with model in the loop. Heavily
bounded, single trial, full operator attention.

**Pre-flight:** H passed. Workspace prepared with the actual task
fixture. Operator at e-stop. Second observer.

**Run** (script TBD step 6):
```bash
python -m scripts.deploy_diag.09_policy_execute_bounded \
    --task peg_insertion_vrr_5fps \
    --ckpt path/to/checkpoint.ckpt \
    --max-trial-duration 30 \
    --max-step-m 0.005 \
    --max-step-rad 0.02 \
    --workspace-bbox ... \
    --force-abort 25 \              # N
    --speed-scale 0.5 \
    --confirm
```

Same safety wrapper as Phase G, plus:
- Hard 30-s trial timeout. Abort + retract to home.
- Single trial per script invocation. To run again, re-confirm.
- All actions logged so failure can be reconstructed.

**Pass criteria:**
- Trial completes within timeout, OR aborts cleanly on bound hit.
- No protective stop, no operator e-stop required.
- Logs allow full replay of obs+action sequence offline.

**On fail (whether protective-stop or bound-hit):** stop the session.
Review logs. Determine root cause before next trial.

---

## After Phase I

Gradually loosen bounds: increase `speed-scale`, widen workspace
bbox, raise force limit, increase max trial duration. Each loosening
is its own confirm-required script run, and the operator stays at
e-stop until you're confident in repeatability.

Re-baseline (rerun Phase A bit-equality + Phase D live get_obs)
whenever:
- Hardware cfg changes (serial swap, sensor remount, controller
  firmware update).
- The training data distribution shifts (new dataset bundled).
- The model checkpoint is updated.

---

## Test scripts at-a-glance

| File | Phase | Status |
| --- | --- | --- |
| `tests/test_obs_builder_bitequality.py` | A | implemented |
| `scripts/deploy_diag/04_replay_visualize.py` | A | implemented |
| `scripts/deploy_diag/01_visualize_sensors.py` | B | implemented |
| `scripts/deploy_diag/02_robot_readonly.py` | C | implemented |
| `scripts/deploy_diag/03_get_obs_live.py` | D | implemented |
| `scripts/deploy_diag/05_action_dry_run.py` | E | TODO (step 6) |
| `scripts/deploy_diag/06_jog_to_start.py` | F | TODO (step 6) |
| `scripts/deploy_diag/07_replay_execute.py` | G | TODO (step 6) |
| `scripts/deploy_diag/08_policy_dry_run.py` | H | TODO (step 6) |
| `scripts/deploy_diag/09_policy_execute_bounded.py` | I | TODO (step 6) |
