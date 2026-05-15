# Phase H — policy dry-run setup

Goal: load a trained checkpoint, run inference on live obs, log the
predicted actions (decoded through the task's action_sinks chain), do
NOT command the robot.

## 1. Bring the checkpoint over

You need a `<run_dir>/` from a training run. The directory has to
contain at minimum:

```
<run_dir>/
├── .hydra/config.yaml          # the full resolved cfg the run trained with
└── checkpoints/
    ├── latest.ckpt             # always present
    └── epoch=XXXX-val_action_mse=Y.YYYY.ckpt   # zero or more top-k ckpts
```

Suggested scp:
```bash
# from the training host
rsync -avh --progress \
    <training_host>:/path/to/data/outputs/<date>/<time>_train_diffusion_unet_image_real_ee2_dice \
    /home/labpc2x2080ti/ethan/robo_flow/data/outputs/<date>/
```

If `data/outputs/` doesn't exist yet locally, the rsync will create it.

## 2. Populate camera serials

The task yaml's `shape_meta.obs.rgb.client.serial` is `null` by default.
Pick a stable place to override:

**Option A — task yaml (simplest, per-deploy):**
```yaml
# trainflow/config/task/real_ee2_dice.yaml — add to the _self_ section
shape_meta:
  obs:
    rgb:
      client:
        serial: 248622300418   # D455 platform (discovered via 00_inventory)
```

Don't edit `sensor_cfgs/platform_realsense_rgb.yaml` directly — it's
the shared schema. Per-rig values belong in env_cfgs or task yamls.

## 3. Robot

For `real_ee2_dice` (which needs `eef_state` from RTDE):

- Power on the UR3 controller.
- Press the e-stop (or leave the robot in idle).
- `RTDEReceiveInterface` reads pose/force without needing
  `RTDEControlInterface` — no motion is commanded by this script.
- Confirm reach: `python -m scripts.deploy_diag.00_inventory --task real_ee2_dice`
  should print `=== inventory: OK ===`.

## 4. Run the dry-run

```bash
python -m scripts.deploy_diag.08_policy_dry_run \
    --task real_ee2_dice \
    --run-dir data/outputs/<date>/<time>_..._real_ee2_dice/ \
    --duration 60 \
    --out /tmp/policy_dry_run
```

Outputs:
- `/tmp/policy_dry_run/ticks.jsonl` — one JSON line per inference tick:
  ```
  {"t": ..., "latency_ms": ..., "action_pred_first": [...9..],
   "decoded": {"tcp_target": [x,y,z,rx,ry,rz]}, "eef_state_last": [...]}
  ```
- Stdout summary: tick count, latency p50/p99/max, on-budget %.

## 5. What to look for

Pass criteria (matches the original deploy_diag/README Phase H):

- **Latency p99 < `(1/inference_fps) - 50ms`.** At 5 Hz inference, that's
  p99 < 150ms. If you blow this, increase `--inference-fps` overhead
  budget or check if cuDNN is initialising every tick.
- **Decoded `tcp_target` stays in plausible workspace bounds** — i.e.
  not jumping outside a few cm of the current `eef_state.last`.
- **Predictions are stable when the robot is idle.** Stand still →
  predicted TCP target should be near-stationary (within mm).
- **No NaNs in the action vector.** A `predict_action` returning NaN
  is a normalizer/checkpoint mismatch.

## 6. After it works

- Plot `ticks.jsonl` offline to inspect the action sequence.
- If the trajectory looks reasonable, Phase I is the next step (real
  execution with the safety-wrapped `action_executor`). That requires
  writing `trainflow/env/ur3_bir/action_executor.py` — out of scope
  for this phase.

## 7. Known gotchas

- The current `vrr_19d.yaml` action mode only declares the `tcp_target`
  sink (the `virtual_target` and `stiffness` sinks were dropped in the
  May 15 refactor). If you're testing a VRR-trained checkpoint, the
  decoder will only emit `tcp_target` — the model still outputs a 19-D
  vector but only 9 dims are decoded.
- If `policy.predict_action` returns 13-D actions (rpy_for_rotation
  collapsed VRR), the decoder will fail because the task's
  `shape_meta.action.shape == [19]` and slice ranges expect 19-D input.
  Phase H currently does NOT re-expand collapsed-rpy actions.
- The `test_replay_clients.py:test_live_mode_unimplemented` test is
  stale (asserts `NotImplementedError` from live `get_latest()`).
  Delete or skip when you next run pytest.
