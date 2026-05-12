# robo_flow

A training-only package for **diffusion / flow / reactive diffusion policy** on robotic manipulation data, derived from [Chen-Wendi/trainflow](https://github.com/Chen-Wendi/trainflow) (MIT, © 2026 Wendi Chen) with the deployment, tactile, and force halves removed.

You bring a zarr replay buffer of `(image, state, action)` trajectories; this package trains a policy on it via Hydra + accelerate + W&B.

Three policy variants ship out of the box:

| Workspace config                                            | Policy                    | Wrapper script       |
| ----------------------------------------------------------- | ------------------------- | -------------------- |
| `train_diffusion_unet_real_image_workspace`                 | DDPM/DDIM UNet            | `train_dp.sh`        |
| `train_diffusion_transformer_real_image_workspace`          | Transformer denoiser      | `train_dpt.sh`       |
| `train_reactive_diffusion_transformer_real_image_workspace` | Reactive Diffusion Policy | `train_rdp.sh`       |

---

## 1. Install

```bash
# conda env (Python 3.10) — works on driver CUDA 12.8
conda create -n robo_flow python=3.10 -y
conda activate robo_flow

# torch matching your CUDA driver (cu128 wheels work on driver 12.x via forward compat)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# remaining deps
pip install -r requirements.txt

# one-time setup
accelerate config         # pick GPU / precision
wandb login               # only if you want logging.mode=online
```

---

## 2. Mental model

The whole pipeline is driven by **one config tree** composed at run time by Hydra:

```
workspace yaml  ← chooses policy class, scheduler, EMA, optimizer, loop length
└── task yaml   ← chooses dataset (zarr path + shape_meta)
```

At launch, `train.py`:

1. Resolves the config (`workspace yaml + task yaml + CLI overrides`)
2. Instantiates the workspace class named in `_target_`
3. Calls `workspace.run()`, which builds dataset → dataloader → policy → optimizer → EMA → loops

**`shape_meta`** is the single source of truth — every component (dataset, encoder, policy) reads it. Once your zarr keys match `shape_meta`, no Python changes are needed.

```
trainflow/
├── train.py                      # Hydra entry point
├── inspect_zarr.py               # script to dump structure/shapes/plots from any zarr
├── convert_ee2dice_to_zarr.py    # example raw-→-zarr converter
├── scripts/                      # per-experiment launcher .sh wrappers (this is where YOU work)
├── requirements.txt
└── trainflow/
    ├── common/      # replay buffer, sampler, action utils, normalizers, EMA
    ├── config/
    │   ├── train_*_workspace.yaml      # workspace-level (paradigm) configs
    │   └── task/<task>.yaml            # per-dataset (task) configs
    ├── dataset/     # zarr-backed BaseImageDataset
    ├── model/       # diffusion (UNet1D, transformer), vision (resnet, timm)
    ├── policy/      # DiffusionUnet / DiffusionTransformer image policies
    └── workspace/   # accelerate-driven training loops
```

---

## 3. What each script does

| Script | Purpose |
|---|---|
| `train.py` | Hydra entry. Resolves `--config-name=<workspace>` + `task=<task>` + CLI overrides, instantiates the workspace class, calls `.run()`. Never edit this. |
| `convert_ee2dice_to_zarr.py` | Example raw-`.npy` → DP-style zarr converter for the ee2dice dataset layout. Copy + adapt for your own raw format. |
| `inspect_zarr.py` | Diagnostic. Prints the zarr tree, shapes, dtypes, episode stats, value ranges. Saves a frame grid + lowdim/action plots to `inspect_out/`. Run any time you build a new zarr. |
| `trainflow/notebooks/inspect_zarr.ipynb` | Same as the script but interactive — same knobs at the top of cell 1. |
| `scripts/train_<task>_<paradigm>.sh` | Per-experiment wrapper. Holds the `accelerate launch` invocation + the hyperparameters you care to tune. This is the file you edit + commit. |

---

## 4. Add a new training configuration (task)

Each *training configuration* = a new zarr dataset wired to one of the existing policy paradigms. Five steps.

### Step 1 — convert your raw data to zarr

The trainer reads a single zarr directory in this layout:

```
<dataset_path>/replay_buffer.zarr/
├── meta/
│   └── episode_ends             # int64, cumulative episode boundaries
└── data/
    ├── <camera_key>             # (N, H, W, 3) uint8
    ├── <state_key>              # (N, D)       float32
    └── action                   # (N, A)       float32
```

`N` is total frames across all episodes concatenated. `episode_ends[i]` is the index just past the last frame of episode `i`.

If you have ee2dice per-episode `.npy` dumps (`eef_state.npy`, `eef_action.npy`, `rgb.npy`, ...):

```bash
python convert_ee2dice_to_zarr.py \
    --src trainflow/dataset/ee2dice \
    --dst data/ee2dice_zarr
```

For any other source format, copy `convert_ee2dice_to_zarr.py` and adapt the `load_episode` function. Rotations should be encoded as 6D (Zhou et al. 2019); the file has a reference `axis_angle_to_rot6d` you can reuse.

### Step 2 — sanity-check the zarr

```bash
python inspect_zarr.py --zarr data/<your_zarr>
```

You should see (a) all expected keys present under `data/`, (b) image frames showing actual content, (c) lowdim plots showing the trajectories vary sensibly. If `rgb` is all-zero or rotation is exactly constant across frames, you have a converter bug — fix before training.

### Step 3 — write a task YAML

Copy a template and edit three things — the `name`, the `dataset_path`, and the `shape_meta` keys:

```bash
cp trainflow/config/task/real_flip_image_dp_10fps.yaml \
   trainflow/config/task/<your_task>.yaml
```

```yaml
name: <your_task>                         # used in run-dir / wandb name

image_shape: [3, 256, 256]                # C, H, W of your camera
dataset_path: data/<your_zarr>

shape_meta: &shape_meta
  obs:
    rgb:                                  # MUST match a key under data/ in your zarr
      shape: ${task.image_shape}
      type: rgb
    eef_state:                            # MUST match the lowdim key in your zarr
      shape: [9]                          # full width of that array
      type: low_dim
  action:
    shape: [9]                            # MUST match data/action width

dataset:
  _target_: trainflow.dataset.real_image_tactile_dataset.RealImageTactileDataset
  shape_meta: *shape_meta
  dataset_path: ${task.dataset_path}
  horizon: ${horizon}
  pad_before: ${eval:'${dataset_obs_steps}-1+${n_latency_steps}'}
  pad_after: ${eval:'${n_action_steps}-1'}
  n_obs_steps: ${dataset_obs_steps}
  n_latency_steps: ${n_latency_steps}
  seed: 42
  val_ratio: 0.1
  max_train_episodes: null
  delta_action: False
  relative_action: False                  # set True ONLY if actions are 9D xyz+rot6d AND you want them recomputed relative to window start
```

Naming heuristic in the dataset class:
- a lowdim key whose name contains `robot_tcp_pose` gets a pose-aware normalizer
- a key with `wrt` in the name triggers bimanual code (avoid)

### Step 4 — smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=<your_task> \
    policy.obs_encoder.resize_shape=null \
    training.debug=True \
    training.num_epochs=1 \
    training.max_train_steps=5 \
    dataloader.batch_size=8 \
    logging.mode=disabled
```

The `resize_shape=null` override is needed when your images aren't 360×640 (the upstream default). Set to `null` to pass through, or `[H, W]` to resize.

If this completes with a checkpoint in `data/outputs/.../checkpoints/`, the wiring is correct.

### Step 5 — wrap a launcher script

Create `scripts/train_<your_task>_dp.sh` (use `scripts/train_ee2dice_dp.sh` as a template). All the hyperparameters live in a `CONFIG` block at the top:

```bash
bash scripts/train_<your_task>_dp.sh
```

Commit the script. You can now reproduce that experiment by checking out the same commit.

---

## 5. Add a new paradigm

A *paradigm* = a new training loop / policy / loss combo (e.g. flow matching, IBC, behavior cloning baseline). Four steps.

### Step 1 — write the policy class

New file at `trainflow/policy/<your_policy>.py`. Inherit from `policy/base_image_policy.py:BaseImagePolicy` and implement:

```python
class YourPolicy(BaseImagePolicy):
    def __init__(self, shape_meta, obs_encoder, model, horizon, n_obs_steps, ...):
        super().__init__()
        # build modules

    def compute_loss(self, batch):
        # batch = {'obs': {key: tensor}, 'action': tensor}
        # return scalar loss

    def predict_action(self, obs_dict):
        # return {'action': tensor, 'action_pred': tensor}

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
```

Reuse `obs_encoder.output_shape()` to size your conditioning, and reuse `self.normalizer` for action normalization.

### Step 2 — write (or reuse) a workspace class

If your training loop is just diffusion-with-a-different-loss, **reuse** `workspace/train_diffusion_unet_image_workspace.py` and only override the policy in the workspace YAML.

If you need a different loop (e.g. a two-stage train), copy `train_diffusion_unet_image_workspace.py` to `workspace/train_<your_paradigm>_workspace.py` and adapt the `run()` method.

### Step 3 — write a workspace YAML

Copy `config/train_diffusion_unet_real_image_workspace.yaml` to `config/train_<your_paradigm>_workspace.yaml`. Change:

```yaml
_target_: trainflow.workspace.<your_workspace_module>.<YourWorkspaceClass>

policy:
  _target_: trainflow.policy.<your_policy_module>.YourPolicy
  # policy-specific kwargs
```

Keep `obs_encoder`, `ema`, `dataloader`, `optimizer`, `training`, `logging`, `checkpoint` blocks — they're paradigm-agnostic.

### Step 4 — launch

```bash
accelerate launch train.py \
    --config-name=train_<your_paradigm>_workspace \
    task=<any_existing_task>
```

The same task YAMLs work — `shape_meta` is paradigm-agnostic.

---

## 6. Tweak hyperparameters — three patterns

Pick the right tool for the job.

### A. One-off CLI overrides (probes)

Anything in the resolved config tree is overridable on the CLI. Good for "let me see what happens if…" probes you won't re-run.

```bash
accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=<your_task> \
    horizon=8 \
    n_obs_steps=4 \
    optimizer.lr=5e-5 \
    dataloader.batch_size=16
```

Not reproducible — leaves no trace in git. Use only for exploration.

### B. Wrapper `.sh` script (the default for tracked experiments) — recommended

`scripts/train_ee2dice_dp.sh` is a template. Its `CONFIG` block holds every knob you tune; everything else is plumbing.

```bash
# --- CONFIG ------------------------------------------------------------------
RUN_NAME=ee2dice_dp_v1
TASK=real_ee2_dice
HORIZON=16
N_OBS_STEPS=2
NUM_EPOCHS=200
BATCH_SIZE=32
LR=1.0e-4
RESIZE_SHAPE=null
SHARE_RGB_MODEL=False
PREDICTION_TYPE=epsilon
LOGGING_MODE=online
TOPK_K=3
CHECKPOINT_EVERY=20
# -----------------------------------------------------------------------------
```

Workflow:

```bash
# edit the CONFIG block
$EDITOR scripts/train_ee2dice_dp.sh

bash scripts/train_ee2dice_dp.sh           # run
git diff scripts/train_ee2dice_dp.sh       # what did I change vs last run?
git commit -am "ee2dice: lr 1e-4 -> 5e-5"  # snapshot
```

`git log` becomes your experiment journal. Use this for any experiment you might want to re-run.

### C. Custom workspace YAML (when you have multiple named profiles for one task)

Copy + override:

```yaml
# trainflow/config/train_ee2dice_dp_small_workspace.yaml
defaults:
  - train_diffusion_unet_real_image_workspace
  - override task: real_ee2_dice
  - _self_

horizon: 16
n_obs_steps: 2
training.num_epochs: 200
dataloader.batch_size: 32
val_dataloader.batch_size: 32
policy.obs_encoder.resize_shape: null
checkpoint.topk.k: 3
```

```bash
accelerate launch train.py --config-name=train_ee2dice_dp_small_workspace
```

Pros: the full resolved config snapshot at `data/outputs/.../.hydra/config.yaml` matches the YAML exactly — easier to audit. Use this when you want "small" vs "big" vs "ablation" as separate, named profiles for one task.

### Where does each knob live?

| Knob category | Lives in |
|---|---|
| `shape_meta`, `dataset_path`, dataset class kwargs | `config/task/<task>.yaml` |
| architecture defaults (UNet dims, scheduler, EMA), optimizer skeleton, training loop length | `config/train_*_workspace.yaml` |
| per-run-i-tune-often values (lr, batch, epochs, encoder flags) | `scripts/train_*.sh` CONFIG block |
| one-off probes | CLI overrides |

Don't edit the upstream workspace YAMLs in place for experiments — they're shared defaults.

### Vision encoder swap

To use DinoV2 instead of ResNet18, comment out the `obs_encoder` (ResNet) block in `config/train_diffusion_unet_real_image_workspace.yaml` and uncomment the `TimmObsEncoder` block beneath it. That can also be done as a CLI override or in a custom workspace YAML.

---

## 7. Sweep hyperparameters

Hydra has built-in `--multirun` (a.k.a. `-m`) that launches one job per cell of a grid.

### Grid sweep

```bash
accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_ee2_dice \
    policy.obs_encoder.resize_shape=null \
    --multirun \
    optimizer.lr=1e-4,5e-5,2e-5 \
    horizon=8,16,24
```

Launches 3×3=9 runs sequentially. Each gets its own dir under `data/outputs/<date>/<time>_<name>_<task>/<hydra.job.num>/`.

### Range syntax

```bash
accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_ee2_dice \
    --multirun \
    'optimizer.lr=range(1e-5, 1e-3, 5)' \
    'dataloader.batch_size=choice(16, 32, 64)'
```

### Wrap a sweep in a script

`scripts/sweep_ee2dice_lr.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_ee2_dice \
    policy.obs_encoder.resize_shape=null \
    task_name=sweep_lr_v1 \
    logging.mode=online \
    --multirun \
    optimizer.lr=1e-4,5e-5,2e-5,1e-5 \
    horizon=8,16
```

Each run logs to W&B under a separate run; group them in the UI by `task_name`.

### Bayesian / TPE sweep (optional plugin)

For real hyperparameter search rather than a grid, install the Hydra Optuna sweeper:

```bash
pip install hydra-optuna-sweeper
```

Then add to your workspace YAML:

```yaml
defaults:
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    sampler:
      _target_: optuna.samplers.TPESampler
    direction: minimize
    n_trials: 20
    params:
      optimizer.lr: range(1e-5, 1e-3, log=true)
      horizon: choice(8, 16, 24)
```

And launch with `--multirun`. Each run reports its final val_loss back to Optuna.

---

## 8. Outputs

Each run writes to `data/outputs/<date>/<time>_<workspace_name>_<task_name>/`:

```
checkpoints/
  latest.ckpt
  epoch=0090-train_loss=0.041.ckpt   # top-k by checkpoint.topk.monitor_key
normalizer.pkl                       # fit at the start of training, frozen
logs.json.txt                        # one JSON line per step (loss, lr, val_loss, ...)
train.log                            # text log
.hydra/
  config.yaml                        # fully resolved config (your full settings)
  overrides.yaml                     # CLI overrides for this run
  hydra.yaml                         # Hydra runtime state
wandb/                               # only if logging.mode=online
```

### Resume from a crash

```bash
accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_ee2_dice \
    policy.obs_encoder.resize_shape=null \
    training.resume=True \
    hydra.run.dir=data/outputs/2026.05.11/14.32.01_train_diffusion_unet_image_real_ee2_dice
```

---

## 9. Credits

Original work and the diffusion/RDP architectures: Wendi Chen et al. — [arXiv 2512.10946](https://arxiv.org/abs/2512.10946), upstream repo [Chen-Wendi/trainflow](https://github.com/Chen-Wendi/trainflow).

## License

MIT — see [LICENSE](LICENSE) (upstream copyright preserved).
