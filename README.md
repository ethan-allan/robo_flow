# robo_flow

A training-only package for **diffusion / flow / reactive diffusion policy** on robotic manipulation data, derived from [Chen-Wendi/ImplicitRDP](https://github.com/Chen-Wendi/ImplicitRDP) (MIT, © 2026 Wendi Chen) with the deployment, tactile, and force halves removed.

You bring a zarr replay buffer of `(image, state, action)` trajectories; this package trains a policy on it via Hydra + accelerate + W&B.

Three policy variants are supported:

| Workspace config                                          | Policy                  | Launcher       |
| --------------------------------------------------------- | ----------------------- | -------------- |
| `train_diffusion_unet_real_image_workspace`               | DDPM/DDIM UNet          | `train_dp.sh`  |
| `train_diffusion_transformer_real_image_workspace`        | Transformer denoiser    | `train_dpt.sh` |
| `train_reactive_diffusion_transformer_real_image_workspace` | Reactive Diffusion Policy | `train_rdp.sh` |

---

## 1. Install

```bash
# Python 3.10
python3.10 -m venv .venv          # or: uv venv
source .venv/bin/activate         # if you used uv, run binaries from .venv/bin directly

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install -r requirements.txt

accelerate config                 # one-time: pick GPU/precision
wandb login                       # optional, only if logging.mode=online
```

---

## 2. Prepare your data

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

### Convert ee2dice episode dumps

If your data is in the ee2dice per-episode `.npy` layout (`eef_state.npy`, `eef_action.npy`, `rgb_hand.npy`, ...), use the bundled converter:

```bash
python convert_ee2dice_to_zarr.py \
    --src /path/to/ee2dice \
    --dst data/ee2dice_zarr
```

It writes `data/ee2dice_zarr/replay_buffer.zarr` with:

- `data/rgb_hand`   — `(N, 256, 256, 3)` uint8
- `data/eef_state`  — `(N, 9)` float32, xyz + rot6d (gripper dropped)
- `data/action`     — `(N, 9)` float32, same layout
- `meta/episode_ends` — int64

### Bringing your own data

Write whatever script makes sense for your source format, but match the zarr layout above. Rotations should be encoded as 6D (Zhou et al. 2019); see `convert_ee2dice_to_zarr.py` for a reference implementation of axis-angle → rot6d.

---

## 3. Wire up a task config

Each task is one YAML in `ImplicitRDP/config/task/`. Start by copying a template:

```bash
cp ImplicitRDP/config/task/real_flip_image_dp_10fps.yaml \
   ImplicitRDP/config/task/ee2dice_dp_10fps.yaml
```

Edit three things:

```yaml
name: ee2dice_dp_10fps
image_shape: [3, 256, 256]               # C, H, W of your camera
dataset_path: data/ee2dice_zarr

shape_meta: &shape_meta
  obs:
    rgb_hand:                            # must match the zarr key
      shape: ${task.image_shape}
      type: rgb
    eef_state:                           # must match the zarr key
      shape: [9]
      type: low_dim
  action:
    shape: [9]                           # must match data/action width
```

The keys under `shape_meta.obs` must exactly match the keys in `data/` in your zarr — the dataset wires them up by name.

---

## 4. Train

### Diffusion-Policy UNet (DDPM / DDIM)

```bash
accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=ee2dice_dp_10fps \
    task.dataset_path=data/ee2dice_zarr \
    logging.mode=online
```

### Diffusion-Policy Transformer

```bash
accelerate launch train.py \
    --config-name=train_diffusion_transformer_real_image_workspace \
    task=ee2dice_dp_10fps \
    policy.noise_scheduler.prediction_type=v_prediction \
    logging.mode=online
```

### Reactive Diffusion Policy

```bash
accelerate launch train.py \
    --config-name=train_reactive_diffusion_transformer_real_image_workspace \
    task=ee2dice_dp_10fps
```

### Single-GPU, no W&B, debug run

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=ee2dice_dp_10fps \
    training.debug=True \
    logging.mode=disabled
```

### Common overrides

Anything in the workspace YAML is overridable on the CLI:

```bash
accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=ee2dice_dp_10fps \
    horizon=16 \
    n_obs_steps=2 \
    dataloader.batch_size=32 \
    training.num_epochs=300 \
    optimizer.lr=5e-5
```

To swap the vision encoder to DinoV2, comment out the ResNet `obs_encoder` block in `ImplicitRDP/config/train_diffusion_unet_real_image_workspace.yaml` and uncomment the timm block beneath it.

---

## 5. Outputs

Each run writes to `data/outputs/<date>/<time>_<name>_<task_name>/`:

```
checkpoints/
  latest.ckpt
  epoch=0090-train_loss=0.041.ckpt   # top-k by training.checkpoint.topk
.hydra/                              # resolved config snapshot
wandb/                               # if logging.mode=online
```

Resume from a crash:

```bash
accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=ee2dice_dp_10fps \
    training.resume=True \
    hydra.run.dir=data/outputs/2026.05.11/14.32.01_train_diffusion_unet_image_ee2dice_dp_10fps
```

---

## Layout

```
ImplicitRDP/
├── train.py                      # Hydra entry
├── convert_ee2dice_to_zarr.py    # ee2dice → replay-buffer zarr
├── train_dp.sh / train_dpt.sh / train_rdp.sh   # example launchers
├── requirements.txt
└── ImplicitRDP/
    ├── common/      # replay buffer, sampler, action utils, normalizers
    ├── config/      # workspace YAMLs + task/ templates
    ├── dataset/     # zarr-backed BaseImageDataset
    ├── model/       # diffusion (UNet1D, transformer), vision (resnet, timm)
    ├── policy/      # DiffusionUnet / DiffusionTransformer image policies
    └── workspace/   # accelerate-driven training loops
```

---

## Credits

Original work and the diffusion/RDP architectures: Wendi Chen et al. — [arXiv 2512.10946](https://arxiv.org/abs/2512.10946), upstream repo [Chen-Wendi/ImplicitRDP](https://github.com/Chen-Wendi/ImplicitRDP).

## License

MIT — see [LICENSE](LICENSE) (upstream copyright preserved).
