# robo_flow

A training-only package for **diffusion / flow / reactive diffusion policy** on robotic manipulation data, stripped down from [Chen-Wendi/ImplicitRDP](https://github.com/Chen-Wendi/ImplicitRDP) (MIT, © 2026 Wendi Chen).

## What this is

A minimal core for learning manipulation policies from **DP-style zarr replay buffers**:

- **Diffusion policies** — UNet (DDPM/DDIM) and Transformer denoisers
- **Reactive Diffusion Policy (RDP)** — the reactive transformer workspace from upstream
- **Vision encoders** — ResNet18/34 (BN→GN) and timm-backed DinoV2 / ViT
- **Hydra + accelerate + W&B** stack — same as upstream

## What was stripped

Everything not needed for training-on-zarr:

- Real-robot deployment (Flexiv control service, ROS-style publishers, `eval_real_robot_flexiv.py`, `vcamera_*`, camera launchers)
- Kinematic-teaching data collection (`record_data.py`, `post_process_data.py`)
- Tactile sensor stack (GelSight / McTac publishers, marker tracking, PCA scripts)
- Force/torque modality (`model/force/rnn.py`, wrench task configs, force-RNN obs encoder branch in DPT)
- Latent diffusion variant (`policy/latent_diffusion_unet_image_policy.py`, VAE, vector_quantize_pytorch)
- Autoencoder pretraining workspace (`workspace/train_at_workspace.py`)
- Real-robot env / env_runner (closed-loop on-robot evaluation)
- Bimanual inter-gripper relative-action computation in the dataset
- Legacy + autoencoder configs (`config/at/`, `config/legacy/`, 22 wrench/kineteach task yamls)

See `git log` for the strip commit and the [upstream README](https://github.com/Chen-Wendi/ImplicitRDP) for the original feature set.

## Layout

```
ImplicitRDP/
├── train.py                  # Hydra entry
├── train_dp.sh               # DDPM/DDIM UNet
├── train_dpt.sh              # Diffusion-Policy Transformer
├── train_rdp.sh              # Reactive Diffusion Policy
├── requirements.txt
└── ImplicitRDP/
    ├── common/               # replay buffer, sampler, action utils, normalizers
    ├── config/               # Hydra configs (workspace + task templates)
    ├── dataset/              # zarr-backed BaseImageDataset
    ├── model/                # diffusion (UNet1D, transformer), vision (resnet, timm)
    ├── policy/               # DiffusionUnet / DiffusionTransformer image policies
    └── workspace/            # accelerate-driven training loops
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install -r requirements.txt
accelerate config
```

## Data

Each dataset is a single zarr directory:

```
<dataset_path>/replay_buffer.zarr/
├── meta/
│   └── episode_ends         # int array, cumulative episode boundaries
└── data/
    ├── <camera_key>          # [N, H, W, 3] uint8
    ├── <state_key>           # [N, D] float32
    └── action                # [N, A] float32
```

`N` = total timesteps across all episodes concatenated. `episode_ends[i]` is the index just past the last frame of episode `i`.

## The `shape_meta` contract

Every component (dataset, encoder, policy) is driven by one block in the task YAML:

```yaml
shape_meta:
  obs:
    <camera_key>:
      shape: [3, H, W]
      type: rgb
    <state_key>:
      shape: [D]
      type: low_dim
  action:
    shape: [A]
```

If your zarr keys match the names in `shape_meta`, no Python edits are needed.

## Train

```bash
# DDPM/DDIM UNet
accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=<your_task> \
    task.dataset_path=/abs/path/to/your_zarr.zarr

# Transformer denoiser
accelerate launch train.py \
    --config-name=train_diffusion_transformer_real_image_workspace \
    task=<your_task>

# Reactive Diffusion Policy
accelerate launch train.py \
    --config-name=train_reactive_diffusion_transformer_real_image_workspace \
    task=<your_task>
```

`task=<your_task>` selects `ImplicitRDP/config/task/<your_task>.yaml`. Use the two stripped templates (`real_flip_image_dp_10fps.yaml`, `real_toggle_image_dp_10fps.yaml`) as a starting point — point `dataset_path` at your zarr and update `shape_meta` keys.

## Credits

Original work and the diffusion/RDP architectures: Wendi Chen et al. — [arXiv 2512.10946](https://arxiv.org/abs/2512.10946), upstream repo [Chen-Wendi/ImplicitRDP](https://github.com/Chen-Wendi/ImplicitRDP). The training framework (replay buffer, sampler, workspaces, policies) is theirs; this fork removes the deployment + tactile/force halves and keeps the training-on-zarr core.

## License

MIT — see [LICENSE](LICENSE) (upstream copyright preserved).
