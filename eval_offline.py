"""Run a trained diffusion/RDP policy on a held-out (validation) episode.

Reproduces the val split from the run's cfg, replays one episode through
`policy.predict_action` in a sliding receding-horizon loop, then compares the
predicted action trajectory against ground truth.

Usage:
    python eval_offline.py --run-dir data/outputs/2026.05.11/11.31.38_train_diffusion_unet_image_ee2dice_dp_v1
    python eval_offline.py --run-dir <dir> --val-episode 1 --num-inference-steps 16
    python eval_offline.py --run-dir <dir> --ckpt <dir>/checkpoints/latest.ckpt
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import dill
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

# Make the package importable when run from the project root.
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

from trainflow.common.replay_buffer import ReplayBuffer
from trainflow.common.sampler import get_val_mask


def find_best_ckpt(run_dir: pathlib.Path) -> pathlib.Path:
    """Pick the top-k checkpoint with the lowest loss in its filename, else `latest.ckpt`."""
    topk = list((run_dir / 'checkpoints').glob('epoch=*.ckpt'))

    def loss_of(p: pathlib.Path) -> float:
        try:
            return float(p.stem.split('=')[-1])
        except ValueError:
            return float('inf')

    if topk:
        return min(topk, key=loss_of)
    return run_dir / 'checkpoints' / 'latest.ckpt'


def build_obs_window(raw_obs: dict[str, np.ndarray], cfg, t: int, n_obs_steps: int,
                     device: str) -> dict[str, torch.Tensor]:
    """Match the formatting done by RealImageTactileDataset.__getitem__."""
    obs_dict = {}
    for k, attr in cfg.task.shape_meta.obs.items():
        window = raw_obs[k][t:t + n_obs_steps]  # (n_obs_steps, ...)
        if attr.type == 'rgb':
            # uint8 (T, H, W, 3) -> float (T, 3, H, W) in [0, 1]
            window = np.moveaxis(window, -1, -3).astype(np.float32) / 255.0
        else:
            # lowdim: truncate to declared width
            width = attr.shape[0]
            window = window[:, :width].astype(np.float32)
        obs_dict[k] = torch.from_numpy(window[None]).to(device)  # add batch dim
    return obs_dict


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=pathlib.Path, required=True,
                   help='data/outputs/<date>/<time>_<workspace>_<task>/')
    p.add_argument('--ckpt', type=pathlib.Path, default=None,
                   help='Specific checkpoint; defaults to best top-k by filename loss')
    p.add_argument('--val-episode', type=int, default=0,
                   help='0-indexed position into the validation split')
    p.add_argument('--out-dir', type=pathlib.Path, default=None,
                   help='Where to write the plot; defaults to <run-dir>/eval/')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-inference-steps', type=int, default=None,
                   help='Override DDIM denoise steps (default uses policy.num_inference_steps from cfg)')
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    ckpt_path = (args.ckpt or find_best_ckpt(run_dir)).resolve()
    out_dir = (args.out_dir or run_dir / 'eval').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'run_dir : {run_dir}')
    print(f'ckpt    : {ckpt_path}')
    print(f'out_dir : {out_dir}')

    # 1. Load cfg + build workspace skeleton.
    cfg = OmegaConf.load(run_dir / '.hydra' / 'config.yaml')
    WorkspaceCls = hydra.utils.get_class(cfg._target_)
    workspace = WorkspaceCls(cfg)

    # 2. Load checkpoint payload (skip optimizer state — we're not training).
    payload = torch.load(
        open(ckpt_path, 'rb'), pickle_module=dill,
        map_location='cpu', weights_only=False,
    )
    workspace.load_payload(payload, exclude_keys=['optimizer'])

    # 3. Pick EMA weights for inference if available.
    policy = workspace.model
    if cfg.training.use_ema and workspace.ema_model is not None:
        policy = workspace.ema_model
        print('using EMA weights')
    policy = policy.to(args.device).eval()
    if args.num_inference_steps is not None:
        policy.num_inference_steps = args.num_inference_steps
        print(f'num_inference_steps overridden -> {args.num_inference_steps}')

    # 4. Re-open the zarr the model was trained on; reproduce the val mask.
    project_root = pathlib.Path(__file__).parent.resolve()
    zarr_path = pathlib.Path(cfg.task.dataset.dataset_path) / 'replay_buffer.zarr'
    if not zarr_path.is_absolute():
        zarr_path = project_root / zarr_path
    assert zarr_path.exists(), f'zarr not found: {zarr_path}'

    obs_keys = list(cfg.task.shape_meta.obs.keys())
    rb = ReplayBuffer.copy_from_path(str(zarr_path), keys=obs_keys + ['action'])
    val_mask = get_val_mask(
        n_episodes=rb.n_episodes,
        val_ratio=cfg.task.dataset.val_ratio,
        seed=cfg.task.dataset.seed,
    )
    val_ep_indices = np.flatnonzero(val_mask)
    assert args.val_episode < len(val_ep_indices), (
        f'val_episode {args.val_episode} out of range (n_val={len(val_ep_indices)})')

    ep = int(val_ep_indices[args.val_episode])
    ep_ends = np.asarray(rb.episode_ends)
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    start, end = int(ep_starts[ep]), int(ep_ends[ep])
    T = end - start
    print(f'episode {ep}  (val idx {args.val_episode}, length {T})')

    # 5. Pull raw arrays for the episode.
    gt_action = np.asarray(rb['action'])[start:end]  # (T, A)
    raw_obs = {k: np.asarray(rb[k])[start:end] for k in obs_keys}

    # 6. Sliding-window inference, stride = n_action_steps (receding horizon).
    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    pred_action = np.full_like(gt_action, np.nan)

    t = 0
    while t + n_obs_steps <= T:
        obs_dict = build_obs_window(raw_obs, cfg, t, n_obs_steps, args.device)
        with torch.no_grad():
            result = policy.predict_action(obs_dict)
        chunk = result['action'][0].cpu().numpy()  # (n_action_steps, A)

        out_start = t + n_obs_steps - 1
        out_end = min(out_start + n_action_steps, T)
        usable = out_end - out_start
        pred_action[out_start:out_end] = chunk[:usable]
        t += n_action_steps

    # 7. Per-dim MSE on covered frames.
    valid = ~np.isnan(pred_action).any(axis=1)
    mse = ((pred_action[valid] - gt_action[valid]) ** 2).mean(axis=0)
    overall = float(mse.mean())
    print('\nper-dim action MSE:')
    for d, m in enumerate(mse):
        print(f'  dim {d}: {m:.5f}')
    print(f'overall MSE: {overall:.5f}  (over {valid.sum()}/{T} frames)')

    # 8. Plot predicted vs ground truth per dim.
    n_dim = gt_action.shape[1]
    fig, axes = plt.subplots(n_dim, 1, figsize=(11, 1.4 * n_dim), sharex=True, squeeze=False)
    axes = axes.ravel()
    for d in range(n_dim):
        axes[d].plot(gt_action[:, d], label='ground truth', linewidth=1.0)
        axes[d].plot(pred_action[:, d], label='predicted', linewidth=1.0, linestyle='--')
        axes[d].set_title(f'action[{d}]  MSE={mse[d]:.5f}', fontsize=9)
        axes[d].grid(alpha=0.3)
        axes[d].legend(fontsize=7, loc='upper right')
    axes[-1].set_xlabel('frame within episode')
    fig.suptitle(f'{run_dir.name}\nval episode {ep} (val idx {args.val_episode}) — overall MSE = {overall:.5f}')
    fig.tight_layout()
    out_path = out_dir / f'pred_vs_gt_val_ep{args.val_episode}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
