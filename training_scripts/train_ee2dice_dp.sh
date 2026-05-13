#!/usr/bin/env bash
# Train the Diffusion-Policy UNet on the ee2dice dataset.
# All knobs live in the CONFIG block below — edit there, then `bash scripts/train_ee2dice_dp.sh`.
set -euo pipefail

# --- CONFIG ------------------------------------------------------------------
GPU_ID=0
RUN_NAME=ee2dice_dp_v1

# data + horizon
TASK=real_ee2_dice                     # name of config/task/<TASK>.yaml
HORIZON=16
N_OBS_STEPS=2

# optimization
NUM_EPOCHS=200                         # 33 episodes -> overfits fast at 600
BATCH_SIZE=64
LR=1.0e-4

# encoder
# resize_shape now comes from task yaml (image_resize_shape).
SHARE_RGB_MODEL=False

# diffusion
PREDICTION_TYPE=epsilon                # or v_prediction

# logging / checkpoints
LOGGING_MODE=online                    # online | offline | disabled
TOPK_K=3
CHECKPOINT_EVERY=20
# -----------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=${TASK} \
    task_name=${RUN_NAME} \
    horizon=${HORIZON} \
    n_obs_steps=${N_OBS_STEPS} \
    training.num_epochs=${NUM_EPOCHS} \
    dataloader.batch_size=${BATCH_SIZE} \
    val_dataloader.batch_size=${BATCH_SIZE} \
    optimizer.lr=${LR} \
    policy.obs_encoder.share_rgb_model=${SHARE_RGB_MODEL} \
    policy.noise_scheduler.prediction_type=${PREDICTION_TYPE} \
    checkpoint.topk.k=${TOPK_K} \
    training.checkpoint_every=${CHECKPOINT_EVERY} \
    logging.mode=${LOGGING_MODE}
