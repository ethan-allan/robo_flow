#!/usr/bin/env bash
# Train the Diffusion-Policy UNet on the peg hole dataset.
# All knobs live in the CONFIG block below — edit there, then `bash scripts/train_peghole.sh`.
set -euo pipefail

# --- CONFIG ------------------------------------------------------------------
GPU_ID=0
RUN_NAME=peg_hole_vrr_01

# data + horizon
TASK=peg_insertion_vrr_5fps                     # name of config/task/<TASK>.yaml
HORIZON=8
N_OBS_STEPS=2

# optimization
NUM_EPOCHS=500                         # okay 33 episodes -> overfits fast at 600
BATCH_SIZE=64
LR=1.0e-4

# encoder
# Transformer obs encoder has no resize_shape field — image resize lives in the
# task yaml's image_resize_shape (consumed by the unet encoder path only).
SHARE_RGB_MODEL=False

# diffusion
PREDICTION_TYPE=epsilon                # or v_prediction

# logging / checkpoints
LOGGING_MODE=online                    # online | offline | disabled
TOPK_K=3
CHECKPOINT_EVERY=20
# -----------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_diffusion_transformer_real_image_workspace \
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
