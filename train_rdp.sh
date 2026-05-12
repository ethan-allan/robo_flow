#!/bin/bash

GPU_ID=0

TASK="flip"
DATASET_VERSION="v2"
ACTION_TYPE="right_arm_6DOF_wrench"
DATASET_PATH="/home/wendi/Desktop/trainflow/data/${TASK}_${DATASET_VERSION}_zarr"
TRAINING_DEBUG=False
LOGGING_MODE="online"

IDENTIFIER=$(date +%m%d%H%M%S)
SEARCH_PATH="./data/outputs"

# Stage 0: Post-process data
echo "Stage 0: post-processing data..."
python post_process_data.py --tag ${TASK}_${DATASET_VERSION} --action_type ${ACTION_TYPE} --skip_static_frames

# Stage 1: Train Asymmetric Tokenizer
echo "Stage 1: training Asymmetric Tokenizer..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --config-name=train_at_workspace \
    task=real_${TASK}_image_wrench_at_10fps \
    task.dataset_path=${DATASET_PATH} \
    task.name=real_${TASK}_image_wrench_at_10fps_${IDENTIFIER} \
    at=at_${TASK} \
    training.debug=${TRAINING_DEBUG} \
    logging.mode=${LOGGING_MODE}

# find the latest checkpoint
echo ""
echo "Searching for the latest AT checkpoint..."
AT_LOAD_DIR=$(find "${SEARCH_PATH}" -maxdepth 2 -path "*${IDENTIFIER}*" -type d)/checkpoints/latest.ckpt

if [ ! -f "${AT_LOAD_DIR}" ]; then
    echo "Error: VAE checkpoint not found at ${AT_LOAD_DIR}"
    exit 1
fi

# Stage 2: Train Latent Diffusion Policy
echo ""
echo "Stage 2: training Latent Diffusion Policy..."
CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch train.py \
    --config-name=train_latent_diffusion_unet_real_image_workspace \
    task=real_${TASK}_image_wrench_ldp_10fps \
    task.dataset_path=${DATASET_PATH} \
    task.name=real_${TASK}_image_wrench_ldp_10fps_${IDENTIFIER} \
    at=at_${TASK} \
    at_load_dir=${AT_LOAD_DIR} \
    training.debug=${TRAINING_DEBUG} \
    logging.mode=${LOGGING_MODE}