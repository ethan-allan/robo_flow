#!/bin/bash

# Stage 0: Post-process data
echo "Stage 0: post-processing data..."
python post_process_data.py --tag flip_v2 --action_type right_arm_6DOF --skip_static_frames

# Stage 1: Train Policy
echo "Stage 1: training Policy..."
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=real_flip_image_dp_10fps \
    task.dataset_path=/home/wendi/Desktop/trainflow/data/flip_v2_zarr \
    task.name=real_flip_image_dp_10fps_v2 \
    logging.mode=online