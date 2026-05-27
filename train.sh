#!/usr/bin/bash

NUM_GPUS=2
# export NCCL_P2P_DISABLE=1  # optional
export EDPOSE_ExLPose_PATH="data/ExLPose"
export pretrain_model_path="ckpts"  # dir to put backbone

PRETRAIN_MODEL_PATH="ckpts/wl.pth"  # pose model weight

DIR="logs/ll"
torchrun --standalone --nproc_per_node=$NUM_GPUS main.py \
    -c config/edpose.cfg.py \
    --options batch_size=8 epochs=100 lr_drop=85 num_body_points=14 backbone='swin_T_224_22k'  \
    --dataset_file="exlpose" \
    --output_dir $DIR \
    --pretrain_model_path=$PRETRAIN_MODEL_PATH \
    --stage "ll"

# DIR="logs/wl"
# python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS main.py \
#     -c config/edpose.cfg.py \
#     --options batch_size=8 epochs=100 lr_drop=85 num_body_points=14 backbone='swin_T_224_22k'  \
#     --dataset_file="exlpose" \
#     --output_dir $DIR \
#     --stage "wl"
