#!/usr/bin/bash

NUM_GPUS=1
export EDPOSE_ExLPose_PATH="data/ExLPose"
export pretrain_model_path="ckpts"

PRETRAIN_MODEL_PATH="ckpts/final.pth"

for stage in "wl" "ll" "lln" "llh" "lle" "a7m3" "ricoh3"; do
    python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS  main.py \
    -c config/edpose.cfg.py \
    --options batch_size=32 num_body_points=14 backbone='swin_T_224_22k' \
    --dataset_file="exlpose" \
    --output_dir "output" \
    --pretrain_model_path=$PRETRAIN_MODEL_PATH \
    --eval \
    --stage $stage
done

# python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS  main.py \
#     -c config/edpose.cfg.py \
#     --options batch_size=32 num_body_points=14 backbone='swin_T_224_22k' \
#     --dataset_file="exlpose" \
#     --output_dir "output" \
#     --pretrain_model_path=$PRETRAIN_MODEL_PATH \
#     --eval \
#     --stage "ll"