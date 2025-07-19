#!/bin/bash

echo "Starting XenLite Xen3 Training..."

python train.py \
    --batch_size 64 \
    --num_epochs 50 \
    --learning_rate 2e-4 \
    --model_channels 64 \
    --num_workers 4 \
    --save_interval 10 \
    --model_save_path ./models

echo "Training completed!"