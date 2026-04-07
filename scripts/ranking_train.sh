#!/bin/bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  ranking/main.py \
  --model DCNv1 \
  --data_dir data \
  --batch_size 20000 \
  --num_epochs 20
