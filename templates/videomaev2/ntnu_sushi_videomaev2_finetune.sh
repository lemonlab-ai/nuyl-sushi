#!/usr/bin/env bash
# Template for OpenGVLab/VideoMAEv2 fine-tuning.
# Run this script from VideoMAEv2 repo after editing paths.

set -euo pipefail

python run_class_finetuning.py \
  --model vit_small_patch16_224 \
  --data_path /ABS/PATH/TO/TASK_A_DATA \
  --nb_classes 32 \
  --epochs 30 \
  --batch_size 8 \
  --num_frames 16 \
  --sampling_rate 4 \
  --output_dir /ABS/PATH/TO/OUTPUT/VIDEOMAEV2 \
  --finetune /ABS/PATH/TO/PRETRAINED_CHECKPOINT
