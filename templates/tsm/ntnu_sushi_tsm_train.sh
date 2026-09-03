#!/usr/bin/env bash
# Template for mit-han-lab/temporal-shift-module.
# Run this script from temporal-shift-module repo after editing dataset wiring.

set -euo pipefail

python main.py ntnu_sushi RGB \
  --arch resnet50 \
  --num_segments 8 \
  --epochs 50 \
  -b 16 \
  -j 8 \
  --lr 0.01
