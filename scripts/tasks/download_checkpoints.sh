#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com

hf download yuanty/fastwam \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release

hf download Amazon-FAR/rgb-head-imagenet \
  pytorch_model.bin \
  --local-dir ./checkpoints/rgb-head-imagenet

hf download Amazon-FAR/deltatok-kinetics \
  pytorch_model.bin \
  --local-dir ./checkpoints/deltatok-kinetics

hf download nvidia/DreamDojo \
  LAM_400k.ckpt \
  --local-dir ./checkpoints/DreamDojo/
