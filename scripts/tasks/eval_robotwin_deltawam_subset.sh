#!/usr/bin/env bash
set -euo pipefail
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export DELTAWAM_DINOV3_TEMPLATE="$(pwd)/checkpoints/dinov3-vitb16-pretrain-lvd1689m"

: "${CKPT:?Set CKPT to the checkpoint path before running this script.}"

DEFAULT_ARGS=(
  task=sub_robotwin_deltawam_lam_3cam_384_1e-4
  "ckpt=${CKPT}"
  MULTIRUN.num_gpus=1
)

python experiments/robotwin/run_robotwin_manager.py "${DEFAULT_ARGS[@]}" "$@"
