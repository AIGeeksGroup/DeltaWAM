#!/usr/bin/env bash
set -euo pipefail

: "${RESUME_CKPT:?Set RESUME_CKPT to the checkpoint path before running this script.}"

DEFAULT_ARGS=(
  task=sub_robotwin_deltawam_lam_3cam_384_1e-4
  "resume=${RESUME_CKPT}"
  output_dir=./runs/eval_once/robotwin_deltawam_lam
  +eval_save_all_video_modes=true
)

python scripts/eval_train_once.py "${DEFAULT_ARGS[@]}" "$@"
