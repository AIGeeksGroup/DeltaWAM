#!/usr/bin/env bash
set -euo pipefail

export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export DELTAWAM_DINOV3_TEMPLATE="$(pwd)/checkpoints/dinov3-vitb16-pretrain-lvd1689m"

DEFAULT_ARGS=(
  task=sub_robotwin_deltakv_3cam_384_1e-4
  ckpt=./checkpoints/deltawam/initial_five_task/deltawam.pt
  EVALUATION.dataset_stats_path=./checkpoints/deltawam/initial_five_task/dataset_stats.json
  'MULTIRUN.gpu_ids=[0,1]'
  MULTIRUN.max_tasks_per_gpu=2
)

python experiments/robotwin/run_robotwin_manager.py "${DEFAULT_ARGS[@]}" "$@"
