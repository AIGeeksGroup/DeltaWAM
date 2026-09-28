#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export DELTAWAM_DINOV3_TEMPLATE=checkpoints/dinov3-vitb16-pretrain-lvd1689m
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

: "${TRAIN_NPROC_PER_NODE:=1}"
DEFAULT_ARGS=(task=sub_robotwin_deltawam_joint_light_3cam_384_1e-4)

bash scripts/train_zero1.sh "${TRAIN_NPROC_PER_NODE}" "${DEFAULT_ARGS[@]}" "$@"
