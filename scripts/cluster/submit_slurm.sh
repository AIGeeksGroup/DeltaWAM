#!/usr/bin/env bash
# Single Slurm entrypoint.
# Edit the config block below before each submission.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUNS_DIR="${RUNS_DIR:-${PROJECT_ROOT}/runs}"
mkdir -p "${RUNS_DIR}"

############################
# Config (edit these)
############################
JOB_NAME="precompute_text_embeds"
TASK_SCRIPT="scripts/tasks/precompute_text_embeds.sh"
TASK_ARGS=(
  # Example overrides:
  # "task=robotwin_deltawam_3cam_384_1e-4"
)

SLURM_PARTITION="day"
SLURM_TIME="12:00:00"
SLURM_GPUS="1"
SLURM_CPUS="4"
SLURM_MEM="32G"
SLURM_EXTRA_ARGS=(
  # Example extra args:
  # "--constraint=a100"
)
############################

SBATCH_CMD=(
  sbatch
  "--job-name=${JOB_NAME}"
  "--output=${RUNS_DIR}/${JOB_NAME}.log"
  "--error=${RUNS_DIR}/${JOB_NAME}.err"
  "--partition=${SLURM_PARTITION}"
  "--time=${SLURM_TIME}"
  "--cpus-per-task=${SLURM_CPUS}"
  "--mem=${SLURM_MEM}"
)

if [[ "${SLURM_GPUS}" != "0" ]]; then
  SBATCH_CMD+=("--gres=gpu:${SLURM_GPUS}")
fi

if [[ ${#SLURM_EXTRA_ARGS[@]} -gt 0 ]]; then
  SBATCH_CMD+=("${SLURM_EXTRA_ARGS[@]}")
fi

RUN_TASK_CMD=(
  bash
  "${PROJECT_ROOT}/scripts/cluster/run_task.sh"
  "${TASK_SCRIPT}"
  "${TASK_ARGS[@]}"
)
WRAP_STR="$(printf '%q ' "${RUN_TASK_CMD[@]}")"
WRAP_STR="${WRAP_STR% }"

"${SBATCH_CMD[@]}" --wrap "${WRAP_STR}"
