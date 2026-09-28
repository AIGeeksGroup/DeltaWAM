#!/usr/bin/env bash
# Single PBS entrypoint.
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

PBS_QUEUE="day"
PBS_WALLTIME="12:00:00"
PBS_SELECT="select=1:ncpus=4:ngpus=1:mem=32gb"
PBS_EXTRA_ARGS=(
  # Example extra args:
  # "-l"
  # "place=free"
)
############################

RUN_TASK_CMD=(
  bash
  "${PROJECT_ROOT}/scripts/cluster/run_task.sh"
  "${TASK_SCRIPT}"
  "${TASK_ARGS[@]}"
)
RUN_TASK_STR="$(printf '%q ' "${RUN_TASK_CMD[@]}")"
RUN_TASK_STR="${RUN_TASK_STR% }"

TMP_JOB_SCRIPT="$(mktemp)"
trap 'rm -f "${TMP_JOB_SCRIPT}"' EXIT

cat > "${TMP_JOB_SCRIPT}" <<PBS
#!/usr/bin/env bash
#PBS -N ${JOB_NAME}
#PBS -q ${PBS_QUEUE}
#PBS -l walltime=${PBS_WALLTIME}
#PBS -l ${PBS_SELECT}
#PBS -o ${RUNS_DIR}/${JOB_NAME}.pbs.log
#PBS -e ${RUNS_DIR}/${JOB_NAME}.pbs.err
set -euo pipefail
${RUN_TASK_STR}
PBS

QSUB_CMD=(qsub)
if [[ ${#PBS_EXTRA_ARGS[@]} -gt 0 ]]; then
  QSUB_CMD+=("${PBS_EXTRA_ARGS[@]}")
fi
QSUB_CMD+=("${TMP_JOB_SCRIPT}")

"${QSUB_CMD[@]}"
