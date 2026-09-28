#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/cluster/run_task.sh <task_script> [task_args...]" >&2
  exit 1
fi

TASK_SCRIPT_INPUT="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

if [[ "${TASK_SCRIPT_INPUT}" = /* ]]; then
  TASK_SCRIPT="${TASK_SCRIPT_INPUT}"
else
  TASK_SCRIPT="${PROJECT_ROOT}/${TASK_SCRIPT_INPUT}"
fi

if [[ ! -f "${TASK_SCRIPT}" ]]; then
  echo "Task script not found: ${TASK_SCRIPT}" >&2
  exit 1
fi

echo "Job started: $(date)"
echo "Node: $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU:"
  nvidia-smi --query-gpu=name --format=csv,noheader || true
fi
echo "Python: $(command -v python || command -v python3 || echo 'not found')"
echo "Project root: ${PROJECT_ROOT}"
echo "Task script: ${TASK_SCRIPT}"
echo ""

bash "${TASK_SCRIPT}" "$@"

echo ""
echo "Job finished: $(date)"
