#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _COMMON_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  _COMMON_ENV_DIR="$(pwd)"
fi

: "${PROJECT_ROOT:=$(cd "${_COMMON_ENV_DIR}/../.." && pwd)}"
: "${RUNS_DIR:=${PROJECT_ROOT}/runs}"

mkdir -p "${RUNS_DIR}"
cd "${PROJECT_ROOT}"

if [[ "${SKIP_CONDA_ACTIVATE:-0}" == "1" ]]; then
  return 0 2>/dev/null || exit 0
fi

: "${CONDA_ENV_NAME:=deltawam}"
: "${CONDA_SH:=}"

if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV_NAME}"
  return 0 2>/dev/null || exit 0
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
  return 0 2>/dev/null || exit 0
fi

echo "[warn] conda not found; continue without activation." >&2
