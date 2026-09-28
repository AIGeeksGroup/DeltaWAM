#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ARGS=(task=sub_robotwin_deltakv_3cam_384_1e-4)
python scripts/precompute_text_embeds.py "${DEFAULT_ARGS[@]}" "$@"
