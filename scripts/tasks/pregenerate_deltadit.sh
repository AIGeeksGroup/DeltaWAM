#!/usr/bin/env bash
set -euo pipefail

python scripts/preprocess_dit_backbone.py \
  --model-config configs/model/pregenerate_deltawam_dit.yaml \
  --output checkpoints/DeltaDiT_lam_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16 \
  --target delta \
  "$@"
