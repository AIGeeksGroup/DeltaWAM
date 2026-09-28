#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

TASK="${TASK:-sub_robotwin_uncond_3cam_384_1e-4}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/flops_results}"

FASTWAM_LORA_RANK="${FASTWAM_LORA_RANK:-128}"
FASTWAM_LORA_ALPHA="${FASTWAM_LORA_ALPHA:-256}"

DELTAWAM_LORA_RANK="${DELTAWAM_LORA_RANK:-64}"
DELTAWAM_LORA_ALPHA="${DELTAWAM_LORA_ALPHA:-128}"

JOINT_LORA_RANK="${JOINT_LORA_RANK:-64}"
JOINT_LORA_ALPHA="${JOINT_LORA_ALPHA:-128}"
JOINT_ADAPTER_LAYERS="${JOINT_ADAPTER_LAYERS:-last_6}"
JOINT_ADAPTER_BOTTLENECK_RATIO="${JOINT_ADAPTER_BOTTLENECK_RATIO:-16}"
PROFILE_JOINT_DELTATOK="${PROFILE_JOINT_DELTATOK:-false}"
DELTATOK_PRETRAINED_PATH="${DELTATOK_PRETRAINED_PATH:-checkpoints/deltatok-kinetics/pytorch_model.bin}"
DELTATOK_BACKBONE_NAME="${DELTATOK_BACKBONE_NAME:-${DELTAWAM_DINOV3_TEMPLATE:-facebook/dinov3-vitb16-pretrain-lvd1689m}}"
export DELTAWAM_DINOV3_TEMPLATE="${DELTAWAM_DINOV3_TEMPLATE:-${DELTATOK_BACKBONE_NAME}}"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
  --base-overrides
  "task=${TASK}"
  "num_workers=0"
  --batch-size "${BATCH_SIZE}"
  --synthetic
  --mixed-precision "${MIXED_PRECISION}"
  --device "${DEVICE}"
  --train-profile backward
  --no-profile-infer
)

echo "[1/4] Profiling original FastWAM train_forward_backward FLOPs..."
python scripts/profile_flops.py \
  --models fastwam \
  "${COMMON_ARGS[@]}" \
  --model-overrides \
  "model.wan_lora_config.enabled=false" \
  --output "${OUTPUT_DIR}/flops_fastwam_original_train_backward.json"

echo "[2/4] Profiling FastWAM-LoRA train_forward_backward FLOPs..."
python scripts/profile_flops.py \
  --models fastwam \
  "${COMMON_ARGS[@]}" \
  --model-overrides \
  "model.wan_lora_config.enabled=true" \
  "model.wan_lora_config.rank=${FASTWAM_LORA_RANK}" \
  "model.wan_lora_config.alpha=${FASTWAM_LORA_ALPHA}" \
  --output "${OUTPUT_DIR}/flops_fastwam_lora_r${FASTWAM_LORA_RANK}_a${FASTWAM_LORA_ALPHA}_train_backward.json"

echo "[3/4] Profiling DeltaWAM train_forward_backward FLOPs..."
python scripts/profile_flops.py \
  --models deltawam \
  "${COMMON_ARGS[@]}" \
  --model-overrides \
  "model.wan_lora_config.enabled=true" \
  "model.wan_lora_config.rank=${DELTAWAM_LORA_RANK}" \
  "model.wan_lora_config.alpha=${DELTAWAM_LORA_ALPHA}" \
  --output "${OUTPUT_DIR}/flops_deltawam_lora_r${DELTAWAM_LORA_RANK}_a${DELTAWAM_LORA_ALPHA}_train_backward.json"

echo "[4/4] Profiling DeltaWAM-Joint train_forward_backward FLOPs..."
python scripts/profile_flops.py \
  --models deltawam_joint \
  "${COMMON_ARGS[@]}" \
  --model-overrides \
  "model.wan_lora_config.enabled=true" \
  "model.wan_lora_config.rank=${JOINT_LORA_RANK}" \
  "model.wan_lora_config.alpha=${JOINT_LORA_ALPHA}" \
  "model.action_delta_dit_config.task_adapter.enabled=true" \
  "model.action_delta_dit_config.task_adapter.layers=${JOINT_ADAPTER_LAYERS}" \
  "model.action_delta_dit_config.task_adapter.bottleneck_ratio=${JOINT_ADAPTER_BOTTLENECK_RATIO}" \
  --output "${OUTPUT_DIR}/flops_deltawam_joint_lora_r${JOINT_LORA_RANK}_a${JOINT_LORA_ALPHA}_${JOINT_ADAPTER_LAYERS}_br${JOINT_ADAPTER_BOTTLENECK_RATIO}_train_backward.json"

if [[ "${PROFILE_JOINT_DELTATOK}" == "true" ]]; then
  echo "[extra] Profiling DeltaWAM-Joint + DeltaTok encoder train_forward_backward FLOPs..."
  python scripts/profile_flops.py \
    --models deltawam_joint \
    "${COMMON_ARGS[@]}" \
    --model-overrides \
    "model.wan_lora_config.enabled=true" \
    "model.wan_lora_config.rank=${JOINT_LORA_RANK}" \
    "model.wan_lora_config.alpha=${JOINT_LORA_ALPHA}" \
    "model.delta_tokenizer_pretrained_path=${DELTATOK_PRETRAINED_PATH}" \
    "model.delta_tokenizer_config.impl=deltatok" \
    "+model.delta_tokenizer_config.num_hidden_layers=12" \
    "+model.delta_tokenizer_config.use_delta=true" \
    "+model.delta_tokenizer_config.use_decoder_head=null" \
    "+model.delta_tokenizer_config.use_decoder_xy_embed=null" \
    "+model.delta_tokenizer_config.layer_scale_init=1e-5" \
    "+model.delta_tokenizer_config.use_qk_norm=true" \
    "+model.delta_tokenizer_config.use_gated_attn=true" \
    "+model.delta_tokenizer_config.use_swiglu=true" \
    "+model.delta_tokenizer_config.use_rope_aug=true" \
    "+model.delta_tokenizer_config.backbone._target_=deltawam.models.wan22.delta_tokenizer.DINOv3BackboneWithRGBHead" \
    "+model.delta_tokenizer_config.backbone.backbone_name=${DELTATOK_BACKBONE_NAME}" \
    "+model.delta_tokenizer_config.backbone.rgb_head_path=null" \
    "+model.delta_tokenizer_config.backbone.strict_rgb_head=false" \
    "+model.delta_tokenizer_config.backbone.rgb_head_hidden_size=1152" \
    "+model.delta_tokenizer_config.backbone.rgb_head_depth=28" \
    "+model.delta_tokenizer_config.backbone.rgb_head_num_heads=16" \
    "+model.delta_tokenizer_config.backbone.rgb_head_mlp_size=4096" \
    "+model.delta_tokenizer_config.backbone.rgb_head_img_size=512" \
    "model.action_delta_dit_config.delta_token_dim=768" \
    "model.action_delta_dit_config.task_adapter.enabled=true" \
    "model.action_delta_dit_config.task_adapter.layers=${JOINT_ADAPTER_LAYERS}" \
    "model.action_delta_dit_config.task_adapter.bottleneck_ratio=${JOINT_ADAPTER_BOTTLENECK_RATIO}" \
    --output "${OUTPUT_DIR}/flops_deltawam_joint_deltatok_lora_r${JOINT_LORA_RANK}_a${JOINT_LORA_ALPHA}_${JOINT_ADAPTER_LAYERS}_br${JOINT_ADAPTER_BOTTLENECK_RATIO}_train_backward.json"
fi

echo "Done. CSV outputs:"
echo "  ${OUTPUT_DIR}/flops_fastwam_original_train_backward.csv"
echo "  ${OUTPUT_DIR}/flops_fastwam_lora_r${FASTWAM_LORA_RANK}_a${FASTWAM_LORA_ALPHA}_train_backward.csv"
echo "  ${OUTPUT_DIR}/flops_deltawam_lora_r${DELTAWAM_LORA_RANK}_a${DELTAWAM_LORA_ALPHA}_train_backward.csv"
echo "  ${OUTPUT_DIR}/flops_deltawam_joint_lora_r${JOINT_LORA_RANK}_a${JOINT_LORA_ALPHA}_${JOINT_ADAPTER_LAYERS}_br${JOINT_ADAPTER_BOTTLENECK_RATIO}_train_backward.csv"
if [[ "${PROFILE_JOINT_DELTATOK}" == "true" ]]; then
  echo "  ${OUTPUT_DIR}/flops_deltawam_joint_deltatok_lora_r${JOINT_LORA_RANK}_a${JOINT_LORA_ALPHA}_${JOINT_ADAPTER_LAYERS}_br${JOINT_ADAPTER_BOTTLENECK_RATIO}_train_backward.csv"
fi
