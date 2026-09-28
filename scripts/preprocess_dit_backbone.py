import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from deltawam.models.wan22.action_dit import ActionDiT
from deltawam.models.wan22.delta_dit import DeltaDiT
from deltawam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components


@dataclass(frozen=True)
class TargetSpec:
    name: str
    cfg_key: str
    model_class: type
    int_fields: tuple[str, ...]
    float_fields: tuple[str, ...]
    shared_with_video_keys: tuple[str, ...]
    output_name: str
    extra_unresolved_defaults: dict[str, Any]


TARGET_SPECS: dict[str, TargetSpec] = {
    "action": TargetSpec(
        name="action",
        cfg_key="action_dit_config",
        model_class=ActionDiT,
        int_fields=(
            "hidden_dim",
            "action_dim",
            "ffn_dim",
            "num_layers",
            "num_heads",
            "attn_head_dim",
            "text_dim",
            "freq_dim",
        ),
        float_fields=("eps",),
        shared_with_video_keys=("num_heads", "attn_head_dim", "num_layers", "text_dim", "freq_dim"),
        output_name="ActionDiT",
        extra_unresolved_defaults={"action_dim": 7},
    ),
    "delta": TargetSpec(
        name="delta",
        cfg_key="delta_dit_config",
        model_class=DeltaDiT,
        int_fields=(
            "delta_token_dim",
            "hidden_dim",
            "ffn_dim",
            "num_layers",
            "num_heads",
            "attn_head_dim",
            "text_dim",
            "freq_dim",
        ),
        float_fields=("eps",),
        shared_with_video_keys=("num_heads", "attn_head_dim", "num_layers", "text_dim", "freq_dim"),
        output_name="DeltaDiT",
        extra_unresolved_defaults={},
    ),
}


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}. Expected one of: float32, float16, bfloat16.")


def _parse_bool(name: str) -> bool:
    value = str(name).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {name}")


def _is_unresolved_interpolation(value: Any) -> bool:
    return isinstance(value, str) and "${" in value and "}" in value


def _resolve_from_video_cfg(value: Any, video_cfg: dict[str, Any]) -> Any:
    if not _is_unresolved_interpolation(value):
        return value
    text = str(value).strip()
    if not (text.startswith("${") and text.endswith("}")):
        return value
    expr = text[2:-1]
    if not expr.startswith("video_dit_config."):
        return value
    key = expr.split(".", 1)[1]
    if key not in video_cfg:
        return value
    resolved = video_cfg[key]
    return value if _is_unresolved_interpolation(resolved) else resolved


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_tensor_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(
            f"Resize produced wrong shape for tensor. src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def _require_int_config(cfg: dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return int(value)


def _require_float_config(cfg: dict[str, Any], key: str) -> float:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return float(value)


def _load_model_configs(path: Path, spec: TargetSpec) -> tuple[dict[str, Any], dict[str, Any], Any]:
    cfg = OmegaConf.load(str(path))
    if "video_dit_config" not in cfg or spec.cfg_key not in cfg:
        raise ValueError(
            f"`{path}` must contain both `video_dit_config` and `{spec.cfg_key}` at top level."
        )

    video_cfg = OmegaConf.to_container(cfg.video_dit_config, resolve=False)
    target_cfg = OmegaConf.to_container(getattr(cfg, spec.cfg_key), resolve=False)
    if not isinstance(video_cfg, dict) or not isinstance(target_cfg, dict):
        raise ValueError("`video_dit_config` and target dit config must resolve to dicts.")

    for key, default_value in spec.extra_unresolved_defaults.items():
        if _is_unresolved_interpolation(target_cfg.get(key)):
            print(
                f"[WARN] `{spec.cfg_key}.{key}` is unresolved; defaulting to {default_value} for preprocessing."
            )
            target_cfg[key] = default_value

    for key in spec.shared_with_video_keys:
        target_cfg[key] = _resolve_from_video_cfg(target_cfg.get(key), video_cfg)

    return video_cfg, target_cfg, cfg


def _normalize_target_cfg(target_cfg: dict[str, Any], spec: TargetSpec) -> dict[str, Any]:
    out = dict(target_cfg)
    for key in spec.int_fields:
        out[key] = _require_int_config(out, key)
    for key in spec.float_fields:
        out[key] = _require_float_config(out, key)
    return out


def _validate_mot_compatibility(target_cfg: dict[str, Any], video_expert) -> None:
    if int(target_cfg["num_heads"]) != int(video_expert.num_heads):
        raise ValueError("Target DiT `num_heads` must match video expert for MoT mixed attention.")
    if int(target_cfg["attn_head_dim"]) != int(video_expert.attn_head_dim):
        raise ValueError("Target DiT `attn_head_dim` must match video expert for MoT mixed attention.")
    if int(target_cfg["num_layers"]) != int(len(video_expert.blocks)):
        raise ValueError("Target DiT `num_layers` must match video expert.")


def _build_payload(
    *,
    spec: TargetSpec,
    target_cfg: dict[str, Any],
    target_state: dict[str, torch.Tensor],
    video_state: dict[str, torch.Tensor],
    apply_alpha_scaling: bool,
) -> tuple[dict[str, Any], int, int, int]:
    backbone_keys = spec.model_class.backbone_key_set(target_state.keys())

    backbone_state_dict: dict[str, torch.Tensor] = {}
    copied = 0
    interpolated = 0
    for key in sorted(backbone_keys):
        if key not in video_state:
            raise ValueError(f"Key `{key}` not found in video expert state dict.")
        src = video_state[key]
        target = target_state[key]
        if tuple(src.shape) == tuple(target.shape):
            value = src
            copied += 1
        else:
            value = _resize_tensor_to_shape(src, tuple(target.shape))
            if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target.shape[-1]:
                alpha = (float(src.shape[-1]) / float(target.shape[-1])) ** 0.5
                value = value.to(torch.float32) * alpha
            interpolated += 1
        backbone_state_dict[key] = value.detach().to(dtype=target.dtype, device="cpu").contiguous()

    meta = {
        "hidden_dim": int(target_cfg["hidden_dim"]),
        "ffn_dim": int(target_cfg["ffn_dim"]),
        "num_layers": int(target_cfg["num_layers"]),
        "num_heads": int(target_cfg["num_heads"]),
        "attn_head_dim": int(target_cfg["attn_head_dim"]),
        "text_dim": int(target_cfg["text_dim"]),
        "freq_dim": int(target_cfg["freq_dim"]),
        "eps": float(target_cfg["eps"]),
    }
    if "action_dim" in target_cfg:
        meta["action_dim"] = int(target_cfg["action_dim"])
    if "delta_token_dim" in target_cfg:
        meta["delta_token_dim"] = int(target_cfg["delta_token_dim"])

    payload = {
        "policy": {
            "skip_prefixes": list(spec.model_class.ACTION_BACKBONE_SKIP_PREFIXES)
            if spec.name == "action"
            else list(spec.model_class.DELTA_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": meta,
    }

    skipped = len(target_state) - len(backbone_keys)
    return payload, copied, interpolated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess target DiT backbone weights from WanVideoDiT and save as .pt payload."
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=sorted(TARGET_SPECS.keys()),
        help="Target backbone type to preprocess: action or delta.",
    )
    parser.add_argument("--model-config", required=True, help="Path to model yaml.")
    parser.add_argument("--output", required=True, help="Output .pt path for preprocessed backbone payload.")
    parser.add_argument("--device", default="cpu", help="Device for loading model and preprocessing.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Whether to apply alpha=sqrt(dv/da) when the last dimension is resized (true/false). Default: true.",
    )
    args = parser.parse_args()

    spec = TARGET_SPECS[args.target]
    model_config_path = Path(args.model_config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)
    torch_dtype = _parse_dtype(args.dtype)

    video_cfg, target_cfg_raw, cfg = _load_model_configs(model_config_path, spec)
    target_cfg = _normalize_target_cfg(target_cfg_raw, spec)
    redirect_common_files = _parse_bool(cfg.get("redirect_common_files", False))

    print(
        f"[INFO] Loaded model config from {model_config_path}. "
        f"Preprocessing {spec.output_name} backbone with dtype={torch_dtype} on device={args.device}, "
        f"apply_alpha_scaling={apply_alpha_scaling}."
    )

    components = load_wan22_ti2v_5b_components(
        device=args.device,
        torch_dtype=torch_dtype,
        model_id=cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"),
        tokenizer_model_id=cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
        redirect_common_files=redirect_common_files,
        dit_config=video_cfg,
    )
    video_expert = components.dit

    target_expert = spec.model_class(**target_cfg).to(device=args.device, dtype=torch_dtype)
    # _validate_mot_compatibility(target_cfg, video_expert)

    payload, copied, interpolated, skipped = _build_payload(
        spec=spec,
        target_cfg=target_cfg,
        target_state=target_expert.state_dict(),
        video_state=video_expert.state_dict(),
        apply_alpha_scaling=apply_alpha_scaling,
    )
    torch.save(payload, str(output_path))

    print(
        f"[INFO] Saved {spec.output_name} backbone payload to {output_path} "
        f"(copied={copied}, interpolated={interpolated}, skipped={skipped})."
    )


if __name__ == "__main__":
    main()
