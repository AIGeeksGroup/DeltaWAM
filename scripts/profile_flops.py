#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from deltawam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    build_datasets,
)
from deltawam.trainer import Wan22Trainer
from deltawam.utils.config_resolvers import register_default_resolvers
from deltawam.utils.logging_config import setup_logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile train/inference FLOPs for FastWAM, DeltaWAM, and DeltaWAM-Joint "
            "using existing Hydra configs."
        )
    )
    parser.add_argument(
        "--models",
        default="fastwam,deltawam,deltawam_joint",
        help="Comma-separated model config names to profile.",
    )
    parser.add_argument(
        "--base-overrides",
        nargs="*",
        default=[],
        help=(
            "Hydra overrides shared by all models, e.g. "
            "task=robotwin_uncond_3cam_384_1e-4 batch_size=1 num_workers=0."
        ),
    )
    parser.add_argument(
        "--model-overrides",
        nargs="*",
        default=[],
        help="Hydra overrides appended after model=<name> for every profiled model.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Profiling batch size.")
    parser.add_argument("--device", default="cuda:0", help="Device used for profiling.")
    parser.add_argument(
        "--mixed-precision",
        default=None,
        choices=[None, "no", "fp16", "bf16"],
        help="Override cfg.mixed_precision for model dtype/autocast.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a synthetic batch from config shapes instead of reading the dataset.",
    )
    parser.add_argument(
        "--profile-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Profile training_loss forward and forward+backward.",
    )
    parser.add_argument(
        "--train-profile",
        default="all",
        choices=["all", "forward", "backward"],
        help=(
            "Which training FLOPs profile to run when --profile-train is enabled. "
            "`backward` records forward+backward only."
        ),
    )
    parser.add_argument(
        "--profile-infer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Profile infer_action.",
    )
    parser.add_argument("--infer-steps", type=int, default=4, help="num_inference_steps for infer_action.")
    parser.add_argument(
        "--infer-mode",
        default=None,
        choices=[None, "action_only", "joint"],
        help="Optional DeltaWAM/Joint inference_mode override.",
    )
    parser.add_argument(
        "--output",
        default="flops_profile.json",
        help="JSON output path. A CSV with the same stem is also written.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _compose_cfg(model_name: str, args: argparse.Namespace) -> DictConfig:
    overrides = list(args.base_overrides)
    overrides.append(f"model={model_name}")
    overrides.append(f"batch_size={args.batch_size}")
    overrides.extend(args.model_overrides)
    if args.mixed_precision is not None:
        overrides.append(f"mixed_precision={args.mixed_precision}")
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="train", overrides=overrides)


def _autocast_context(device: torch.device, mixed_precision: str):
    precision = _normalize_mixed_precision(mixed_precision)
    if device.type != "cuda" or precision == "no":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _make_synthetic_batch(cfg: DictConfig, *, batch_size: int) -> dict[str, Any]:
    data_cfg = cfg.data.train
    sample_span = int(data_cfg.num_frames)
    action_video_freq_ratio = int(data_cfg.action_video_freq_ratio)
    video_query_indices = data_cfg.get("video_query_indices", None)
    if video_query_indices is None:
        num_frames = (sample_span - 1) // action_video_freq_ratio + 1
    else:
        num_frames = len(video_query_indices)
    video_h, video_w = [int(v) for v in data_cfg.video_size]
    action_dim = int(data_cfg.processor.action_output_dim)
    proprio_dim = int(data_cfg.processor.proprio_output_dim)
    action_horizon = sample_span - 1

    context_len = int(data_cfg.get("context_len", 128))
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    text_dim = int(
        model_cfg.get("action_delta_dit_config", {}).get(
            "text_dim",
            model_cfg.get("action_dit_config", {}).get(
                "text_dim",
                model_cfg.get("video_dit_config", {}).get("text_dim", 4096),
            ),
        )
    )

    return {
        "video": torch.randn(batch_size, 3, num_frames, video_h, video_w).clamp(-1.0, 1.0),
        "action": torch.randn(batch_size, action_horizon, action_dim),
        "proprio": torch.randn(batch_size, action_horizon, proprio_dim),
        "context": torch.randn(batch_size, context_len, text_dim),
        "context_mask": torch.ones(batch_size, context_len, dtype=torch.bool),
        "action_is_pad": torch.zeros(batch_size, action_horizon, dtype=torch.bool),
        "image_is_pad": torch.zeros(batch_size, num_frames, dtype=torch.bool),
        "prompt": ["synthetic"] * batch_size,
    }


def _load_real_batch(cfg: DictConfig, *, batch_size: int) -> dict[str, Any]:
    train_ds, _ = build_datasets(cfg.data)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return next(iter(loader))


def _count_unique_params(params: list[torch.nn.Parameter]) -> int:
    seen: set[int] = set()
    total = 0
    for param in params:
        ident = id(param)
        if ident in seen:
            continue
        seen.add(ident)
        total += int(param.numel())
    return total


def _optimizer_param_list(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    def add_params(candidate_params) -> None:
        for param in candidate_params:
            if not param.requires_grad:
                continue
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            params.append(param)

    add_params(model.dit.parameters())
    extra_trainable_parameters = getattr(model, "extra_trainable_parameters", None)
    if callable(extra_trainable_parameters):
        add_params(extra_trainable_parameters())
    proprio_encoder = getattr(model, "proprio_encoder", None)
    if proprio_encoder is not None:
        add_params(proprio_encoder.parameters())
    return params


def _param_stats(model: torch.nn.Module) -> dict[str, int]:
    Wan22Trainer._apply_dit_only_train_mode(model)
    optimizer_params = _optimizer_param_list(model)
    return {
        "total_params": _count_unique_params(list(model.parameters())),
        "requires_grad_params": _count_unique_params([p for p in model.parameters() if p.requires_grad]),
        "optimizer_params": _count_unique_params(optimizer_params),
        "dit_params": _count_unique_params(list(model.dit.parameters())),
        "extra_params": _count_unique_params(
            [p for p in optimizer_params if all(p is not q for q in model.dit.parameters())]
        ),
    }


def _event_category(event_key: str) -> str:
    key = event_key.lower()
    if _is_attention_event(key):
        return "attention"
    if "conv" in key:
        return "conv"
    if "mm" in key or "matmul" in key or "addmm" in key or "bmm" in key or "linear" in key:
        return "matmul_linear"
    if "norm" in key or "layer_norm" in key or "rms_norm" in key:
        return "norm"
    if any(token in key for token in ("gelu", "silu", "relu", "mul", "add", "sub", "div")):
        return "elementwise"
    return "other"


def _is_attention_event(event_key: str) -> bool:
    key = event_key.lower()
    return any(
        token in key
        for token in (
            "attention",
            "scaled_dot_product",
            "flash_attn",
            "flash_attention",
            "efficient_attention",
            "_sdp",
            "fmha",
        )
    )


def _profile_callable(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    label: str,
    count_fused_attention_flops: bool = False,
) -> dict[str, Any]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.synchronize(device)

    with profile(
        activities=activities,
        with_flops=True,
        record_shapes=count_fused_attention_flops,
        profile_memory=False,
    ) as prof:
        result = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    total_flops = 0
    category_flops: dict[str, int] = {}
    top_ops: list[dict[str, Any]] = []
    attention_ops: list[dict[str, Any]] = []
    attention_cpu_time_total_us = 0.0
    attention_cuda_time_total_us = 0.0
    attention_calls = 0
    for event in prof.key_averages():
        flops = int(getattr(event, "flops", 0) or 0)
        cpu_time_total_us = float(event.cpu_time_total)
        cuda_time_total_us = float(getattr(event, "cuda_time_total", 0.0))
        total_flops += flops
        category = _event_category(event.key)
        category_flops[category] = category_flops.get(category, 0) + flops
        if _is_attention_event(event.key):
            attention_cpu_time_total_us += cpu_time_total_us
            attention_cuda_time_total_us += cuda_time_total_us
            attention_calls += int(event.count)
            attention_ops.append(
                {
                    "op": event.key,
                    "flops": flops,
                    "cpu_time_total_us": cpu_time_total_us,
                    "cuda_time_total_us": cuda_time_total_us,
                    "calls": int(event.count),
                }
            )
        if flops > 0:
            top_ops.append(
                {
                    "op": event.key,
                    "flops": flops,
                    "cpu_time_total_us": cpu_time_total_us,
                    "cuda_time_total_us": cuda_time_total_us,
                    "calls": int(event.count),
                }
            )

    # PyTorch 2.7 reports zero FLOPs for fused SDPA kernels. Count the two
    # attention matrix multiplications (QK^T and softmax(QK^T)V) from the
    # top-level SDPA input shapes. Softmax/scaling operations are intentionally
    # excluded, matching the usual matmul-only attention FLOPs convention.
    manual_attention_matmul_flops = 0
    if count_fused_attention_flops:
        for event in prof.events():
            if event.name != "aten::scaled_dot_product_attention" or event.flops:
                continue
            shapes = event.input_shapes
            if len(shapes) < 2 or len(shapes[0]) < 3 or len(shapes[1]) < 3:
                continue
            q_shape = [int(value) for value in shapes[0]]
            k_shape = [int(value) for value in shapes[1]]
            if q_shape[:-2] != k_shape[:-2] or q_shape[-1] != k_shape[-1]:
                continue
            batch_heads = math.prod(q_shape[:-2])
            manual_attention_matmul_flops += (
                4 * batch_heads * q_shape[-2] * k_shape[-2] * q_shape[-1]
            )
    if manual_attention_matmul_flops:
        total_flops += manual_attention_matmul_flops
        category_flops["attention"] = (
            category_flops.get("attention", 0) + manual_attention_matmul_flops
        )
    top_ops.sort(key=lambda item: item["flops"], reverse=True)
    attention_ops.sort(
        key=lambda item: (item["flops"], item["cuda_time_total_us"], item["cpu_time_total_us"]),
        reverse=True,
    )

    return {
        "label": label,
        "total_flops": total_flops,
        "total_tflops": total_flops / 1.0e12,
        "category_flops": category_flops,
        "category_tflops": {k: v / 1.0e12 for k, v in sorted(category_flops.items())},
        "attention_op_calls": attention_calls,
        "manual_attention_matmul_flops": manual_attention_matmul_flops,
        "attention_cpu_time_total_us": attention_cpu_time_total_us,
        "attention_cuda_time_total_us": attention_cuda_time_total_us,
        "top_attention_ops": attention_ops[:20],
        "profiler_note": (
            "Fused SDPA QK^T and attention-value matmul FLOPs are counted manually "
            "from recorded tensor shapes; softmax/scaling FLOPs remain excluded."
            if count_fused_attention_flops
            else "Fused/flash attention kernels may be absent from the FLOPs total."
        ),
        "top_ops": top_ops[:20],
        "result": _to_jsonable(result),
    }


def _profile_training(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device,
    mixed_precision: str,
    backward: bool,
) -> dict[str, Any]:
    model.train()
    Wan22Trainer._apply_dit_only_train_mode(model)
    for param in model.parameters():
        param.grad = None

    def run():
        with _autocast_context(device, mixed_precision):
            loss, loss_dict = model.training_loss(batch)
        if backward:
            loss.backward()
            for param in model.parameters():
                param.grad = None
        return {
            "loss": float(loss.detach().float().item()),
            "loss_dict": loss_dict,
        }

    label = "train_forward_backward" if backward else "train_forward"
    return _profile_callable(run, device=device, label=label)


def _profile_inference(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device,
    mixed_precision: str,
    num_inference_steps: int,
    inference_mode: str | None,
) -> dict[str, Any]:
    model.eval()
    video = batch["video"]
    action = batch["action"]
    proprio = batch.get("proprio", None)
    context = batch.get("context", None)
    context_mask = batch.get("context_mask", None)
    input_image = video[0, :, 0].to(device=device)
    action_horizon = int(action.shape[1])
    proprio0 = None
    if proprio is not None:
        proprio0 = proprio[0, 0].to(device=device)
    context0 = context[:1] if context is not None else None
    context_mask0 = context_mask[:1] if context_mask is not None else None

    kwargs: dict[str, Any] = {
        "prompt": None,
        "input_image": input_image,
        "action_horizon": action_horizon,
        "proprio": proprio0,
        "context": context0,
        "context_mask": context_mask0,
        "num_inference_steps": int(num_inference_steps),
        "rand_device": "cpu",
    }
    class_name = model.__class__.__name__.lower()
    if "deltawam" in class_name:
        kwargs["num_delta_tokens"] = int(video.shape[2] - 1)
        if inference_mode is not None:
            kwargs["inference_mode"] = inference_mode

    def run():
        with torch.no_grad():
            with _autocast_context(device, mixed_precision):
                out = model.infer_action(**kwargs)
        return {
            "keys": sorted(out.keys()),
            "action_shape": list(out["action"].shape) if "action" in out else None,
        }

    return _profile_callable(run, device=device, label="infer_action")


def _profile_one_model(model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = _compose_cfg(model_name, args)
    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    logging.info("Instantiating model=%s device=%s dtype=%s", model_name, device, model_dtype)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(device))
    param_stats = _param_stats(model)

    if args.synthetic:
        batch = _make_synthetic_batch(cfg, batch_size=args.batch_size)
    else:
        batch = _load_real_batch(cfg, batch_size=args.batch_size)

    profiles: dict[str, Any] = {}
    if args.profile_train:
        if args.train_profile in ("all", "forward"):
            logging.info("Profiling %s train forward", model_name)
            profiles["train_forward"] = _profile_training(
                model,
                batch,
                device=device,
                mixed_precision=mixed_precision,
                backward=False,
            )
        if args.train_profile in ("all", "backward"):
            logging.info("Profiling %s train forward+backward", model_name)
            profiles["train_forward_backward"] = _profile_training(
                model,
                batch,
                device=device,
                mixed_precision=mixed_precision,
                backward=True,
            )
    if args.profile_infer:
        logging.info("Profiling %s infer_action", model_name)
        profiles["infer_action"] = _profile_inference(
            model,
            batch,
            device=device,
            mixed_precision=mixed_precision,
            num_inference_steps=args.infer_steps,
            inference_mode=args.infer_mode,
        )

    return {
        "model": model_name,
        "class_name": model.__class__.__name__,
        "mixed_precision": mixed_precision,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "synthetic": bool(args.synthetic),
        "param_stats": param_stats,
        "profiles": profiles,
    }


def _write_outputs(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    rows = []
    for result in payload["results"]:
        params = result["param_stats"]
        for label, profile_result in result["profiles"].items():
            rows.append(
                {
                    "model": result["model"],
                    "class_name": result["class_name"],
                    "profile": label,
                    "total_flops": profile_result["total_flops"],
                    "total_tflops": profile_result["total_tflops"],
                    "total_params": params["total_params"],
                    "requires_grad_params": params["requires_grad_params"],
                    "optimizer_params": params["optimizer_params"],
                    "dit_params": params["dit_params"],
                    "extra_params": params["extra_params"],
                    "attention_op_calls": profile_result.get("attention_op_calls", 0),
                    "attention_cpu_time_total_us": profile_result.get("attention_cpu_time_total_us", 0.0),
                    "attention_cuda_time_total_us": profile_result.get("attention_cuda_time_total_us", 0.0),
                    **{
                        f"tflops_{name}": value
                        for name, value in profile_result.get("category_tflops", {}).items()
                    },
                }
            )
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    setup_logging(log_level=logging.INFO)
    register_default_resolvers()
    torch.manual_seed(int(args.seed))

    model_names = [item.strip() for item in str(args.models).split(",") if item.strip()]
    payload = {
        "args": vars(args),
        "results": [],
    }
    for model_name in model_names:
        payload["results"].append(_profile_one_model(model_name, args))

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    _write_outputs(payload, output_path)
    logging.info("Wrote FLOPs profile to %s and %s", output_path, output_path.with_suffix(".csv"))


if __name__ == "__main__":
    main()
