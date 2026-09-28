#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from deltawam.utils.config_resolvers import register_default_resolvers
from profile_flops import _profile_callable


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fastwam", "deltakv"], required=True)
    parser.add_argument("--infer-steps", type=int, default=1)
    parser.add_argument("--replan-steps", type=int, default=12)
    parser.add_argument("--observation-interval", type=int, default=12)
    parser.add_argument("--refresh-interval", type=int, default=4)
    parser.add_argument("--pre-replan-append-steps", type=int, default=4)
    parser.add_argument("--batch-pre-replan-append", action="store_true")
    parser.add_argument(
        "--async-replan-append", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fastwam-wan-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Profile FastWAM with Wan LoRA; disable for non-LoRA checkpoints.",
    )
    parser.add_argument(
        "--cache-previous-image-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--preallocate-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cache-delta-context-kv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-delta-encoder",
        action="store_true",
        help=(
            "Replace DeltaTok/DINO encoding with precomputed [B,1,D] tokens while "
            "retaining the complete Wan append/cache/action execution graph."
        ),
    )
    parser.add_argument(
        "--component-breakdown",
        action="store_true",
        help=(
            "Also profile DeltaKV reference, action, and two-token Wan append "
            "calls at every cache length in the configured refresh cycle."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.infer_steps <= 0:
        raise ValueError("--infer-steps must be positive")
    if args.replan_steps <= 0 or args.observation_interval <= 0:
        raise ValueError("replan and observation intervals must be positive")
    if args.refresh_interval <= 0:
        raise ValueError("--refresh-interval must be positive")
    if not 0 <= args.pre_replan_append_steps < args.replan_steps:
        raise ValueError(
            "--pre-replan-append-steps must be non-negative and smaller than "
            "--replan-steps"
        )
    if args.pre_replan_append_steps and args.async_replan_append:
        raise ValueError(
            "--pre-replan-append-steps and --async-replan-append are mutually exclusive"
        )
    if args.batch_pre_replan_append and not args.pre_replan_append_steps:
        raise ValueError(
            "--batch-pre-replan-append requires --pre-replan-append-steps>0"
        )
    if args.skip_delta_encoder and args.model != "deltakv":
        raise ValueError("--skip-delta-encoder is only valid with --model deltakv")
    register_default_resolvers()
    task = (
        "sub_robotwin_uncond_3cam_384_1e-4"
        if args.model == "fastwam"
        else "sub_robotwin_deltakv_3cam_384_1e-4"
    )
    overrides = [f"task={task}", "batch_size=1", "model.load_text_encoder=false"]
    if args.model == "fastwam":
        overrides.append(
            "model.wan_lora_config.enabled="
            f"{str(args.fastwam_wan_lora).lower()}"
        )
    else:
        overrides.extend(
            [
                "model.deltakv.cache_previous_image_features="
                f"{str(args.cache_previous_image_features).lower()}",
                "model.deltakv.preallocate_kv_cache="
                f"{str(args.preallocate_kv_cache).lower()}",
                "model.deltakv.cache_delta_context_kv="
                f"{str(args.cache_delta_context_kv).lower()}",
            ]
        )
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="train",
            overrides=overrides,
        )
    device = torch.device("cuda:0")
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=str(device)).eval()
    image0 = torch.randn(1, 3, 384, 320, device=device, dtype=torch.bfloat16).clamp_(-1, 1)
    image1 = torch.randn_like(image0).clamp_(-1, 1)
    context = torch.randn(1, 128, 4096, device=device, dtype=torch.bfloat16)
    context_mask = torch.ones(1, 128, device=device, dtype=torch.bool)
    proprio = torch.randn(1, 14, device=device, dtype=torch.bfloat16)
    if args.skip_delta_encoder:
        precomputed_delta = torch.zeros(
            1,
            1,
            int(model.delta_tokenizer.token_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        # The feature-cache payload is opaque to append_observations and is only
        # passed back to the encoder on the next update. The replacement encoder
        # deliberately ignores it, modelling an already available DeltaToken.
        precomputed_features = torch.empty(0, device=device, dtype=torch.bfloat16)

        def _precomputed_pair_with_cache(
            _tokenizer,
            frame_t,
            frame_t1,
            frame_t_features=None,
        ):
            return precomputed_delta, precomputed_features

        def _precomputed_pair(_tokenizer, frame_t, frame_t1):
            return precomputed_delta

        model.delta_tokenizer.encode_pair_with_feature_cache = MethodType(
            _precomputed_pair_with_cache, model.delta_tokenizer
        )
        model.delta_tokenizer.encode_pair = MethodType(
            _precomputed_pair, model.delta_tokenizer
        )
    reuse_calls_per_cycle = args.refresh_interval - 1
    if args.pre_replan_append_steps:
        periodic_offsets = set(
            range(args.observation_interval, args.replan_steps, args.observation_interval)
        )
        periodic_offsets.add(args.replan_steps - args.pre_replan_append_steps)
        pre_replan_append_offsets = sorted(periodic_offsets)
    else:
        pre_replan_append_offsets = []
    appends_per_cycle = reuse_calls_per_cycle * len(pre_replan_append_offsets)

    profiles = {}
    if args.model == "fastwam":
        def fastwam_call():
            result = model.infer_action(
                prompt=None,
                input_image=image0,
                action_horizon=32,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=args.infer_steps,
                rand_device="cpu",
            )
            return {"action_shape": list(result["action"].shape)}

        fastwam_call()
        torch.cuda.synchronize()
        profiles["one_step_replan"] = _profile_callable(
            fastwam_call,
            device=device,
            label="fastwam_one_step_replan",
            count_fused_attention_flops=True,
        )

        def fastwam_configured_replan_cycle():
            shapes = []
            for _ in range(args.refresh_interval):
                result = model.infer_action(
                    prompt=None,
                    input_image=image0,
                    action_horizon=32,
                    proprio=proprio,
                    context=context,
                    context_mask=context_mask,
                    num_inference_steps=args.infer_steps,
                    rand_device="cpu",
                )
                shapes.append(list(result["action"].shape))
            return {"action_shapes": shapes}

        fastwam_configured_replan_cycle()
        torch.cuda.synchronize()
        profiles["configured_replan_cycle"] = _profile_callable(
            fastwam_configured_replan_cycle,
            device=device,
            label="fastwam_configured_replan_cycle",
            count_fused_attention_flops=True,
        )

        if args.component_breakdown:
            video_model = model._video_model()
            with torch.no_grad():
                first_frame_latents = model._encode_input_image_latents_tensor(
                    input_image=image0,
                    tiled=False,
                )
                component_context, component_context_mask = model._append_proprio_to_context(
                    context=context,
                    context_mask=context_mask,
                    proprio=proprio,
                )
                timestep_video = torch.zeros(
                    (1,), device=device, dtype=torch.bfloat16
                )
                component_video_pre = video_model.pre_dit(
                    x=first_frame_latents,
                    timestep=timestep_video,
                    context=component_context,
                    context_mask=component_context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=bool(
                        getattr(video_model, "fuse_vae_embedding_in_latents", False)
                    ),
                )
                component_video_seq_len = int(component_video_pre["tokens"].shape[1])
                component_attention_mask = model._build_mot_attention_mask(
                    video_seq_len=component_video_seq_len,
                    action_seq_len=32,
                    video_tokens_per_frame=int(
                        component_video_pre["meta"]["tokens_per_frame"]
                    ),
                    device=device,
                )
                component_video_kv = model.mot.prefill_video_cache(
                    video_tokens=component_video_pre["tokens"],
                    video_freqs=component_video_pre["freqs"],
                    video_t_mod=component_video_pre["t_mod"],
                    video_context_payload={
                        "context": component_video_pre["context"],
                        "mask": component_video_pre["context_mask"],
                    },
                    video_attention_mask=component_attention_mask[
                        :component_video_seq_len, :component_video_seq_len
                    ],
                )
                component_action_latents = torch.randn(
                    1, 32, model.action_expert.action_dim,
                    device=device, dtype=torch.bfloat16,
                )
                component_action_steps, component_action_deltas = (
                    model.infer_action_scheduler.build_inference_schedule(
                        num_inference_steps=args.infer_steps,
                        device=device,
                        dtype=torch.bfloat16,
                    )
                )

            def vae_image_encode_call():
                latents = model._encode_input_image_latents_tensor(
                    input_image=image0,
                    tiled=False,
                )
                return {"latent_shape": list(latents.shape)}

            def proprio_encode_call():
                embedded, mask = model._append_proprio_to_context(
                    context=context,
                    context_mask=context_mask,
                    proprio=proprio,
                )
                return {"context_shape": list(embedded.shape), "mask_shape": list(mask.shape)}

            def wan_pre_dit_call():
                pre = video_model.pre_dit(
                    x=first_frame_latents,
                    timestep=timestep_video,
                    context=component_context,
                    context_mask=component_context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=bool(
                        getattr(video_model, "fuse_vae_embedding_in_latents", False)
                    ),
                )
                return {"token_shape": list(pre["tokens"].shape)}

            def wan_reference_blocks_call():
                kv = model.mot.prefill_video_cache(
                    video_tokens=component_video_pre["tokens"],
                    video_freqs=component_video_pre["freqs"],
                    video_t_mod=component_video_pre["t_mod"],
                    video_context_payload={
                        "context": component_video_pre["context"],
                        "mask": component_video_pre["context_mask"],
                    },
                    video_attention_mask=component_attention_mask[
                        :component_video_seq_len, :component_video_seq_len
                    ],
                )
                return {"layers": len(kv), "cache_len": int(kv[0]["k"].shape[1])}

            def action_denoise_call():
                action = component_action_latents
                for step_t, step_delta in zip(
                    component_action_steps, component_action_deltas
                ):
                    timestep = step_t.unsqueeze(0).to(
                        dtype=action.dtype, device=device
                    )
                    prediction = model._predict_action_noise_with_cache(
                        latents_action=action,
                        timestep_action=timestep,
                        context=component_context,
                        context_mask=component_context_mask,
                        video_kv_cache=component_video_kv,
                        attention_mask=component_attention_mask,
                        video_seq_len=component_video_seq_len,
                    )
                    action = model.infer_action_scheduler.step(
                        prediction, step_delta, action
                    )
                return {"action_shape": list(action.shape)}

            fastwam_components = {
                "vae_image_encode": vae_image_encode_call,
                "proprio_encode": proprio_encode_call,
                "wan_pre_dit": wan_pre_dit_call,
                "wan_reference_blocks": wan_reference_blocks_call,
                "action_denoise": action_denoise_call,
            }
            component_total = 0
            for key, call in fastwam_components.items():
                profiles[key] = _profile_callable(
                    call,
                    device=device,
                    label=f"fastwam_{key}",
                    count_fused_attention_flops=True,
                )
                component_total += profiles[key]["total_flops"]
            profiles["component_sum"] = {
                "label": "fastwam_component_sum",
                "total_flops": component_total,
                "total_tflops": component_total / 1.0e12,
                "difference_from_one_step_replan_flops": (
                    profiles["one_step_replan"]["total_flops"] - component_total
                ),
            }
    else:
        with torch.no_grad():
            base_cache = model.encode_reference(
                image0, context=context, context_mask=context_mask
            )
            reuse_cache = base_cache
            previous = image0
            for _ in range(args.refresh_interval - 1):
                current = torch.randn_like(previous).clamp_(-1, 1)
                reuse_cache = model.append_observation(
                    reuse_cache, current, context=context, context_mask=context_mask
                )
                previous = current
            append_profile_cache = model.encode_reference(
                image0, context=context, context_mask=context_mask
            )
        torch.cuda.synchronize()

        def refresh_action_call():
            cache = model.encode_reference(image0, context=context, context_mask=context_mask)
            action = model.infer_action_from_cache(
                cache,
                action_horizon=32,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=args.infer_steps,
                rand_device="cpu",
            )
            return {"action_shape": list(action.shape), "cache_len": cache.seq_len}

        def reuse_action_call():
            action = model.infer_action_from_cache(
                reuse_cache,
                action_horizon=32,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=args.infer_steps,
                rand_device="cpu",
            )
            return {"action_shape": list(action.shape), "cache_len": reuse_cache.seq_len}

        def append_call():
            nonlocal append_profile_cache
            append_profile_cache = model.append_observation(
                append_profile_cache,
                image1,
                context=context,
                context_mask=context_mask,
            )
            return {"cache_len": append_profile_cache.seq_len}

        for call in (refresh_action_call, reuse_action_call, append_call):
            call()
            torch.cuda.synchronize()
        profiles["refresh_action"] = _profile_callable(
            refresh_action_call,
            device=device,
            label="deltakv_refresh_action",
            count_fused_attention_flops=True,
        )
        profiles["reuse_action"] = _profile_callable(
            reuse_action_call,
            device=device,
            label="deltakv_reuse_action",
            count_fused_attention_flops=True,
        )
        profiles["append_observation"] = _profile_callable(
            append_call,
            device=device,
            label="deltakv_append_observation",
            count_fused_attention_flops=True,
        )

        def deltakv_configured_replan_cycle():
            # Inference #1 refreshes. Each later inference consumes all periodic
            # Delta KVs plus the explicitly requested early Delta KV prepared
            # before that reuse replan.
            # Scheduling does not change FLOPs, so append and action are
            # profiled together here to count all end-to-end policy work.
            cache = model.encode_reference(
                image0, context=context, context_mask=context_mask
            )
            shapes = []
            action = model.infer_action_from_cache(
                cache,
                action_horizon=32,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=args.infer_steps,
                rand_device="cpu",
            )
            shapes.append(list(action.shape))
            for _ in range(reuse_calls_per_cycle):
                pending_images = [image1 for _ in pre_replan_append_offsets]
                if args.batch_pre_replan_append:
                    cache = model.append_observations(
                        cache,
                        pending_images,
                        context=context,
                        context_mask=context_mask,
                    )
                else:
                    for pending_image in pending_images:
                        cache = model.append_observation(
                            cache,
                            pending_image,
                            context=context,
                            context_mask=context_mask,
                        )
                action = model.infer_action_from_cache(
                    cache,
                    action_horizon=32,
                    proprio=proprio,
                    context=context,
                    context_mask=context_mask,
                    num_inference_steps=args.infer_steps,
                    rand_device="cpu",
                )
                shapes.append(list(action.shape))
            return {
                "action_shapes": shapes,
                "final_cache_len": cache.seq_len,
                "append_count": appends_per_cycle,
                "delta_tokens": appends_per_cycle,
                "wan_append_calls": (
                    reuse_calls_per_cycle
                    if args.batch_pre_replan_append
                    else appends_per_cycle
                ),
            }

        deltakv_configured_replan_cycle()
        torch.cuda.synchronize()
        profiles["configured_replan_cycle"] = _profile_callable(
            deltakv_configured_replan_cycle,
            device=device,
            label="deltakv_configured_replan_cycle",
            count_fused_attention_flops=True,
        )

        if args.component_breakdown:
            if not args.batch_pre_replan_append:
                raise ValueError(
                    "--component-breakdown currently requires --batch-pre-replan-append"
                )

            def make_cache(delta_tokens: int):
                cache = model.encode_reference(
                    image0, context=context, context_mask=context_mask
                )
                remaining = int(delta_tokens)
                while remaining:
                    count = min(len(pre_replan_append_offsets), remaining)
                    cache = model.append_observations(
                        cache,
                        [image1 for _ in range(count)],
                        context=context,
                        context_mask=context_mask,
                    )
                    remaining -= count
                return cache

            def reference_only_call():
                cache = model.encode_reference(
                    image0, context=context, context_mask=context_mask
                )
                return {"cache_len": cache.seq_len}

            profiles["reference_only"] = _profile_callable(
                reference_only_call,
                device=device,
                label="deltakv_reference_only",
                count_fused_attention_flops=True,
            )

            component_action_total = 0
            component_wan_append_total = 0
            delta_tokens_per_append = len(pre_replan_append_offsets)
            for reuse_index in range(args.refresh_interval):
                existing_delta_tokens = reuse_index * delta_tokens_per_append
                action_cache = make_cache(existing_delta_tokens)

                def action_at_cache_call(_cache=action_cache):
                    action = model.infer_action_from_cache(
                        _cache,
                        action_horizon=32,
                        proprio=proprio,
                        context=context,
                        context_mask=context_mask,
                        num_inference_steps=args.infer_steps,
                        rand_device="cpu",
                    )
                    return {"action_shape": list(action.shape), "cache_len": _cache.seq_len}

                action_key = f"action_delta_tokens_{existing_delta_tokens}"
                profiles[action_key] = _profile_callable(
                    action_at_cache_call,
                    device=device,
                    label=f"deltakv_{action_key}",
                    count_fused_attention_flops=True,
                )
                component_action_total += profiles[action_key]["total_flops"]

                if reuse_index + 1 < args.refresh_interval:
                    append_cache = make_cache(existing_delta_tokens)

                    def append_at_cache_call(_cache=append_cache):
                        updated = model.append_observations(
                            _cache,
                            [image1 for _ in pre_replan_append_offsets],
                            context=context,
                            context_mask=context_mask,
                        )
                        return {
                            "start_cache_len": _cache.seq_len,
                            "end_cache_len": updated.seq_len,
                        }

                    append_key = f"wan_append_start_delta_tokens_{existing_delta_tokens}"
                    profiles[append_key] = _profile_callable(
                        append_at_cache_call,
                        device=device,
                        label=f"deltakv_{append_key}",
                        count_fused_attention_flops=True,
                    )
                    component_wan_append_total += profiles[append_key]["total_flops"]

            profiles["component_sum_without_delta_encoder"] = {
                "label": "deltakv_component_sum_without_delta_encoder",
                "total_flops": (
                    profiles["reference_only"]["total_flops"]
                    + component_action_total
                    + component_wan_append_total
                ),
                "total_tflops": (
                    profiles["reference_only"]["total_flops"]
                    + component_action_total
                    + component_wan_append_total
                )
                / 1.0e12,
                "reference_flops": profiles["reference_only"]["total_flops"],
                "action_flops": component_action_total,
                "wan_append_flops": component_wan_append_total,
            }

    payload = {
        "model": args.model,
        "checkpoint_loaded": False,
        "checkpoint_note": "FLOPs depend on the configured execution graph and tensor shapes, not weight values.",
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model_dtype": str(torch.bfloat16),
        "batch_size": 1,
        "input_image_shape": [1, 3, 384, 320],
        "context_shape": [1, 128, 4096],
        "random_seed": None,
        "num_inference_steps": args.infer_steps,
        "action_horizon": 32,
        "replan_steps": args.replan_steps,
        "observation_interval": args.observation_interval,
        "refresh_interval": args.refresh_interval,
        "pre_replan_append_steps": args.pre_replan_append_steps,
        "pre_replan_append_offsets": pre_replan_append_offsets,
        "batch_pre_replan_append": args.batch_pre_replan_append,
        "async_replan_append": args.async_replan_append,
        "cache_previous_image_features": args.cache_previous_image_features,
        "preallocate_kv_cache": args.preallocate_kv_cache,
        "cache_delta_context_kv": args.cache_delta_context_kv,
        "skip_delta_encoder": args.skip_delta_encoder,
        "fastwam_wan_lora": (
            args.fastwam_wan_lora if args.model == "fastwam" else None
        ),
        "measurement_scope": (
            "model pipeline from GPU-resident image/context tensors to CPU action; "
            "includes all refresh, append, and action operations in one refresh interval"
        ),
        "profiles": profiles,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v["total_tflops"] for k, v in profiles.items()}, indent=2))


if __name__ == "__main__":
    main()
