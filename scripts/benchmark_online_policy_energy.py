#!/usr/bin/env python
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import threading
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from deltawam.utils.config_resolvers import register_default_resolvers


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


class NvmlDevice:
    def __init__(self, gpu_index: int):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        self.lib.nvmlInit_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetPowerUsage.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        self.lib.nvmlDeviceGetPowerUsage.restype = ctypes.c_int
        self.lib.nvmlDeviceGetTotalEnergyConsumption.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulonglong),
        ]
        self.lib.nvmlDeviceGetTotalEnergyConsumption.restype = ctypes.c_int
        self.lib.nvmlShutdown.restype = ctypes.c_int
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")
        self.handle = ctypes.c_void_p()
        self._check(
            self.lib.nvmlDeviceGetHandleByIndex_v2(
                int(gpu_index), ctypes.byref(self.handle)
            ),
            "nvmlDeviceGetHandleByIndex_v2",
        )

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"{operation} failed with NVML status {status}")

    def power_w(self) -> float:
        value = ctypes.c_uint()
        self._check(
            self.lib.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(value)),
            "nvmlDeviceGetPowerUsage",
        )
        return value.value / 1000.0

    def total_energy_j(self) -> float:
        value = ctypes.c_ulonglong()
        self._check(
            self.lib.nvmlDeviceGetTotalEnergyConsumption(
                self.handle, ctypes.byref(value)
            ),
            "nvmlDeviceGetTotalEnergyConsumption",
        )
        return value.value / 1000.0

    def close(self) -> None:
        self._check(self.lib.nvmlShutdown(), "nvmlShutdown")


class PowerSampler:
    def __init__(self, device: NvmlDevice, interval_ms: int = 20):
        self.device = device
        self.interval_s = interval_ms / 1000.0
        self.samples: list[float] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            started = time.perf_counter()
            self.samples.append(self.device.power_w())
            self.stop_event.wait(max(0.0, self.interval_s - (time.perf_counter() - started)))

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()


def physical_gpu_index(device: torch.device) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    return int(visible[device.index or 0])


def measure_idle_power(device: NvmlDevice, seconds: float = 2.0) -> float:
    sampler = PowerSampler(device)
    sampler.start()
    time.sleep(seconds)
    sampler.stop()
    if not sampler.samples:
        raise RuntimeError("No idle-power samples collected")
    return sum(sampler.samples) / len(sampler.samples)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fastwam", "deltakv"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup-cycles", type=int, default=2)
    parser.add_argument("--latency-cycles", type=int, default=6)
    parser.add_argument("--idle-settle-seconds", type=float, default=3.0)
    parser.add_argument("--idle-sample-seconds", type=float, default=5.0)
    parser.add_argument("--infer-steps", type=int, default=1)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--observation-interval", type=int, default=4)
    parser.add_argument("--refresh-interval", type=int, default=4)
    parser.add_argument("--pre-replan-append-steps", type=int, default=0)
    parser.add_argument("--batch-pre-replan-append", action="store_true")
    parser.add_argument("--async-replan-append", action="store_true")
    parser.add_argument("--kv-lag", action="store_true")
    parser.add_argument(
        "--fastwam-wan-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Instantiate FastWAM with Wan LoRA; disable for non-LoRA checkpoints.",
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.cycles <= 0 or args.trials <= 0:
        raise ValueError("--cycles and --trials must be positive")
    if args.warmup_cycles < 0 or args.latency_cycles <= 0:
        raise ValueError("--warmup-cycles must be non-negative and --latency-cycles positive")
    if (
        args.replan_steps <= 0
        or args.observation_interval <= 0
        or args.replan_steps % args.observation_interval
    ):
        raise ValueError(
            "--observation-interval must be a positive divisor of --replan-steps"
        )
    if args.refresh_interval <= 0:
        raise ValueError("--refresh-interval must be positive")
    if args.idle_settle_seconds < 0 or args.idle_sample_seconds <= 0:
        raise ValueError("idle settle/sample durations must be non-negative/positive")
    if not 0 <= args.pre_replan_append_steps < args.replan_steps:
        raise ValueError(
            "--pre-replan-append-steps must be non-negative and smaller than "
            "--replan-steps"
        )
    if args.pre_replan_append_steps and (args.async_replan_append or args.kv_lag):
        raise ValueError(
            "--pre-replan-append-steps is mutually exclusive with "
            "--async-replan-append/--kv-lag"
        )
    if args.batch_pre_replan_append and not args.pre_replan_append_steps:
        raise ValueError(
            "--batch-pre-replan-append requires --pre-replan-append-steps>0"
        )
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
    model.load_checkpoint(str(Path(args.checkpoint).resolve()))

    torch.manual_seed(0)
    reuse_calls_per_cycle = args.refresh_interval - 1
    appends_before_replan = args.replan_steps // args.observation_interval - 1
    lagged_replan_append = args.kv_lag or args.async_replan_append
    if args.pre_replan_append_steps:
        periodic_offsets = set(
            range(args.observation_interval, args.replan_steps, args.observation_interval)
        )
        periodic_offsets.add(args.replan_steps - args.pre_replan_append_steps)
        pre_replan_append_offsets = sorted(periodic_offsets)
        appends_per_reuse_window = len(pre_replan_append_offsets)
        # Updates in the chunk leading to a reference refresh are skipped.
        appends_per_cycle = reuse_calls_per_cycle * appends_per_reuse_window
    else:
        pre_replan_append_offsets = []
        appends_per_reuse_window = 0
        appends_per_cycle = (
            reuse_calls_per_cycle * appends_before_replan
            + max(reuse_calls_per_cycle - 1, 0)
            if lagged_replan_append
            else reuse_calls_per_cycle * (appends_before_replan + 1)
        )
    wan_append_calls_per_cycle = (
        reuse_calls_per_cycle
        if args.batch_pre_replan_append
        else appends_per_cycle
    )
    images = [
        torch.randn(1, 3, 384, 320, device=device, dtype=torch.bfloat16).clamp_(-1, 1)
        for _ in range(max(args.refresh_interval, appends_per_cycle + 1))
    ]
    context = torch.randn(1, 128, 4096, device=device, dtype=torch.bfloat16)
    context_mask = torch.ones(1, 128, device=device, dtype=torch.bool)
    proprio = torch.randn(1, 14, device=device, dtype=torch.bfloat16)

    def action_fastwam(image: torch.Tensor) -> None:
        model.infer_action(
            prompt=None,
            input_image=image,
            action_horizon=32,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=args.infer_steps,
            rand_device="cpu",
        )

    def cycle_fastwam() -> None:
        for index in range(args.refresh_interval):
            action_fastwam(images[index])

    def cycle_deltakv() -> None:
        # One refresh action followed by refresh_interval-1 cache-reuse actions.
        # Pre-replan mode performs all periodic updates plus the explicitly
        # requested early append ahead of each reuse action. This
        # serial synthetic workload preserves all compute for energy accounting;
        # critical-path latency is measured separately below.
        cache = model.encode_reference(images[0], context=context, context_mask=context_mask)
        model.infer_action_from_cache(
            cache,
            action_horizon=32,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=args.infer_steps,
            rand_device="cpu",
        )
        image_index = 1
        for reuse_index in range(reuse_calls_per_cycle):
            if args.pre_replan_append_steps:
                pending_images = []
                for _ in pre_replan_append_offsets:
                    pending_images.append(images[image_index])
                    image_index += 1
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
            # Intermediate observations produced while the previous action
            # chunk is executing are ready before the next replan.
            for _ in range(0 if args.pre_replan_append_steps else appends_before_replan):
                cache = model.append_observation(
                    cache,
                    images[image_index],
                    context=context,
                    context_mask=context_mask,
                )
                image_index += 1
            # Without KV lag, the boundary/current observation is appended
            # synchronously and consumed by this action. With KV lag, the action
            # uses the cache through t-1 and the boundary append is deferred.
            if not args.pre_replan_append_steps and not lagged_replan_append:
                cache = model.append_observation(
                    cache,
                    images[image_index],
                    context=context,
                    context_mask=context_mask,
                )
                image_index += 1
            model.infer_action_from_cache(
                cache,
                action_horizon=32,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=args.infer_steps,
                rand_device="cpu",
            )
            if (
                not args.pre_replan_append_steps
                and lagged_replan_append
                and reuse_index < reuse_calls_per_cycle - 1
            ):
                cache = model.append_observation(
                    cache,
                    images[image_index],
                    context=context,
                    context_mask=context_mask,
                )
                image_index += 1

    cycle = cycle_fastwam if args.model == "fastwam" else cycle_deltakv
    gpu_index = physical_gpu_index(device)
    nvml_device = NvmlDevice(gpu_index)
    torch.cuda.synchronize(device)
    # Let the device settle after checkpoint loading before establishing the
    # model-resident idle baseline used for dynamic energy.
    time.sleep(args.idle_settle_seconds)
    idle_power_w = measure_idle_power(nvml_device, seconds=args.idle_sample_seconds)

    # Keep the GPU warm immediately before formal cumulative-energy sampling.
    with torch.no_grad():
        for _ in range(args.warmup_cycles):
            cycle()
        torch.cuda.synchronize(device)

    resident_allocated_bytes = torch.cuda.memory_allocated(device)
    resident_reserved_bytes = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)

    trial_results = []
    chunks_per_trial = args.cycles * args.refresh_interval
    for trial_index in range(args.trials):
        sampler = PowerSampler(nvml_device)
        torch.cuda.synchronize(device)
        sampler.start()
        started = time.perf_counter()
        energy_started_j = nvml_device.total_energy_j()
        with torch.no_grad():
            for _ in range(args.cycles):
                cycle()
            torch.cuda.synchronize(device)
        measured_energy_j = nvml_device.total_energy_j() - energy_started_j
        duration_s = time.perf_counter() - started
        sampler.stop()
        if not sampler.samples:
            raise RuntimeError("No active-power samples collected")
        mean_power_w = measured_energy_j / duration_s
        trial_results.append(
            {
                "trial": trial_index + 1,
                "duration_s": duration_s,
                "seconds_per_chunk": duration_s / chunks_per_trial,
                "throughput_chunks_s": chunks_per_trial / duration_s,
                "power_samples": len(sampler.samples),
                "mean_power_w": mean_power_w,
                "peak_power_w": max(sampler.samples),
                "total_energy_j_per_chunk": measured_energy_j / chunks_per_trial,
                "dynamic_energy_j_per_chunk": max(
                    measured_energy_j - idle_power_w * duration_s,
                    0.0,
                )
                / chunks_per_trial,
            }
        )

    # Measure observation-to-action response latency separately from the total
    # compute workload above. In pre-replan mode, append_observation has had four
    # environment steps to finish and is therefore outside a reuse replan's
    # critical path. It still remains inside energy and FLOPs accounting.
    critical_path_latencies_s: list[float] = []
    refresh_latencies_s: list[float] = []
    reuse_latencies_s: list[float] = []

    def timed_cuda_call(call) -> float:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        call()
        torch.cuda.synchronize(device)
        return time.perf_counter() - started

    with torch.no_grad():
        for _ in range(args.latency_cycles):
            if args.model == "fastwam":
                for image_index in range(args.refresh_interval):
                    latency_s = timed_cuda_call(lambda i=image_index: action_fastwam(images[i]))
                    critical_path_latencies_s.append(latency_s)
                    refresh_latencies_s.append(latency_s)
                continue

            cache = None

            def refresh_call() -> None:
                nonlocal cache
                cache = model.encode_reference(
                    images[0], context=context, context_mask=context_mask
                )
                model.infer_action_from_cache(
                    cache,
                    action_horizon=32,
                    proprio=proprio,
                    context=context,
                    context_mask=context_mask,
                    num_inference_steps=args.infer_steps,
                    rand_device="cpu",
                )

            latency_s = timed_cuda_call(refresh_call)
            critical_path_latencies_s.append(latency_s)
            refresh_latencies_s.append(latency_s)
            assert cache is not None
            latency_image_index = 1
            for reuse_index in range(reuse_calls_per_cycle):
                # Exact requested pre-replan schedule: the append finishes while
                # earlier actions execute, before observation-to-action timing.
                if args.pre_replan_append_steps:
                    pending_images = []
                    for _ in pre_replan_append_offsets:
                        pending_images.append(images[latency_image_index])
                        latency_image_index += 1
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
                    torch.cuda.synchronize(device)

                def reuse_call() -> None:
                    nonlocal cache
                    if not args.pre_replan_append_steps and not lagged_replan_append:
                        cache = model.append_observation(
                            cache,
                            images[reuse_index + 1],
                            context=context,
                            context_mask=context_mask,
                        )
                    model.infer_action_from_cache(
                        cache,
                        action_horizon=32,
                        proprio=proprio,
                        context=context,
                        context_mask=context_mask,
                        num_inference_steps=args.infer_steps,
                        rand_device="cpu",
                    )

                latency_s = timed_cuda_call(reuse_call)
                critical_path_latencies_s.append(latency_s)
                reuse_latencies_s.append(latency_s)

    duration_s = sum(item["duration_s"] for item in trial_results)
    chunks = chunks_per_trial * args.trials
    total_energy_j = sum(
        item["total_energy_j_per_chunk"] * chunks_per_trial
        for item in trial_results
    )
    dynamic_energy_j = sum(
        item["dynamic_energy_j_per_chunk"] * chunks_per_trial
        for item in trial_results
    )
    mean_power_w = total_energy_j / duration_s
    total_power_samples = sum(item["power_samples"] for item in trial_results)
    result = {
        "model": args.model,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_loaded": True,
        "physical_gpu_index": gpu_index,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model_dtype": str(torch.bfloat16),
        "batch_size": 1,
        "input_image_shape": [1, 3, 384, 320],
        "context_shape": [1, 128, 4096],
        "action_horizon": 32,
        "random_seed": 0,
        "cycles_per_trial": args.cycles,
        "trials": args.trials,
        "chunks": chunks,
        "num_inference_steps": args.infer_steps,
        "replan_steps": args.replan_steps,
        "observation_interval": args.observation_interval,
        "refresh_interval": args.refresh_interval,
        "pre_replan_append_steps": args.pre_replan_append_steps,
        "pre_replan_append_offsets": pre_replan_append_offsets,
        "batch_pre_replan_append": args.batch_pre_replan_append,
        "async_replan_append": args.async_replan_append,
        "kv_lag": args.kv_lag,
        "cache_previous_image_features": (
            args.cache_previous_image_features if args.model == "deltakv" else None
        ),
        "preallocate_kv_cache": (
            args.preallocate_kv_cache if args.model == "deltakv" else None
        ),
        "cache_delta_context_kv": (
            args.cache_delta_context_kv if args.model == "deltakv" else None
        ),
        "fastwam_wan_lora": (
            args.fastwam_wan_lora if args.model == "fastwam" else None
        ),
        "duration_s": duration_s,
        "seconds_per_chunk": duration_s / chunks,
        "throughput_chunks_s": chunks / duration_s,
        "power_samples": total_power_samples,
        "energy_source": "NVML total_energy_consumption counter",
        "measurement_scope": (
            "model pipeline from GPU-resident image/context tensors to CPU action; "
            "prompt encoding, image preprocessing, environment execution, and idle gaps excluded"
        ),
        "idle_power_w": idle_power_w,
        "mean_power_w": mean_power_w,
        "peak_power_w": max(item["peak_power_w"] for item in trial_results),
        "total_energy_j_per_chunk": total_energy_j / chunks,
        "dynamic_energy_j_per_chunk": dynamic_energy_j / chunks,
        "critical_path_latency_s_mean": statistics.mean(critical_path_latencies_s),
        "critical_path_latency_s_std": statistics.pstdev(critical_path_latencies_s),
        "critical_path_latency_s_p50": percentile(critical_path_latencies_s, 0.50),
        "critical_path_latency_s_p95": percentile(critical_path_latencies_s, 0.95),
        "refresh_latency_s_mean": statistics.mean(refresh_latencies_s),
        "reuse_latency_s_mean": (
            statistics.mean(reuse_latencies_s) if reuse_latencies_s else None
        ),
        "latency_samples": len(critical_path_latencies_s),
        "idle_settle_seconds": args.idle_settle_seconds,
        "idle_sample_seconds": args.idle_sample_seconds,
        "total_energy_j_per_chunk_std": statistics.pstdev(
            item["total_energy_j_per_chunk"] for item in trial_results
        ),
        "mean_power_w_std": statistics.pstdev(
            item["mean_power_w"] for item in trial_results
        ),
        "resident_allocated_gib": resident_allocated_bytes / (1024**3),
        "resident_reserved_gib": resident_reserved_bytes / (1024**3),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        "deltakv_appends_per_cycle": appends_per_cycle if args.model == "deltakv" else 0,
        "deltakv_delta_tokens_per_cycle": (
            appends_per_cycle if args.model == "deltakv" else 0
        ),
        "deltakv_wan_append_calls_per_cycle": (
            wan_append_calls_per_cycle if args.model == "deltakv" else 0
        ),
        "trial_results": trial_results,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    nvml_device.close()


if __name__ == "__main__":
    main()
