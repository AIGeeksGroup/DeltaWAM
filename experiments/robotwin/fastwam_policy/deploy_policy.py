import logging
import os
import sys
import threading
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deltawam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from deltawam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from deltawam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        timing_num_inference_steps_schedule: Optional[list[int]],
        timing_benchmark_inference_calls: int,
        num_video_frames: int,
        deltakv_observation_interval: int,
        deltakv_refresh_interval: int,
        deltakv_async_encoding: bool,
        deltakv_async_replan_append: bool,
        deltakv_pre_replan_append_steps: int,
        deltakv_batch_pre_replan_append: bool,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self.timing_num_inference_steps_schedule = (
            [int(value) for value in timing_num_inference_steps_schedule]
            if timing_num_inference_steps_schedule is not None
            else None
        )
        if self.timing_num_inference_steps_schedule is not None and any(
            value <= 0 for value in self.timing_num_inference_steps_schedule
        ):
            raise ValueError("All scheduled inference-step counts must be positive.")
        self.timing_benchmark_inference_calls = int(timing_benchmark_inference_calls)
        if self.timing_benchmark_inference_calls < 0:
            raise ValueError("`timing_benchmark_inference_calls` must be non-negative.")
        self._num_video_frames = int(num_video_frames)
        self.deltakv_observation_interval = int(deltakv_observation_interval)
        if self.deltakv_observation_interval < 0:
            raise ValueError("`deltakv_observation_interval` must be non-negative.")
        self.deltakv_refresh_interval = int(deltakv_refresh_interval)
        if self.deltakv_refresh_interval < 0:
            raise ValueError("`deltakv_refresh_interval` must be non-negative.")

        supports_deltakv = hasattr(self.model, "append_observation")
        model_device = torch.device(self.model.device)
        async_requested = bool(deltakv_async_encoding) and supports_deltakv
        if async_requested and model_device.type != "cuda":
            logger.warning(
                "DeltaKV asynchronous encoding requires CUDA; falling back to synchronous encoding."
            )
        self.deltakv_async_encoding = async_requested and model_device.type == "cuda"
        self.deltakv_async_replan_append = bool(deltakv_async_replan_append)
        if self.deltakv_async_replan_append and not self.deltakv_async_encoding:
            raise ValueError(
                "`deltakv_async_replan_append=true` requires CUDA DeltaKV and "
                "`deltakv_async_encoding=true`."
            )
        self.deltakv_pre_replan_append_steps = int(deltakv_pre_replan_append_steps)
        if self.deltakv_pre_replan_append_steps < 0:
            raise ValueError("`deltakv_pre_replan_append_steps` must be non-negative.")
        if self.deltakv_pre_replan_append_steps >= self.replan_steps:
            raise ValueError(
                "`deltakv_pre_replan_append_steps` must be smaller than `replan_steps`."
            )
        if self.deltakv_pre_replan_append_steps and not self.deltakv_async_encoding:
            raise ValueError(
                "`deltakv_pre_replan_append_steps>0` requires CUDA DeltaKV and "
                "`deltakv_async_encoding=true`."
            )
        if self.deltakv_pre_replan_append_steps and self.deltakv_async_replan_append:
            raise ValueError(
                "`deltakv_pre_replan_append_steps>0` and "
                "`deltakv_async_replan_append=true` are mutually exclusive."
            )
        self.deltakv_batch_pre_replan_append = bool(
            deltakv_batch_pre_replan_append
        )
        if self.deltakv_batch_pre_replan_append and not self.deltakv_pre_replan_append_steps:
            raise ValueError(
                "`deltakv_batch_pre_replan_append=true` requires "
                "`deltakv_pre_replan_append_steps>0`."
            )
        if self.deltakv_batch_pre_replan_append and not hasattr(
            self.model, "append_observations"
        ):
            raise ValueError(
                "The DeltaKV model does not implement batched `append_observations`."
            )
        self._deltakv_stream = (
            torch.cuda.Stream(device=model_device) if self.deltakv_async_encoding else None
        )
        self._deltakv_update_pending = False
        self._deltakv_host_thread: Optional[threading.Thread] = None
        self._deltakv_host_error: Optional[BaseException] = None

        self.pending_actions: deque[np.ndarray] = deque()
        self._deltakv_cache = None
        self._deltakv_prompt = None
        self._deltakv_context = None
        self._deltakv_context_mask = None
        self._deltakv_cache_step = None
        self._deltakv_pre_replan_append_done = False
        self._deltakv_pending_observations: list[torch.Tensor] = []
        self._deltakv_inference_count = 0
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0, "infer_calls": 0}
        self._timing_phase_calls: list[Dict[str, Any]] = []

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | "
            "replan=%d | async_delta=%s | async_replan_append=%s | "
            "pre_replan_append_steps=%d | batch_pre_replan_append=%s",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            self.deltakv_async_encoding,
            self.deltakv_async_replan_append,
            self.deltakv_pre_replan_append_steps,
            self.deltakv_batch_pre_replan_append,
        )

    @staticmethod
    def _record_cache_stream(cache: Any, stream: torch.cuda.Stream) -> None:
        """Keep cache storage alive while a side-stream update consumes it."""
        cache.previous_image.record_stream(stream)
        previous_image_features = getattr(cache, "previous_image_features", None)
        if previous_image_features is not None:
            previous_image_features.record_stream(stream)
        delta_context = getattr(cache, "delta_context", None)
        if delta_context is not None:
            delta_context.record_stream(stream)
        delta_context_kv = getattr(cache, "delta_context_kv", None)
        if delta_context_kv is not None:
            for layer in delta_context_kv:
                layer["k"].record_stream(stream)
                layer["v"].record_stream(stream)
        for layer in cache.kv:
            layer["k"].record_stream(stream)
            layer["v"].record_stream(stream)

    def _wait_for_async_delta(self) -> None:
        """Order the action stream after the latest asynchronous cache update."""
        self._join_deltakv_host_thread()
        if not self._deltakv_update_pending:
            return
        assert self._deltakv_stream is not None
        torch.cuda.current_stream(device=self.model.device).wait_stream(self._deltakv_stream)
        self._deltakv_update_pending = False

    def _synchronize_async_delta(self) -> None:
        """Finish side-stream work before releasing an episode's cache tensors."""
        self._join_deltakv_host_thread()
        if not self._deltakv_update_pending:
            return
        assert self._deltakv_stream is not None
        self._deltakv_stream.synchronize()
        self._deltakv_update_pending = False

    def _join_deltakv_host_thread(self) -> None:
        thread = self._deltakv_host_thread
        if thread is None:
            return
        thread.join()
        self._deltakv_host_thread = None
        if self._deltakv_host_error is not None:
            error = self._deltakv_host_error
            self._deltakv_host_error = None
            raise RuntimeError("Deferred DeltaKV replan append failed.") from error

    def _launch_deferred_replan_append(self, image_tensor: torch.Tensor) -> None:
        """Append the current observation after returning the current action graph."""
        self._join_deltakv_host_thread()

        def _worker() -> None:
            try:
                with torch.cuda.device(self.model.device):
                    self._append_deltakv_observation(image_tensor, asynchronous=True)
            except BaseException as error:
                self._deltakv_host_error = error

        self._deltakv_host_thread = threading.Thread(
            target=_worker,
            name="deltakv-replan-append",
            daemon=True,
        )
        self._deltakv_host_thread.start()

    def _append_deltakv_observation(
        self,
        image_tensor: torch.Tensor,
        *,
        asynchronous: bool,
    ) -> None:
        if self._deltakv_cache is None or self._deltakv_context is None:
            raise RuntimeError("DeltaKV cache must be initialized before an intermediate update.")
        if not asynchronous:
            with torch.no_grad():
                self._deltakv_cache = self.model.append_observation(
                    self._deltakv_cache,
                    image_tensor,
                    context=self._deltakv_context,
                    context_mask=self._deltakv_context_mask,
                )
            return

        assert self._deltakv_stream is not None
        current_stream = torch.cuda.current_stream(device=self.model.device)
        self._deltakv_stream.wait_stream(current_stream)
        old_cache = self._deltakv_cache
        self._record_cache_stream(old_cache, self._deltakv_stream)
        image_tensor.record_stream(self._deltakv_stream)
        with torch.cuda.stream(self._deltakv_stream), torch.no_grad():
            self._deltakv_cache = self.model.append_observation(
                old_cache,
                image_tensor,
                context=self._deltakv_context,
                context_mask=self._deltakv_context_mask,
            )
        self._deltakv_update_pending = True

    def _append_deltakv_observations(
        self,
        image_tensors: list[torch.Tensor],
        *,
        asynchronous: bool,
    ) -> None:
        if self._deltakv_cache is None or self._deltakv_context is None:
            raise RuntimeError("DeltaKV cache must be initialized before a batched update.")
        if not image_tensors:
            raise ValueError("Batched DeltaKV update requires at least one observation.")
        append_many = getattr(self.model, "append_observations", None)
        if not callable(append_many):
            raise RuntimeError("DeltaKV model does not implement `append_observations`.")
        if not asynchronous:
            with torch.no_grad():
                self._deltakv_cache = append_many(
                    self._deltakv_cache,
                    image_tensors,
                    context=self._deltakv_context,
                    context_mask=self._deltakv_context_mask,
                )
            return

        assert self._deltakv_stream is not None
        current_stream = torch.cuda.current_stream(device=self.model.device)
        self._deltakv_stream.wait_stream(current_stream)
        old_cache = self._deltakv_cache
        self._record_cache_stream(old_cache, self._deltakv_stream)
        for image_tensor in image_tensors:
            image_tensor.record_stream(self._deltakv_stream)
        with torch.cuda.stream(self._deltakv_stream), torch.no_grad():
            self._deltakv_cache = append_many(
                old_cache,
                image_tensors,
                context=self._deltakv_context,
                context_mask=self._deltakv_context_mask,
            )
        self._deltakv_update_pending = True

    def _next_deltakv_inference_refreshes(self) -> bool:
        return bool(
            self._deltakv_cache is not None
            and self.deltakv_refresh_interval
            and self._deltakv_inference_count > 0
            and self._deltakv_inference_count % self.deltakv_refresh_interval == 0
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        if self._deltakv_prompt != prompt:
            self._deltakv_context, self._deltakv_context_mask = self.model.encode_prompt(prompt)
            self._deltakv_prompt = prompt
        if hasattr(self.model, "encode_reference") and hasattr(self.model, "infer_action_from_cache"):
            return self._infer_deltakv_action_chunk(
                image_tensor=image_tensor,
                proprio=proprio,
                prompt=prompt,
            )
        infer_kwargs = {
            "prompt": None,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
            "context": self._deltakv_context,
            "context_mask": self._deltakv_context_mask,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            wall_s = time.perf_counter() - infer_t0
            self._timing_rollout["infer_s"] += wall_s
            self._timing_phase_calls.append(
                {
                    "call": self._timing_rollout["infer_calls"] + 1,
                    "cache_mode": "fastwam_full",
                    "wall_ms": wall_s * 1000.0,
                }
            )

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _infer_deltakv_action_chunk(
        self,
        *,
        image_tensor: torch.Tensor,
        proprio: torch.Tensor,
        prompt: str,
    ) -> np.ndarray:
        if self._deltakv_prompt != prompt:
            self._deltakv_context, self._deltakv_context_mask = self.model.encode_prompt(prompt)
            self._deltakv_prompt = prompt

        context = self._deltakv_context
        context_mask = self._deltakv_context_mask
        refresh = self._deltakv_cache is None
        if self._deltakv_cache is not None and self.deltakv_refresh_interval:
            refresh = (
                self._deltakv_inference_count > 0
                and self._deltakv_inference_count % self.deltakv_refresh_interval == 0
            )

        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        phase_events = None
        wait_was_pending = self._deltakv_update_pending
        if self.timing_enabled and torch.device(self.model.device).type == "cuda":
            phase_events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            phase_events[0].record(torch.cuda.current_stream(device=self.model.device))
        self._wait_for_async_delta()
        if phase_events is not None:
            phase_events[1].record(torch.cuda.current_stream(device=self.model.device))
        cache_mode = "refresh" if refresh else "cache_ready"
        action_cache = self._deltakv_cache
        deferred_replan_image = None
        with torch.no_grad():
            if refresh:
                self._deltakv_cache = self.model.encode_reference(
                    image_tensor,
                    context=context,
                    context_mask=context_mask,
                )
                action_cache = self._deltakv_cache
            elif self.deltakv_pre_replan_append_steps:
                if not self._deltakv_pre_replan_append_done:
                    raise RuntimeError(
                        "DeltaKV pre-replan append was not prepared before a reuse replan. "
                        "Ensure the rollout supplies the observation requested by "
                        "`should_request_observation()`."
                    )
                # The side-stream append launched N actions before this replan
                # has already been ordered by _wait_for_async_delta(). Reuse it
                # directly and deliberately ignore the current replan image.
                action_cache = self._deltakv_cache
                cache_mode = "pre_replan_append_ready"
            elif self._deltakv_cache_step != self.step_count:
                if self.deltakv_async_replan_append:
                    # Snapshot the cache already ready through t-1. Enqueue the
                    # complete ActionDiT graph first; the current observation is
                    # submitted to the side stream after that enqueue.
                    action_cache = self._deltakv_cache
                    next_inference_refreshes = bool(
                        self.deltakv_refresh_interval
                        and (self._deltakv_inference_count + 1)
                        % self.deltakv_refresh_interval
                        == 0
                    )
                    if next_inference_refreshes:
                        # This cache would be discarded by the next synchronous
                        # reference refresh before any action can consume it.
                        cache_mode = "skip_append_before_refresh"
                    else:
                        deferred_replan_image = image_tensor
                        cache_mode = "async_append_action_uses_previous"
                else:
                    self._append_deltakv_observation(image_tensor, asynchronous=False)
                    action_cache = self._deltakv_cache
                    cache_mode = "append_observation"
            if refresh or not self.deltakv_pre_replan_append_steps:
                self._deltakv_cache_step = self.step_count
            if phase_events is not None:
                phase_events[2].record(torch.cuda.current_stream(device=self.model.device))
            action_tensor = self.model.infer_action_from_cache(
                action_cache,
                action_horizon=self.action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                seed=self.seed,
                rand_device=self.rand_device,
            )
        if phase_events is not None:
            phase_events[3].record(torch.cuda.current_stream(device=self.model.device))
        if phase_events is not None:
            phase_events[3].synchronize()
        self._deltakv_inference_count += 1
        self._deltakv_pre_replan_append_done = False
        if self.timing_enabled:
            wall_s = time.perf_counter() - infer_t0
            self._timing_rollout["infer_s"] += wall_s
            if phase_events is not None:
                self._timing_phase_calls.append(
                    {
                        "call": self._deltakv_inference_count,
                        "cache_mode": cache_mode,
                        "wait_was_pending": wait_was_pending,
                        "wait_ms": phase_events[0].elapsed_time(phase_events[1]),
                        "cache_prepare_ms": phase_events[1].elapsed_time(phase_events[2]),
                        "action_denoise_ms": phase_events[2].elapsed_time(phase_events[3]),
                        "wall_ms": wall_s * 1000.0,
                        "cache_seq_len": int(action_cache.seq_len),
                        "pending_cache_seq_len": int(action_cache.seq_len)
                        + (1 if deferred_replan_image is not None else 0),
                    }
                )
        if deferred_replan_image is not None:
            self._launch_deferred_replan_append(deferred_replan_image)
        return self._denormalize_action(action_tensor)[0]

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        if self.timing_enabled:
            self._timing_rollout["infer_calls"] += 1
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        if not self.pending_actions:
            return True
        if (
            not hasattr(self.model, "append_observation")
            or self._next_deltakv_inference_refreshes()
            or self._deltakv_cache_step == self.step_count
        ):
            return False
        if self.deltakv_pre_replan_append_steps:
            if self._deltakv_pre_replan_append_done:
                return False
            if len(self.pending_actions) == self.deltakv_pre_replan_append_steps:
                return True
        return bool(
            self.deltakv_observation_interval
            and self.step_count > 0
            and self.step_count % self.deltakv_observation_interval == 0
        )

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if (
            self.pending_actions
            and observation is not None
            and hasattr(self.model, "append_observation")
            and (
                not self.deltakv_pre_replan_append_steps
                or self.should_request_observation()
            )
        ):
            self._join_deltakv_host_thread()
            image_tensor = self._build_robotwin_image_tensor(observation)
            update_t0 = time.perf_counter() if self.timing_enabled else 0.0
            is_pre_replan_observation = bool(
                self.deltakv_pre_replan_append_steps
                and len(self.pending_actions) == self.deltakv_pre_replan_append_steps
            )
            if self.deltakv_batch_pre_replan_append:
                self._deltakv_pending_observations.append(image_tensor)
                if is_pre_replan_observation:
                    pending = self._deltakv_pending_observations
                    self._deltakv_pending_observations = []
                    self._append_deltakv_observations(
                        pending,
                        asynchronous=self.deltakv_async_encoding,
                    )
                    self._deltakv_cache_step = self.step_count
            else:
                self._append_deltakv_observation(
                    image_tensor,
                    asynchronous=self.deltakv_async_encoding,
                )
                self._deltakv_cache_step = self.step_count
            if is_pre_replan_observation:
                self._deltakv_pre_replan_append_done = True
            if self.timing_enabled and not self.deltakv_async_encoding:
                self._timing_rollout["infer_s"] += time.perf_counter() - update_t0

        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0
        self._timing_rollout["infer_calls"] = 0
        self._timing_phase_calls.clear()

    def get_timing_rollout(self) -> Dict[str, Any]:
        result = {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
            "infer_calls": int(self._timing_rollout["infer_calls"]),
            "num_inference_steps": self.num_inference_steps,
            "deltakv_async_encoding": self.deltakv_async_encoding,
            "deltakv_async_replan_append": self.deltakv_async_replan_append,
            "deltakv_pre_replan_append_steps": self.deltakv_pre_replan_append_steps,
            "deltakv_batch_pre_replan_append": self.deltakv_batch_pre_replan_append,
        }
        if self._timing_phase_calls:
            result["phase_calls"] = list(self._timing_phase_calls)
        return result

    def timing_benchmark_complete(self) -> bool:
        return bool(
            self.timing_enabled
            and self.timing_benchmark_inference_calls > 0
            and self._timing_rollout["infer_calls"]
            >= self.timing_benchmark_inference_calls
        )

    def timing_benchmark_enabled(self) -> bool:
        return bool(self.timing_enabled and self.timing_benchmark_inference_calls > 0)

    def reset(self) -> None:
        self._synchronize_async_delta()
        self.pending_actions.clear()
        self._deltakv_cache = None
        self._deltakv_prompt = None
        self._deltakv_context = None
        self._deltakv_context_mask = None
        self._deltakv_cache_step = None
        self._deltakv_pre_replan_append_done = False
        self._deltakv_pending_observations.clear()
        self._deltakv_inference_count = 0
        self.episode_count += 1
        if self.timing_num_inference_steps_schedule is not None:
            schedule_index = self.episode_count - 1
            if schedule_index >= len(self.timing_num_inference_steps_schedule):
                raise IndexError(
                    "Timing inference-step schedule has fewer entries than evaluation episodes."
                )
            self.num_inference_steps = self.timing_num_inference_steps_schedule[schedule_index]
        self.step_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        model_horizon = _parse_optional_int(
            OmegaConf.select(cfg, "model.deltakv.action_horizon", default=None)
        )
        if eval_horizon is not None:
            action_horizon = eval_horizon
        elif model_horizon is not None:
            action_horizon = model_horizon
        else:
            action_horizon = int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    timing_num_inference_steps_schedule = usr_args.get(
        "timing_num_inference_steps_schedule"
    )
    if _is_none_like(timing_num_inference_steps_schedule):
        timing_num_inference_steps_schedule = cfg.EVALUATION.get(
            "timing_num_inference_steps_schedule"
        )
    if not _is_none_like(timing_num_inference_steps_schedule):
        timing_num_inference_steps_schedule = [
            int(value) for value in timing_num_inference_steps_schedule
        ]
    else:
        timing_num_inference_steps_schedule = None
    timing_benchmark_inference_calls = _parse_optional_int(
        usr_args.get("timing_benchmark_inference_calls")
    )
    if timing_benchmark_inference_calls is None:
        timing_benchmark_inference_calls = int(
            cfg.EVALUATION.get("timing_benchmark_inference_calls", 0)
        )
    deltakv_refresh_interval = _parse_optional_int(usr_args.get("deltakv_refresh_interval"))
    if deltakv_refresh_interval is None:
        deltakv_refresh_interval = int(cfg.EVALUATION.get("deltakv_refresh_interval", 2))
    deltakv_observation_interval = _parse_optional_int(
        usr_args.get("deltakv_observation_interval")
    )
    if deltakv_observation_interval is None:
        deltakv_observation_interval = int(
            cfg.EVALUATION.get("deltakv_observation_interval", 4)
        )
    deltakv_async_encoding_value = usr_args.get("deltakv_async_encoding")
    if _is_none_like(deltakv_async_encoding_value):
        deltakv_async_encoding_value = cfg.EVALUATION.get("deltakv_async_encoding", False)
    deltakv_async_encoding = _parse_bool(deltakv_async_encoding_value)
    deltakv_async_replan_append_value = usr_args.get("deltakv_async_replan_append")
    if _is_none_like(deltakv_async_replan_append_value):
        deltakv_async_replan_append_value = cfg.EVALUATION.get(
            "deltakv_async_replan_append", False
        )
    deltakv_async_replan_append = _parse_bool(deltakv_async_replan_append_value)
    deltakv_pre_replan_append_steps = _parse_optional_int(
        usr_args.get("deltakv_pre_replan_append_steps")
    )
    if deltakv_pre_replan_append_steps is None:
        deltakv_pre_replan_append_steps = int(
            cfg.EVALUATION.get("deltakv_pre_replan_append_steps", 0)
        )
    deltakv_batch_pre_replan_append_value = usr_args.get(
        "deltakv_batch_pre_replan_append"
    )
    if _is_none_like(deltakv_batch_pre_replan_append_value):
        deltakv_batch_pre_replan_append_value = cfg.EVALUATION.get(
            "deltakv_batch_pre_replan_append", False
        )
    deltakv_batch_pre_replan_append = _parse_bool(
        deltakv_batch_pre_replan_append_value
    )

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        timing_num_inference_steps_schedule=timing_num_inference_steps_schedule,
        timing_benchmark_inference_calls=timing_benchmark_inference_calls,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        deltakv_observation_interval=deltakv_observation_interval,
        deltakv_refresh_interval=deltakv_refresh_interval,
        deltakv_async_encoding=deltakv_async_encoding,
        deltakv_async_replan_append=deltakv_async_replan_append,
        deltakv_pre_replan_append_steps=deltakv_pre_replan_append_steps,
        deltakv_batch_pre_replan_append=deltakv_batch_pre_replan_append,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
