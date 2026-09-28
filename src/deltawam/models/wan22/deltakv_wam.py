from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F

from .deltawam import DeltaWAM, WanDeltaExpert


LayerKV = list[dict[str, torch.Tensor]]


@dataclass
class DeltaKVBuffer:
    """Preallocated layer-wise K/V storage for one linear cache history."""

    layers: LayerKV
    capacity: int
    length: int


@dataclass(frozen=True)
class DeltaKVCache:
    """Logical streaming snapshot backed by an optional shared linear buffer."""

    kv: LayerKV
    previous_image: torch.Tensor
    reference_tokens: int
    delta_tokens: int = 0
    previous_image_features: Optional[torch.Tensor] = None
    delta_context: Optional[torch.Tensor] = None
    delta_context_kv: Optional[LayerKV] = None
    buffer: Optional[DeltaKVBuffer] = None

    @property
    def seq_len(self) -> int:
        return self.reference_tokens + self.delta_tokens

    @property
    def age(self) -> int:
        return self.delta_tokens


class DeltaKVWAM(DeltaWAM):
    """DeltaWAM with frame-to-frame delta tokens as an incremental Wan KV cache."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._require_wan_delta_expert()
        self.stream_min_history_length = 0
        self.stream_max_history_length = 4
        self.stream_update_stride = 2
        self.stream_action_horizon = 16
        self.stream_cache_previous_image_features = True
        self.stream_preallocate_kv_cache = True
        self.stream_cache_delta_context_kv = True
        self.loss_lambda_deltakv_action = self.loss_lambda_action

    def configure_deltakv(
        self,
        *,
        min_history_length: int = 0,
        max_history_length: int = 4,
        update_stride: int = 2,
        action_horizon: int = 16,
        cache_previous_image_features: bool = True,
        preallocate_kv_cache: bool = True,
        cache_delta_context_kv: bool = True,
        lambda_deltakv_action: float = 1.0,
    ) -> "DeltaKVWAM":
        if min_history_length < 0:
            raise ValueError(f"`min_history_length` must be non-negative, got {min_history_length}")
        if max_history_length < min_history_length:
            raise ValueError(
                "`max_history_length` must be no smaller than `min_history_length`, "
                f"got {max_history_length} and {min_history_length}"
            )
        if update_stride <= 0:
            raise ValueError(f"`update_stride` must be positive, got {update_stride}")
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")
        self.stream_min_history_length = int(min_history_length)
        self.stream_max_history_length = int(max_history_length)
        self.stream_update_stride = int(update_stride)
        self.stream_action_horizon = int(action_horizon)
        self.stream_cache_previous_image_features = bool(cache_previous_image_features)
        self.stream_preallocate_kv_cache = bool(preallocate_kv_cache)
        self.stream_cache_delta_context_kv = bool(cache_delta_context_kv)
        self.loss_lambda_deltakv_action = float(lambda_deltakv_action)
        return self

    def _require_wan_delta_expert(self) -> WanDeltaExpert:
        if not isinstance(self.delta_expert, WanDeltaExpert):
            raise TypeError(
                "DeltaKV requires `delta_expert_impl=wan` so observed delta tokens "
                "share the Wan blocks used by the reference-frame cache."
            )
        return self.delta_expert

    def _prepare_text_context(
        self,
        *,
        prompt: Optional[Union[str, Sequence[str]]],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt is not None:
            if context is not None or context_mask is not None:
                raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
            context, context_mask = self.encode_prompt(prompt)
        elif context is None or context_mask is None:
            raise ValueError("Provide either `prompt` or both `context` and `context_mask`.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        return (
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        )

    @staticmethod
    def _full_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

    @staticmethod
    def _append_layer_kv(prefix: LayerKV, suffix: LayerKV) -> LayerKV:
        if len(prefix) != len(suffix):
            raise ValueError(f"KV layer mismatch: {len(prefix)} vs {len(suffix)}")
        return [
            {
                "k": torch.cat((left["k"], right["k"]), dim=1),
                "v": torch.cat((left["v"], right["v"]), dim=1),
            }
            for left, right in zip(prefix, suffix)
        ]

    @staticmethod
    def _allocate_kv_buffer(kv: LayerKV, capacity: int) -> tuple[LayerKV, DeltaKVBuffer]:
        if not kv:
            raise ValueError("Cannot allocate an empty DeltaKV buffer")
        length = int(kv[0]["k"].shape[1])
        if capacity < length:
            raise ValueError(f"KV capacity {capacity} is smaller than initial length {length}")
        storage: LayerKV = []
        active: LayerKV = []
        for layer_index, layer in enumerate(kv):
            if "k" not in layer or "v" not in layer:
                raise ValueError(f"KV layer {layer_index} must contain `k` and `v`")
            if layer["k"].shape[1] != length or layer["v"].shape[1] != length:
                raise ValueError(f"KV layer {layer_index} has inconsistent sequence length")
            stored_layer: dict[str, torch.Tensor] = {}
            active_layer: dict[str, torch.Tensor] = {}
            for name in ("k", "v"):
                source = layer[name]
                shape = list(source.shape)
                shape[1] = capacity
                target = source.new_empty(shape)
                target[:, :length].copy_(source)
                stored_layer[name] = target
                active_layer[name] = target[:, :length]
            storage.append(stored_layer)
            active.append(active_layer)
        buffer = DeltaKVBuffer(layers=storage, capacity=capacity, length=length)
        return active, buffer

    @staticmethod
    def _append_buffered_layer_kv(
        cache: DeltaKVCache,
        suffix: LayerKV,
    ) -> tuple[LayerKV, Optional[DeltaKVBuffer]]:
        buffer = cache.buffer
        if buffer is None:
            return DeltaKVWAM._append_layer_kv(cache.kv, suffix), None
        if not suffix:
            raise ValueError("Cannot append an empty DeltaKV suffix")
        if len(buffer.layers) != len(suffix):
            raise ValueError(f"KV layer mismatch: {len(buffer.layers)} vs {len(suffix)}")
        if buffer.length != cache.seq_len:
            raise RuntimeError(
                "DeltaKV buffer has already advanced beyond this cache. "
                "Buffered caches support linear append histories only."
            )
        num_new = int(suffix[0]["k"].shape[1])
        stop = cache.seq_len + num_new
        if stop > buffer.capacity:
            raise ValueError(
                f"DeltaKV buffer capacity exceeded: requested {stop}, capacity {buffer.capacity}"
            )
        active: LayerKV = []
        for layer_index, (stored, new) in enumerate(zip(buffer.layers, suffix)):
            for name in ("k", "v"):
                if new[name].shape[1] != num_new:
                    raise ValueError(f"KV layer {layer_index} has inconsistent append length")
                expected = list(stored[name].shape)
                expected[1] = num_new
                if list(new[name].shape) != expected:
                    raise ValueError(
                        f"KV layer {layer_index} {name} shape mismatch: "
                        f"expected {tuple(expected)}, got {tuple(new[name].shape)}"
                    )
                stored[name][:, cache.seq_len:stop].copy_(new[name])
            active.append({"k": stored["k"][:, :stop], "v": stored["v"][:, :stop]})
        buffer.length = stop
        return active, buffer

    @staticmethod
    def _delta_causal_mask(prefix_len: int, delta_len: int, device: torch.device) -> torch.Tensor:
        total = prefix_len + delta_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:prefix_len, :prefix_len] = True
        mask[prefix_len:, :prefix_len] = True
        mask[prefix_len:, prefix_len:] = torch.tril(
            torch.ones((delta_len, delta_len), dtype=torch.bool, device=device)
        )
        return mask

    @staticmethod
    def _batched_delta_causal_mask(
        prefix_len: int,
        history_lengths: torch.Tensor,
        delta_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Causal cache mask with independently padded histories per sample."""
        batch_size = int(history_lengths.shape[0])
        total = prefix_len + delta_len
        mask = torch.zeros((batch_size, 1, total, total), dtype=torch.bool, device=device)
        mask[:, :, :prefix_len, :prefix_len] = True
        for batch_index, length in enumerate(history_lengths.tolist()):
            length = int(length)
            for delta_index in range(delta_len):
                row = prefix_len + delta_index
                if delta_index < length:
                    mask[batch_index, 0, row, :prefix_len] = True
                    mask[batch_index, 0, row, prefix_len : row + 1] = True
                else:
                    # Keep padded SDPA rows nonempty.
                    mask[batch_index, 0, row, row] = True
        return mask

    @staticmethod
    def _batched_action_mask(
        prefix_len: int,
        history_lengths: torch.Tensor,
        padded_delta_len: int,
        action_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = int(history_lengths.shape[0])
        video_len = prefix_len + padded_delta_len
        total = video_len + action_len
        mask = torch.zeros((batch_size, 1, total, total), dtype=torch.bool, device=device)
        action_start = video_len
        for batch_index, length in enumerate(history_lengths.tolist()):
            visible_video = prefix_len + int(length)
            mask[batch_index, 0, action_start:, :visible_video] = True
            mask[batch_index, 0, action_start:, action_start:] = True
        return mask

    def _prefill_reference_kv(
        self,
        image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[LayerKV, int]:
        latents = self._encode_video_latents(image.unsqueeze(2), tiled=False)
        pre = self._build_wan_first_frame_pre_state(
            first_frame_latents=latents,
            context=context,
            context_mask=context_mask,
        )
        seq_len = int(pre["tokens"].shape[1])
        kv = self.mot.prefill_video_cache(
            video_tokens=pre["tokens"],
            video_freqs=pre["freqs"],
            video_t_mod=pre["t_mod"],
            video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
            video_attention_mask=self._full_attention_mask(seq_len, image.device),
            video_kv_expert=self._wan_cache_expert(),
        )
        return kv, seq_len

    def _encode_delta_history_kv(
        self,
        *,
        reference_kv: LayerKV,
        reference_len: int,
        delta_tokens: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> LayerKV:
        delta_expert = self._require_wan_delta_expert()
        batch_size, delta_len, _ = delta_tokens.shape
        timestep = torch.zeros((batch_size,), device=delta_tokens.device, dtype=delta_tokens.dtype)
        pre = delta_expert.pre_dit(
            delta_tokens=delta_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        _, delta_kv = self.mot.forward_expert_with_video_cache(
            expert_name="video",
            expert_tokens=pre["tokens"],
            expert_freqs=pre["freqs"],
            expert_t_mod=pre["t_mod"],
            expert_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
            video_kv_cache=reference_kv,
            attention_mask=(
                attention_mask
                if attention_mask is not None
                else self._delta_causal_mask(reference_len, delta_len, delta_tokens.device)
            ),
            video_seq_len=reference_len,
            return_expert_kv=True,
        )
        return self._append_layer_kv(reference_kv, delta_kv)

    def _sample_stream_window(self, sample: dict) -> dict[str, torch.Tensor]:
        video = sample["video"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        proprio = sample.get("proprio")
        if video.ndim != 5 or action.ndim != 3:
            raise ValueError("DeltaKV expects video/action as [B,3,T,H,W] and [B,T,D].")
        source_frames = int(video.shape[2])
        if self.stream_update_stride != 1:
            video = video[:, :, :: self.stream_update_stride]
        if source_frames < self.stream_max_history_length + 2:
            raise ValueError(
                "DeltaKV sample must contain max_history past/current frames plus one future frame; "
                f"got {source_frames} frames for max_history={self.stream_max_history_length}."
            )
        current_index = self.stream_max_history_length
        horizon = self.stream_action_horizon
        if action.shape[1] < horizon:
            raise ValueError(f"DeltaKV requires {horizon} action slots, got {action.shape[1]}.")
        action_is_pad = sample.get("action_is_pad")
        image_is_pad = sample.get("image_is_pad")
        batch_size = int(video.shape[0])
        if image_is_pad is None:
            available_history = torch.full(
                (batch_size,), self.stream_max_history_length, dtype=torch.long, device=self.device
            )
        else:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
            real_history = ~image_is_pad[:, : current_index + 1]
            available_history = real_history.sum(dim=1).sub(1).clamp(
                min=self.stream_min_history_length, max=self.stream_max_history_length
            )
        sampled_history_lengths = sample.get("deltakv_history_length")
        if sampled_history_lengths is None:
            random_unit = torch.rand((batch_size,), device=self.device)
            history_lengths = torch.floor(random_unit * (available_history + 1)).long()
            history_lengths.clamp_(min=self.stream_min_history_length)
        else:
            history_lengths = sampled_history_lengths.to(
                device=self.device, dtype=torch.long, non_blocking=True
            ).reshape(batch_size)
            if bool((history_lengths < self.stream_min_history_length).any()) or bool(
                (history_lengths > available_history).any()
            ):
                raise ValueError(
                    "Dataset-provided DeltaKV history length is outside the available range: "
                    f"history={history_lengths.tolist()}, available={available_history.tolist()}."
                )
        max_history = int(history_lengths.max().item())
        reference_indices = current_index - history_lengths

        batch_indices = torch.arange(batch_size, device=self.device)
        reference_video = video[batch_indices, :, reference_indices]
        if max_history:
            offsets = torch.arange(max_history + 1, device=self.device).unsqueeze(0)
            history_indices = reference_indices.unsqueeze(1) + offsets
            history_indices = torch.minimum(history_indices, torch.full_like(history_indices, current_index))
            gather_index = history_indices[:, None, :, None, None].expand(
                -1, video.shape[1], -1, video.shape[3], video.shape[4]
            )
            history_video = torch.gather(video, dim=2, index=gather_index)
        else:
            history_video = reference_video.unsqueeze(2)

        if action_is_pad is not None:
            action_is_pad = action_is_pad[:, :horizon].to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        current_proprio = None
        if proprio is not None:
            current_proprio = proprio[:, 0].to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
        return {
            "history_video": history_video,
            "reference_video": reference_video,
            "future_video": video[:, :, current_index:],
            "future_image_is_pad": None if image_is_pad is None else image_is_pad[:, current_index:],
            "history_lengths": history_lengths,
            "action": action[:, :horizon],
            "action_is_pad": action_is_pad,
            "proprio": current_proprio,
        }

    def training_loss(self, sample):
        """DeltaKV-conditioned action loss plus the original delta flow loss."""
        window = self._sample_stream_window(sample)
        video = window["history_video"]
        action = window["action"]
        context = sample["context"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = sample["context_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True)
        batch_size = int(action.shape[0])

        history_lengths = window["history_lengths"]
        max_history = int(history_lengths.max().item())
        reference = window["reference_video"]
        reference_kv, reference_len = self._prefill_reference_kv(reference, context, context_mask)
        stream_kv = reference_kv
        delta_target = None
        if max_history:
            prev = video[:, :, :-1].permute(0, 2, 1, 3, 4).flatten(0, 1)
            curr = video[:, :, 1:].permute(0, 2, 1, 3, 4).flatten(0, 1)
            delta_target = self.delta_tokenizer.encode_pair(frame_t=prev, frame_t1=curr)
            delta_target = delta_target.squeeze(1).reshape(
                batch_size, max_history, self.delta_expert.delta_token_dim
            ).to(device=self.device, dtype=self.torch_dtype)
            stream_kv = self._encode_delta_history_kv(
                reference_kv=reference_kv,
                reference_len=reference_len,
                delta_tokens=delta_target,
                context=context,
                context_mask=context_mask,
                attention_mask=self._batched_delta_causal_mask(
                    reference_len, history_lengths, max_history, action.device
                ),
            )
        stream_len = reference_len + max_history

        future_video = window["future_video"]
        delta_supervision_length = int(future_video.shape[2]) - 1
        future_prev = future_video[:, :, :-1].permute(0, 2, 1, 3, 4).flatten(0, 1)
        future_curr = future_video[:, :, 1:].permute(0, 2, 1, 3, 4).flatten(0, 1)
        delta_supervision_target = self.delta_tokenizer.encode_pair(
            frame_t=future_prev,
            frame_t1=future_curr,
        ).squeeze(1).reshape(
            batch_size, delta_supervision_length, self.delta_expert.delta_token_dim
        ).to(device=self.device, dtype=self.torch_dtype)
        delta_supervision_lengths = torch.full_like(history_lengths, delta_supervision_length)
        if window["future_image_is_pad"] is not None:
            future_real = (~window["future_image_is_pad"]).to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
            delta_supervision_lengths = (
                future_real[:, :-1] & future_real[:, 1:]
            ).sum(dim=1)

        current_context = context
        current_context_mask = context_mask
        if self.proprio_encoder is not None:
            if window["proprio"] is None:
                raise ValueError("DeltaKV training requires proprio when `proprio_dim` is enabled.")
            current_context, current_context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=window["proprio"],
            )

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=current_context,
            context_mask=current_context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
            video_kv_cache=stream_kv,
            attention_mask=self._batched_action_mask(
                reference_len,
                history_lengths,
                max_history,
                action.shape[1],
                action.device,
            ),
            video_seq_len=stream_len,
        )
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)
        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if window["action_is_pad"] is not None:
            valid = (~window["action_is_pad"]).to(action_loss_token.dtype)
            action_loss = (action_loss_token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            action_loss = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss.device, dtype=action_loss.dtype
        )
        loss_deltakv_action = (action_loss * action_weight).mean()

        noise_delta = torch.randn_like(delta_supervision_target)
        timestep_delta = self.train_delta_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=delta_supervision_target.dtype
        )
        noisy_delta = self.train_delta_scheduler.add_noise(
            delta_supervision_target, noise_delta, timestep_delta
        )
        target_delta = self.train_delta_scheduler.training_target(
            delta_supervision_target, noise_delta, timestep_delta
        )
        delta_pre = self.delta_expert.pre_dit(
            delta_tokens=noisy_delta,
            timestep=timestep_delta,
            context=current_context,
            context_mask=current_context_mask,
        )
        delta_tokens = self.mot.forward_expert_with_video_cache(
            expert_name="video",
            expert_tokens=delta_pre["tokens"],
            expert_freqs=delta_pre["freqs"],
            expert_t_mod=delta_pre["t_mod"],
            expert_context_payload={"context": delta_pre["context"], "mask": delta_pre["context_mask"]},
            video_kv_cache=stream_kv,
            attention_mask=self._batched_action_mask(
                reference_len,
                history_lengths,
                max_history,
                delta_supervision_length,
                delta_supervision_target.device,
            ),
            video_seq_len=stream_len,
        )
        pred_delta = self.delta_expert.post_dit(delta_tokens, delta_pre)
        delta_loss_token = F.mse_loss(
            pred_delta.float(), target_delta.float(), reduction="none"
        ).mean(dim=2)
        delta_valid = (
            torch.arange(delta_supervision_length, device=self.device).unsqueeze(0)
            < delta_supervision_lengths.unsqueeze(1)
        ).to(delta_loss_token.dtype)
        delta_loss = (delta_loss_token * delta_valid).sum(dim=1) / delta_valid.sum(dim=1).clamp(min=1)
        delta_weight = self.train_delta_scheduler.training_weight(timestep_delta).to(
            delta_loss.device, dtype=delta_loss.dtype
        )
        loss_delta = (delta_loss * delta_weight).mean()

        total = (
            self.loss_lambda_deltakv_action * loss_deltakv_action
            + self.loss_lambda_delta * loss_delta
        )
        return total, {
            "loss_deltakv_action": self.loss_lambda_deltakv_action
            * float(loss_deltakv_action.detach().item()),
            "loss_delta": self.loss_lambda_delta * float(loss_delta.detach().item()),
            "history_length_mean": float(history_lengths.float().mean().detach().item()),
            "history_length_max": float(history_lengths.max().detach().item()),
        }

    @torch.no_grad()
    def encode_reference(
        self,
        image: torch.Tensor,
        *,
        prompt: Optional[Union[str, Sequence[str]]] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> DeltaKVCache:
        """Run one dense reference frame through Wan and create a fresh cache."""
        self.eval()
        self._require_wan_delta_expert()
        context, context_mask = self._prepare_text_context(
            prompt=prompt, context=context, context_mask=context_mask
        )
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W], got {tuple(image.shape)}")
        image = image.to(device=self.device, dtype=self.torch_dtype)
        if image.shape[0] != context.shape[0]:
            raise ValueError(f"Image/context batch mismatch: {image.shape[0]} vs {context.shape[0]}")
        latents = self._encode_video_latents(image.unsqueeze(2), tiled=False)
        pre = self._build_wan_first_frame_pre_state(
            first_frame_latents=latents,
            context=context,
            context_mask=context_mask,
        )
        seq_len = int(pre["tokens"].shape[1])
        kv = self.mot.prefill_video_cache(
            video_tokens=pre["tokens"],
            video_freqs=pre["freqs"],
            video_t_mod=pre["t_mod"],
            video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
            video_attention_mask=self._full_attention_mask(seq_len, image.device),
            video_kv_expert=self._wan_cache_expert(),
        )
        buffer = None
        if self.stream_preallocate_kv_cache:
            kv, buffer = self._allocate_kv_buffer(
                kv,
                capacity=seq_len + self.stream_max_history_length,
            )
        delta_context = None
        delta_context_kv = None
        if self.stream_cache_delta_context_kv:
            delta_expert = self._require_wan_delta_expert()
            delta_context = delta_expert.text_embedding(context)
            delta_context_kv = self.mot.build_cross_attention_cache(
                expert_name="video",
                context=delta_context,
            )
        return DeltaKVCache(
            kv=kv,
            previous_image=image.detach(),
            reference_tokens=seq_len,
            delta_context=delta_context,
            delta_context_kv=delta_context_kv,
            buffer=buffer,
        )

    @torch.no_grad()
    def append_observation(
        self,
        cache: DeltaKVCache,
        image: torch.Tensor,
        *,
        prompt: Optional[Union[str, Sequence[str]]] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> DeltaKVCache:
        """Append clean DeltaTok features for (previous observation, current observation)."""
        return self.append_observations(
            cache,
            [image],
            prompt=prompt,
            context=context,
            context_mask=context_mask,
        )

    @torch.no_grad()
    def append_observations(
        self,
        cache: DeltaKVCache,
        images: Sequence[torch.Tensor],
        *,
        prompt: Optional[Union[str, Sequence[str]]] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> DeltaKVCache:
        """Encode adjacent observations and append all Delta K/V in one causal Wan pass."""
        self.eval()
        delta_expert = self._require_wan_delta_expert()
        context, context_mask = self._prepare_text_context(
            prompt=prompt, context=context, context_mask=context_mask
        )
        if not images:
            raise ValueError("`images` must contain at least one observation.")
        encode_with_cache = getattr(
            self.delta_tokenizer,
            "encode_pair_with_feature_cache",
            None,
        )
        previous_image = cache.previous_image
        previous_image_features = cache.previous_image_features
        encoded_deltas: list[torch.Tensor] = []
        for image_index, image in enumerate(images):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            image = image.to(device=self.device, dtype=self.torch_dtype)
            if image.shape[0] != context.shape[0]:
                raise ValueError(
                    f"Image/context batch mismatch at index {image_index}: "
                    f"{image.shape[0]} vs {context.shape[0]}"
                )
            if image.shape != previous_image.shape:
                raise ValueError(
                    f"Current/previous image shape mismatch at index {image_index}: "
                    f"{tuple(image.shape)} vs {tuple(previous_image.shape)}"
                )
            if self.stream_cache_previous_image_features and callable(encode_with_cache):
                delta, current_image_features = encode_with_cache(
                    frame_t=previous_image,
                    frame_t1=image,
                    frame_t_features=previous_image_features,
                )
            else:
                delta = self.delta_tokenizer.encode_pair(
                    frame_t=previous_image,
                    frame_t1=image,
                )
                current_image_features = None
            encoded_deltas.append(
                delta.to(device=self.device, dtype=self.torch_dtype)
            )
            previous_image = image.detach()
            previous_image_features = current_image_features

        delta = torch.cat(encoded_deltas, dim=1)
        timestep = torch.zeros(
            (delta.shape[0],), device=delta.device, dtype=self.torch_dtype
        )
        pre = delta_expert.pre_dit(
            delta_tokens=delta,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            precomputed_context=cache.delta_context,
        )
        num_new = int(pre["tokens"].shape[1])
        start = cache.delta_tokens
        stop = start + num_new
        if stop > delta_expert.freqs.shape[0]:
            raise ValueError(f"DeltaKV history exceeds RoPE cache: {stop} tokens")
        pre["freqs"] = delta_expert.freqs[start:stop].view(num_new, 1, -1).to(delta.device)
        total_len = cache.seq_len + num_new
        _, delta_kv = self.mot.forward_expert_with_video_cache(
            expert_name="video",
            expert_tokens=pre["tokens"],
            expert_freqs=pre["freqs"],
            expert_t_mod=pre["t_mod"],
            expert_context_payload={
                "context": pre["context"],
                "mask": pre["context_mask"],
                "layer_kv": cache.delta_context_kv,
            },
            video_kv_cache=cache.kv,
            attention_mask=self._delta_causal_mask(cache.seq_len, num_new, delta.device),
            video_seq_len=cache.seq_len,
            return_expert_kv=True,
        )
        updated_kv, buffer = self._append_buffered_layer_kv(cache, delta_kv)
        return DeltaKVCache(
            kv=updated_kv,
            previous_image=previous_image,
            reference_tokens=cache.reference_tokens,
            delta_tokens=stop,
            previous_image_features=previous_image_features,
            delta_context=cache.delta_context,
            delta_context_kv=cache.delta_context_kv,
            buffer=buffer,
        )

    @torch.no_grad()
    def infer_action_from_cache(
        self,
        cache: DeltaKVCache,
        *,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        prompt: Optional[Union[str, Sequence[str]]] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 10,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> torch.Tensor:
        """Denoise one action chunk against an explicit DeltaKV cache."""
        self.eval()
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")
        context, context_mask = self._prepare_text_context(
            prompt=prompt, context=context, context_mask=context_mask
        )
        if context.shape[0] != cache.previous_image.shape[0]:
            raise ValueError(
                f"Cache/context batch mismatch: {cache.previous_image.shape[0]} vs {context.shape[0]}"
            )
        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn(
            (context.shape[0], action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        timesteps, steps = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=action.dtype,
            shift_override=sigma_shift,
        )
        mask = self._full_attention_mask(cache.seq_len + action_horizon, action.device)
        for timestep, step in zip(timesteps, steps):
            pre = self.action_expert.pre_dit(
                action_tokens=action,
                timestep=timestep.unsqueeze(0).expand(action.shape[0]),
                context=context,
                context_mask=context_mask,
            )
            tokens = self.mot.forward_action_with_video_cache(
                action_tokens=pre["tokens"],
                action_freqs=pre["freqs"],
                action_t_mod=pre["t_mod"],
                action_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
                video_kv_cache=cache.kv,
                attention_mask=mask,
                video_seq_len=cache.seq_len,
            )
            prediction = self.action_expert.post_dit(tokens, pre)
            action = self.infer_action_scheduler.step(prediction, step, action)
        return action.detach().to(device="cpu", dtype=torch.float32)
