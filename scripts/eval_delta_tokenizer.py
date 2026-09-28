import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as transforms_F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import numpy as np

from deltawam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from deltawam.models.wan22.delta_tokenizer import (
    DeltaTokEncoderTokenizer,
    LAMDeltaTokenizer,
)
from deltawam.utils.video_metrics import video_psnr, video_ssim


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data/robotwin2.0/robotwin2.0"
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs/data/robotwin.yaml"
DEFAULT_DELTATOK_CONFIG = PROJECT_ROOT / "configs/model/deltawam.yaml"
DEFAULT_LAM_CONFIG = PROJECT_ROOT / "configs/model/deltawam_lam.yaml"
@dataclass
class TokenizerRunConfig:
    name: str
    model_config_path: Path
    tokenizer: torch.nn.Module
    reconstruction_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DeltaTokenizer reconstruction quality on a LeRobot-format dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="LeRobot-format dataset root. Default: ./data/robotwin2.0/robotwin2.0",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="Dataset yaml used for shape_meta defaults.",
    )
    parser.add_argument(
        "--tokenizers",
        choices=["both", "deltatok", "lam"],
        default="both",
        help="Which tokenizer(s) to evaluate.",
    )
    parser.add_argument(
        "--deltatok-config",
        type=Path,
        default=DEFAULT_DELTATOK_CONFIG,
        help="Model config for DeltaTokEncoderTokenizer.",
    )
    parser.add_argument(
        "--lam-config",
        type=Path,
        default=DEFAULT_LAM_CONFIG,
        help="Model config for LAMDeltaTokenizer.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Torch dtype for tokenizer inference.",
    )
    parser.add_argument("--num-samples", type=int, default=8, help="Number of dataset samples to visualize.")
    parser.add_argument("--start-index", type=int, default=0, help="Dataset index to start scanning from.")
    parser.add_argument("--max-scan", type=int, default=200, help="Max dataset indices to scan for valid samples.")
    parser.add_argument(
        "--frame-gap",
        type=int,
        default=1,
        help="Temporal gap in raw dataset frames between frame_t and frame_t+gap. Default: 1",
    )
    parser.add_argument(
        "--layout",
        choices=["robotwin", "horizontal", "vertical", "single"],
        default="robotwin",
        help="Multi-camera composition mode.",
    )
    parser.add_argument(
        "--camera-key",
        choices=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        default="cam_high",
        help="Camera to use when --layout=single.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to runs/delta_tokenizer_eval/<timestamp>",
    )
    return parser.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def to_project_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    x_stats = x.float()
    if x_stats.min().item() < -0.05 or x_stats.max().item() > 1.05:
        return (x.clamp(-1.0, 1.0) + 1.0) * 0.5
    return x.clamp(0.0, 1.0)


def to_signed_range(x: torch.Tensor) -> torch.Tensor:
    x = to_unit_range(x)
    return x * 2.0 - 1.0


def tensor_chw_to_pil(x: torch.Tensor) -> Image.Image:
    x = to_unit_range(x).detach().cpu().float()
    arr = (x * 255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def draw_caption(image: Image.Image, text: str) -> Image.Image:
    pad = 24
    canvas = Image.new("RGB", (image.width, image.height + pad), color=(255, 255, 255))
    canvas.paste(image, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), text, fill=(0, 0, 0))
    return canvas


def make_panel(frame_t: torch.Tensor, frame_t1: torch.Tensor, recon_t1: torch.Tensor) -> Image.Image:
    diff = (to_unit_range(recon_t1) - to_unit_range(frame_t1)).abs()
    diff = diff / max(float(diff.max().item()), 1e-6)
    tiles = [
        draw_caption(tensor_chw_to_pil(frame_t), "frame_t"),
        draw_caption(tensor_chw_to_pil(frame_t1), "gt_t+1"),
        draw_caption(tensor_chw_to_pil(recon_t1), "recon_t+1"),
        draw_caption(tensor_chw_to_pil(diff), "abs_diff"),
    ]
    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    panel = Image.new("RGB", (width, height), color=(255, 255, 255))
    offset = 0
    for tile in tiles:
        panel.paste(tile, (offset, 0))
        offset += tile.width
    return panel


def load_shape_meta(data_config_path: Path) -> dict[str, Any]:
    cfg = OmegaConf.load(str(data_config_path))
    return OmegaConf.to_container(cfg.train.shape_meta, resolve=True)


def build_dataset(dataset_dir: Path, data_config_path: Path, frame_gap: int) -> BaseLerobotDataset:
    if frame_gap <= 0:
        raise ValueError(f"`frame_gap` must be positive, got {frame_gap}")
    dataset = BaseLerobotDataset(
        dataset_dirs=[str(dataset_dir)],
        shape_meta=load_shape_meta(data_config_path),
        action_size=1,
        obs_size=2,
        val_set_proportion=0.0,
        is_training_set=False,
        global_sample_stride=frame_gap,
    )
    dataset._set_return_images(True)
    return dataset




def compose_robotwin_frames(images: dict[str, torch.Tensor], layout: str, camera_key: str) -> torch.Tensor:
    frames = []
    order = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    for key in order:
        if key not in images:
            raise KeyError(f"Expected camera key `{key}` in sample images, got {list(images.keys())}")
        frames.append(images[key].float() / 255.0)

    if layout == "robotwin":
        cam_top = transforms_F.resize(
            frames[0], size=[256, 320], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        cam_left = transforms_F.resize(
            frames[1], size=[128, 160], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        cam_right = transforms_F.resize(
            frames[2], size=[128, 160], interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )
        bottom = torch.cat([cam_left, cam_right], dim=-1)
        video = torch.cat([cam_top, bottom], dim=-2)
    elif layout == "horizontal":
        video = torch.cat(frames, dim=-1)
    elif layout == "vertical":
        video = torch.cat(frames, dim=-2)
    elif layout == "single":
        video = frames[order.index(camera_key)]
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    return video


def extract_frame_pair(sample: dict[str, Any], layout: str, camera_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    video = compose_robotwin_frames(sample["images"], layout=layout, camera_key=camera_key)
    if video.ndim != 4 or video.shape[0] < 2:
        raise ValueError(f"Expected [T,C,H,W] with T>=2, got {tuple(video.shape)}")
    frame_t = to_signed_range(video[0])
    frame_t1 = to_signed_range(video[1])
    return frame_t, frame_t1


def instantiate_deltatok_tokenizer(
    model_config_path: Path,
    device: str,
    torch_dtype: torch.dtype,
) -> torch.nn.Module:
    cfg = OmegaConf.load(str(model_config_path))
    tok_cfg = cfg.delta_tokenizer_config
    backbone = instantiate(tok_cfg.backbone).to(device=device, dtype=torch_dtype)
    tokenizer = DeltaTokEncoderTokenizer.from_pretrained(
        token_dim=int(cfg.delta_dit_config.delta_token_dim),
        backbone=backbone,
        num_hidden_layers=int(tok_cfg.get("num_hidden_layers", 12)),
        use_delta=bool(tok_cfg.get("use_delta", True)),
        use_decoder_head=tok_cfg.get("use_decoder_head", None),
        use_decoder_xy_embed=tok_cfg.get("use_decoder_xy_embed", None),
        layer_scale_init=float(tok_cfg.get("layer_scale_init", 1e-5)),
        use_qk_norm=bool(tok_cfg.get("use_qk_norm", True)),
        use_gated_attn=bool(tok_cfg.get("use_gated_attn", True)),
        use_swiglu=bool(tok_cfg.get("use_swiglu", True)),
        use_rope_aug=bool(tok_cfg.get("use_rope_aug", True)),
        pretrained_path=cfg.get("delta_tokenizer_pretrained_path", None),
        skip_load_from_pretrain=bool(cfg.get("skip_delta_tokenizer_load_from_pretrain", False)),
        device=device,
        torch_dtype=torch_dtype,
    )
    tokenizer.eval()
    return tokenizer


def instantiate_lam_tokenizer(
    model_config_path: Path,
    device: str,
    torch_dtype: torch.dtype,
) -> torch.nn.Module:
    cfg = OmegaConf.load(str(model_config_path))
    tok_cfg = cfg.delta_tokenizer_config
    tokenizer = LAMDeltaTokenizer.from_pretrained(
        token_dim=int(cfg.delta_dit_config.delta_token_dim),
        image_channels=int(tok_cfg.get("image_channels", 3)),
        lam_model_dim=int(tok_cfg.get("lam_model_dim", 1024)),
        lam_latent_dim=int(tok_cfg.get("lam_latent_dim", 32)),
        lam_patch_size=int(tok_cfg.get("lam_patch_size", 16)),
        lam_enc_blocks=int(tok_cfg.get("lam_enc_blocks", 24)),
        lam_dec_blocks=int(tok_cfg.get("lam_dec_blocks", 24)),
        lam_num_heads=int(tok_cfg.get("lam_num_heads", 16)),
        lam_dropout=float(tok_cfg.get("lam_dropout", 0.0)),
        pretrained_path=cfg.get("delta_tokenizer_pretrained_path", None),
        skip_load_from_pretrain=bool(cfg.get("skip_delta_tokenizer_load_from_pretrain", False)),
        device=device,
        torch_dtype=torch_dtype,
    )
    tokenizer.eval()
    return tokenizer


def build_tokenizer_runs(
    args: argparse.Namespace,
    torch_dtype: torch.dtype,
) -> list[TokenizerRunConfig]:
    runs: list[TokenizerRunConfig] = []
    if args.tokenizers in {"both", "deltatok"}:
        runs.append(
            TokenizerRunConfig(
                name="deltatok",
                model_config_path=to_project_path(args.deltatok_config),
                tokenizer=instantiate_deltatok_tokenizer(
                    model_config_path=to_project_path(args.deltatok_config),
                    device=args.device,
                    torch_dtype=torch_dtype,
                ),
                reconstruction_mode="tokenizer_decode_next",
            )
        )
    if args.tokenizers in {"both", "lam"}:
        runs.append(
            TokenizerRunConfig(
                name="lam",
                model_config_path=to_project_path(args.lam_config),
                tokenizer=instantiate_lam_tokenizer(
                    model_config_path=to_project_path(args.lam_config),
                    device=args.device,
                    torch_dtype=torch_dtype,
                ),
                reconstruction_mode="tokenizer_decode_next",
            )
        )
    return runs


@torch.no_grad()
def reconstruct_next_frame(
    tokenizer: torch.nn.Module,
    frame_t: torch.Tensor,
    frame_t1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = tokenizer.encode_pair(frame_t, frame_t1)
    try:
        recon = tokenizer.decode_next(frame_t, token)
    except NotImplementedError as err:
        raise NotImplementedError(
            f"{tokenizer.__class__.__name__}.decode_next() is not implemented yet. "
            "This evaluation script now requires the tokenizer to expose a working "
            "`decode_next(frame_t, delta_token)` interface for reconstruction."
        ) from err
    return token, recon


def select_valid_samples(
    dataset: BaseLerobotDataset,
    layout: str,
    camera_key: str,
    start_index: int,
    num_samples: int,
    max_scan: int,
) -> list[dict[str, Any]]:
    selected = []
    end_index = min(len(dataset), start_index + max_scan)
    for idx in range(start_index, end_index):
        sample = dataset[idx]
        image_is_pad = sample.get("image_is_pad")
        if image_is_pad is not None and bool(torch.as_tensor(image_is_pad).any().item()):
            continue
        try:
            frame_t, frame_t1 = extract_frame_pair(sample, layout=layout, camera_key=camera_key)
        except Exception:
            continue
        selected.append(
            {
                "dataset_index": int(idx),
                "sample": sample,
                "frame_t": frame_t,
                "frame_t1": frame_t1,
            }
        )
        if len(selected) >= num_samples:
            break
    return selected




def summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    if not metrics:
        return {"num_samples": 0, "avg_psnr": 0.0, "avg_ssim": 0.0}
    return {
        "num_samples": len(metrics),
        "avg_psnr": float(sum(item["psnr"] for item in metrics) / len(metrics)),
        "avg_ssim": float(sum(item["ssim"] for item in metrics) / len(metrics)),
    }


def main() -> None:
    args = parse_args()
    dataset_dir = to_project_path(args.dataset_dir)
    data_config_path = to_project_path(args.data_config)
    output_dir = (
        to_project_path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "runs" / "delta_tokenizer_eval" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = parse_dtype(args.dtype)
    dataset = build_dataset(
        dataset_dir=dataset_dir,
        data_config_path=data_config_path,
        frame_gap=args.frame_gap,
    )
    samples = select_valid_samples(
        dataset=dataset,
        layout=args.layout,
        camera_key=args.camera_key,
        start_index=args.start_index,
        num_samples=args.num_samples,
        max_scan=args.max_scan,
    )
    if not samples:
        raise RuntimeError("No valid samples found. Try increasing --max-scan or changing --start-index.")

    tokenizer_runs = build_tokenizer_runs(args=args, torch_dtype=torch_dtype)
    run_summaries = []

    for run in tokenizer_runs:
        tokenizer_dir = output_dir / run.name
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        per_sample_metrics = []
        print(
            f"[INFO] Evaluating tokenizer={run.name} "
            f"config={run.model_config_path} "
            f"reconstruction_mode={run.reconstruction_mode}"
        )

        for sample_info in samples:
            frame_t = sample_info["frame_t"].unsqueeze(0).to(device=args.device, dtype=torch_dtype)
            frame_t1 = sample_info["frame_t1"].unsqueeze(0).to(device=args.device, dtype=torch_dtype)
            with torch.no_grad():
                delta_token, recon_t1 = reconstruct_next_frame(run.tokenizer, frame_t, frame_t1)

            gt_next_unit = to_unit_range(frame_t1[0]).detach().cpu()
            recon_next_unit = to_unit_range(recon_t1[0]).detach().cpu()
            frame_t_cpu = frame_t[0].detach().cpu()
            psnr = video_psnr(recon_next_unit.unsqueeze(1), gt_next_unit.unsqueeze(1))
            ssim = video_ssim(recon_next_unit.unsqueeze(1), gt_next_unit.unsqueeze(1))

            panel = make_panel(frame_t_cpu, gt_next_unit, recon_next_unit)
            sample_name = f"idx_{sample_info['dataset_index']:06d}"
            panel_path = tokenizer_dir / f"{sample_name}.png"
            panel.save(panel_path)

            task_text = sample_info["sample"].get("task", "")
            if isinstance(task_text, (list, tuple)):
                task_text = task_text[0] if task_text else ""
            task_text = str(task_text)

            metric_record = {
                "dataset_index": int(sample_info["dataset_index"]),
                "task": task_text,
                "panel_path": str(panel_path.relative_to(output_dir)),
                "token_shape": list(delta_token.shape),
                "token_norm": float(delta_token.float().norm(dim=-1).mean().item()),
                "psnr": float(psnr),
                "ssim": float(ssim),
            }
            per_sample_metrics.append(metric_record)
            print(
                f"[sample] tokenizer={run.name} idx={metric_record['dataset_index']} "
                f"psnr={metric_record['psnr']:.4f} ssim={metric_record['ssim']:.4f} "
                f"token_shape={metric_record['token_shape']} token_norm={metric_record['token_norm']:.4f}"
            )

        summary = summarize_metrics(per_sample_metrics)
        summary_record = {
            "tokenizer": run.name,
            "model_config": str(run.model_config_path),
            "reconstruction_mode": run.reconstruction_mode,
            "dataset_format": "lerobot",
            "dataset_dir": str(dataset_dir),
            "layout": args.layout,
            "camera_key": args.camera_key,
            "frame_gap": args.frame_gap,
            "dtype": args.dtype,
            **summary,
        }
        run_summaries.append(summary_record)

        with open(tokenizer_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary_record,
                    "samples": per_sample_metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"[summary] tokenizer={run.name} samples={summary['num_samples']} "
            f"avg_psnr={summary['avg_psnr']:.4f} avg_ssim={summary['avg_ssim']:.4f}"
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"runs": run_summaries}, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Results written to {output_dir}")


if __name__ == "__main__":
    main()
