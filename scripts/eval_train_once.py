import hydra
import torch
import time
from omegaconf import DictConfig

from deltawam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
    _select_eval_manner,
    build_datasets,
)
from deltawam.trainer import Wan22Trainer
from deltawam.utils.config_resolvers import register_default_resolvers
from deltawam.utils.logging_config import setup_logging
from deltawam.utils import misc


register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging()
    misc.register_work_dir(cfg.output_dir)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    model = hydra.utils.instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)
    eval_manner = _select_eval_manner(cfg=cfg, model=model)

    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        eval_manner=eval_manner,
    )

    # Make eval sampling vary across runs when resuming from weights-only checkpoints.
    trainer.global_step = int(time.time() * 1000)

    # Run exactly one eval pass with the same logic as training-time evaluation.
    metrics = trainer.evaluate()
    trainer.accelerator.wait_for_everyone()
    if metrics is not None and trainer.accelerator.is_main_process:
        desc = trainer.eval_manner.format_eval_description(
            metrics=metrics,
            global_step=trainer.global_step,
            trainer=trainer,
        )
        print(desc)
        print(f"[eval_once] video_path={metrics.get('video_path')}")
        video_paths_by_mode = metrics.get("video_paths_by_mode", None)
        if isinstance(video_paths_by_mode, dict):
            for mode_name, path in sorted(video_paths_by_mode.items()):
                print(f"[eval_once] video_path_{mode_name}={path}")


if __name__ == "__main__":
    main()
