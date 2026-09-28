# DeltaWAM: Delta World Action Models for Bimanual Manipulation

This repo is the official implementation of:

> **DeltaWAM: Delta World Action Models for Bimanual Manipulation**
>
> Han Yan\*, Zishang Xiang\*, Haokai Jiang\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, [Qilin Wang](https://kelu007.github.io/), [Weiyu Guo](https://weiyuguo.com/), [Yandong Guo](https://scholar.google.com/citations?hl=en&user=fWDoWsQAAAAJ), [Boxin Shi](https://scholar.google.com/citations?hl=en&user=K1LjZxcAAAAJ), and [Hao Tang](https://ha0tang.github.io/)<sup>‡</sup>
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>‡</sup>Corresponding author.
>
> ### [Paper](https://arxiv.org/abs/2609.28811) | [Website](https://aigeeksgroup.github.io/DeltaWAM/) | [Model](https://huggingface.co/AIGeeksGroup/DeltaWAM)

DeltaWAM is a world action model for bimanual manipulation. This repository
provides the training and RoboTwin evaluation code used in the paper.

## Installation

We recommend Python 3.10 and CUDA 12.8.

```bash
conda create -n deltawam python=3.10 -y
conda activate deltawam

pip install --upgrade pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Data and Checkpoints

Download the preprocessed RoboTwin dataset:

```bash
bash scripts/tasks/download_data.sh
```

The extracted dataset is expected at:

```text
data/robotwin2.0/robotwin2.0/
├── data/
├── meta/
└── videos/
```

Download the released DeltaWAM+SDM checkpoints and their matching dataset
statistics from the [model page](https://huggingface.co/AIGeeksGroup/DeltaWAM):

```bash
hf download AIGeeksGroup/DeltaWAM --local-dir checkpoints/deltawam
```

The downloaded files are expected at:

```text
checkpoints/deltawam/
├── initial_five_task/
│   ├── deltawam.pt
│   ├── deltatok_encoder.ckpt
│   └── dataset_stats.json
└── supplemental_five_task/
    ├── deltawam.pt
    ├── deltatok_encoder.ckpt
    └── dataset_stats.json
```

The default DeltaWAM+SDM model configuration is
[`configs/model/deltakv.yaml`](configs/model/deltakv.yaml). It inherits its
training-time pretrained component settings from
[`configs/model/deltawam_wan_delta_light.yaml`](configs/model/deltawam_wan_delta_light.yaml),
which expects the following files:

```text
checkpoints/
├── ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
├── dinov3-vitb16-pretrain-lvd1689m/
└── rgb-head-imagenet/pytorch_model.bin
```

Download DINOv3 and the RGB reconstruction head before training or evaluation.
DINOv3 is a gated model, so first accept its license on the
[model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
and log in with `hf auth login`.

```bash
hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir checkpoints/dinov3-vitb16-pretrain-lvd1689m

hf download Amazon-FAR/rgb-head-imagenet \
  pytorch_model.bin \
  --local-dir checkpoints/rgb-head-imagenet
```

Update the corresponding config values if these components are stored
elsewhere. The ActionDiT initialization is generated in the next section.

## Training

The DeltaWAM+SDM training launcher uses
[`sub_robotwin_deltakv_3cam_384_1e-4`](configs/task/sub_robotwin_deltakv_3cam_384_1e-4.yaml)
and accepts standard Hydra overrides.

Prepare the initialized action backbone:

```bash
bash scripts/tasks/pregenerate_actiondit.sh
```

Specify the training tasks when precomputing text embeddings:

```bash
bash scripts/tasks/precompute_text_embeds.sh \
  'robotwin_subset.train_task_names=[adjust_bottle,beat_block_hammer,blocks_ranking_size,click_alarmclock,click_bell]'
```

Use the same task list for training. For one GPU:

```bash
bash scripts/tasks/train_deltakv_robotwin.sh \
  model.delta_tokenizer_pretrained_path=checkpoints/deltawam/initial_five_task/deltatok_encoder.ckpt \
  'robotwin_subset.train_task_names=[adjust_bottle,beat_block_hammer,blocks_ranking_size,click_alarmclock,click_bell]'
```

For multi-GPU training, set the number of local processes:

```bash
TRAIN_NPROC_PER_NODE=8 bash scripts/tasks/train_deltakv_robotwin.sh \
  model.delta_tokenizer_pretrained_path=checkpoints/deltawam/initial_five_task/deltatok_encoder.ckpt \
  'robotwin_subset.train_task_names=[adjust_bottle,beat_block_hammer,blocks_ranking_size,click_alarmclock,click_bell]'
```

Replace the names in `robotwin_subset.train_task_names` to train on a different
set of tasks. The text-embedding and training commands must use the same list.

The released DeltaTok encoder checkpoints were fine-tuned separately on the
two corresponding RoboTwin subsets. When training DeltaWAM+SDM on another
subset, we recommend first fine-tuning the DeltaTok encoder on that subset by
following the official [DeltaTok codebase](https://github.com/amazon-far/deltatok),
then passing the resulting checkpoint through
`model.delta_tokenizer_pretrained_path`.

Hydra overrides can be appended directly. For example, to resume training:

```bash
TRAIN_NPROC_PER_NODE=8 bash scripts/tasks/train_deltakv_robotwin.sh \
  resume=/path/to/checkpoint.pt \
  model.delta_tokenizer_pretrained_path=checkpoints/deltawam/initial_five_task/deltatok_encoder.ckpt \
  'robotwin_subset.train_task_names=[adjust_bottle,beat_block_hammer,blocks_ranking_size,click_alarmclock,click_bell]'
```

Training outputs, logs, dataset statistics, and checkpoints are written to
`runs/<task-name>/<run-id>/`. In particular, the normalization statistics are
saved as `runs/<task-name>/<run-id>/dataset_stats.json`.

## RoboTwin Evaluation

The RoboTwin source used by the evaluation wrapper is included under
`third_party/RoboTwin`. Complete the simulator installation and asset setup by
following the [official RoboTwin instructions](https://github.com/RoboTwin-Platform/RoboTwin).

Specify the evaluation tasks with `EVALUATION.task_names`. Use the checkpoint
and dataset statistics that correspond to the selected tasks:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  ckpt=checkpoints/deltawam/initial_five_task/deltawam.pt \
  EVALUATION.dataset_stats_path=checkpoints/deltawam/initial_five_task/dataset_stats.json \
  'EVALUATION.task_names=[adjust_bottle,beat_block_hammer,blocks_ranking_size,click_alarmclock,click_bell]' \
  'MULTIRUN.gpu_ids=[0]'
```

```bash
python experiments/robotwin/run_robotwin_manager.py \
  ckpt=checkpoints/deltawam/supplemental_five_task/deltawam.pt \
  EVALUATION.dataset_stats_path=checkpoints/deltawam/supplemental_five_task/dataset_stats.json \
  'EVALUATION.task_names=[blocks_ranking_rgb,dump_bin_bigbin,grab_roller,handover_block,handover_mic]' \
  'MULTIRUN.gpu_ids=[0]'
```

For multi-GPU evaluation, replace `MULTIRUN.gpu_ids` with a list such as
`[0,1,2,3]` and optionally set `MULTIRUN.max_tasks_per_gpu=2`.

Evaluation logs, videos, and summaries are saved under
`evaluate_results/robotwin/`.

## Acknowledgements

This repository builds on [FastWAM](https://github.com/yuantianyuan01/FastWAM),
[DeltaTok](https://github.com/amazon-far/deltatok), and
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin). We thank their
authors for releasing their work.

## Citation

If you find DeltaWAM useful, please cite:

```bibtex
@article{yan2026deltawam,
  title={DeltaWAM: Delta World Action Models for Bimanual Manipulation},
  author={Yan, Han and Xiang, Zishang and Jiang, Haokai and Zhang, Zeyu and Wang, Qilin and Guo, Weiyu and Guo, Yandong and Shi, Boxin and Tang, Hao},
  journal={arXiv preprint arXiv:2609.28811},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE). Third-party code
and assets remain subject to their original licenses.
