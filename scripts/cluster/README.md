# Cluster/Local Task Runner

This project now splits execution into 2 layers:

1. `scripts/tasks/*.sh`
   - Task-only scripts (no scheduler directives, no fixed server paths).
2. `scripts/cluster/*.sh`
   - Shared environment bootstrap + scheduler submit entrypoints.

## Run on a normal GPU machine (no scheduler)

```bash
bash scripts/tasks/precompute_text_embeds.sh
bash scripts/tasks/train_deltawam_robotwin.sh
bash scripts/tasks/run_robotwin.sh
```

## Run on Slurm (single entrypoint)

1. Edit config block in `scripts/cluster/submit_slurm.sh`.
2. Submit:

```bash
bash scripts/cluster/submit_slurm.sh
```

If your cluster uses different conda locations/env names, override before submit:

```bash
CONDA_ENV_NAME=deltawam_a \
CONDA_SH=/path/to/miniconda/etc/profile.d/conda.sh \
bash scripts/cluster/submit_slurm.sh
```

## Run on PBS (single entrypoint)

1. Edit config block in `scripts/cluster/submit_pbs.sh`.
2. Submit:

```bash
bash scripts/cluster/submit_pbs.sh
```

If your cluster uses different conda locations/env names, override before submit:

```bash
CONDA_ENV_NAME=deltawam_b \
CONDA_SH=/another/path/conda.sh \
bash scripts/cluster/submit_pbs.sh
```

For PBS, if you want current shell env vars passed into job, add `-V` in
`PBS_EXTRA_ARGS` in `scripts/cluster/submit_pbs.sh`.

## Environment variables for portability

- `PROJECT_ROOT` (optional): repository root. Auto-detected by default.
- `RUNS_DIR` (optional): log/output directory. Default: `${PROJECT_ROOT}/runs`.
- `CONDA_SH` (optional): conda init script path.
- `CONDA_ENV_NAME` (optional): conda env name. Default: `deltawam`.
- `SKIP_CONDA_ACTIVATE=1` (optional): skip conda activation.

This keeps task scripts portable across different servers by changing only scheduler config and env vars.
