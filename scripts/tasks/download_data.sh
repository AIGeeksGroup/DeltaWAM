#!/usr/bin/env bash
set -euo pipefail

python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="yuanty/robotwin2.0-fastwam", repo_type="dataset", local_dir="./data/robotwin2.0")'
