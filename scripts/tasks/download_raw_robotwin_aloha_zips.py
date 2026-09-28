#!/usr/bin/env python3
"""Download and extract all `aloha*.zip` files from RoboTwin 2.0 task folders.

Example:
    python3 scripts/tasks/download_raw_robotwin_aloha_zips.py \
        --output-dir ./data/robotwin2_official
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import zipfile

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO_ID = "TianxingChen/RoboTwin2.0"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_PATH_PREFIX = "dataset/"
DEFAULT_FILE_PREFIX = "aloha"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and extract all zip archives under dataset/* whose basename "
            "starts with the given prefix."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT),
        help=(
            "Hub endpoint to use. Defaults to HF_ENDPOINT if set, otherwise "
            f"{DEFAULT_ENDPOINT}."
        ),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision to download from. Default: main",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/robotwin2_official",
        help="Local output directory for downloaded archives and extracted folders.",
    )
    parser.add_argument(
        "--path-prefix",
        default=DEFAULT_PATH_PREFIX,
        help=f"Only keep repo files under this prefix. Default: {DEFAULT_PATH_PREFIX}",
    )
    parser.add_argument(
        "--file-prefix",
        default=DEFAULT_FILE_PREFIX,
        help=f"Only keep zip basenames starting with this prefix. Default: {DEFAULT_FILE_PREFIX}",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload archives even if they already exist locally.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Only download archives, do not extract them.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep zip archives after successful extraction. Default: delete them.",
    )
    return parser.parse_args()


def list_target_archives(
    *,
    repo_id: str,
    revision: str,
    endpoint: str,
    path_prefix: str,
    file_prefix: str,
) -> list[str]:
    api = HfApi(endpoint=endpoint)
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    targets = [
        repo_file
        for repo_file in repo_files
        if repo_file.startswith(path_prefix)
        and repo_file.endswith(".zip")
        and Path(repo_file).name.startswith(file_prefix)
    ]
    targets.sort()
    return targets


def download_archive(
    *,
    repo_id: str,
    repo_file: str,
    revision: str,
    endpoint: str,
    output_dir: Path,
    force_download: bool,
) -> Path:
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=repo_file,
        repo_type="dataset",
        revision=revision,
        endpoint=endpoint,
        local_dir=str(output_dir),
        force_download=force_download,
    )
    return Path(local_path)


def extract_archive(archive_path: Path) -> None:
    marker_path = archive_path.with_suffix(archive_path.suffix + ".extracted")
    if marker_path.exists():
        print(f"[skip-extract] {archive_path}")
        return

    print(f"[extract] {archive_path}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(archive_path.parent)
    marker_path.write_text("ok\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = list_target_archives(
        repo_id=args.repo_id,
        revision=args.revision,
        endpoint=args.endpoint,
        path_prefix=args.path_prefix,
        file_prefix=args.file_prefix,
    )
    if not targets:
        raise SystemExit(
            "No matching archives found. "
            f"repo_id={args.repo_id} path_prefix={args.path_prefix} file_prefix={args.file_prefix}"
        )

    print(
        f"Found {len(targets)} archives under {args.repo_id}@{args.revision} "
        f"via {args.endpoint}."
    )

    for repo_file in targets:
        print(f"[download] {repo_file}")
        archive_path = download_archive(
            repo_id=args.repo_id,
            repo_file=repo_file,
            revision=args.revision,
            endpoint=args.endpoint,
            output_dir=output_dir,
            force_download=args.force_download,
        )
        if not args.skip_extract:
            extract_archive(archive_path)
            if not args.keep_zip and archive_path.exists():
                print(f"[delete-zip] {archive_path}")
                archive_path.unlink()

    print("Done.")


if __name__ == "__main__":
    main()
