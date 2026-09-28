#!/usr/bin/env bash
set -euo pipefail

cd data/robotwin2.0

# Run after downloading all robotwin2.0.tar.gz.part-* files.
cat robotwin2.0.tar.gz.part-* | tar -xzf -
