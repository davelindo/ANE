#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p \
  "$repo_root/.local/build" \
  "$repo_root/.local/cache" \
  "$repo_root/.local/checkpoints" \
  "$repo_root/.local/models" \
  "$repo_root/.local/runs" \
  "$repo_root/.local/venvs"

echo "Prepared local workspace directories under $repo_root/.local"
