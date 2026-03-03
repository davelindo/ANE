#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

"$repo_root/scripts/bootstrap_local_dirs.sh" >/dev/null

move_if_needed() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "$src" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" ]]; then
    if cmp -s "$src" "$dst"; then
      rm -f "$src"
      echo "Removed duplicate: $src"
    else
      echo "Keeping existing destination (differs): $dst"
      echo "Source left in place: $src"
    fi
    return 0
  fi
  mv "$src" "$dst"
  echo "Moved: $src -> $dst"
}

move_if_needed "$repo_root/training/tinystories_data00.bin" "$repo_root/.local/cache/tinystories_data00.bin"
move_if_needed "$repo_root/training/ane_stories110M_ckpt.bin" "$repo_root/.local/checkpoints/ane_stories110M_ckpt.bin"
move_if_needed "$repo_root/.venv-nomic-ane" "$repo_root/.local/venvs/nomic-ane"

if [[ -d "$repo_root/runs" ]]; then
  mkdir -p "$repo_root/.local/runs"
  shopt -s nullglob
  for path in "$repo_root"/runs/*; do
    name="$(basename "$path")"
    move_if_needed "$path" "$repo_root/.local/runs/$name"
  done
  shopt -u nullglob
fi

echo "Legacy artifact migration complete."
