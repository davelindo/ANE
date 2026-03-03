"""Repository-local path helpers for ANE inference artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalPaths:
    repo_root: Path
    cache_dir: Path
    models_dir: Path
    runs_dir: Path


def infer_repo_root(from_file: Path) -> Path:
    """Infer repo root from a file under packages/ane-infer/src/ane/infer/..."""
    return from_file.resolve().parents[6]


def default_local_paths(from_file: Path) -> LocalPaths:
    repo_root = infer_repo_root(from_file)
    return LocalPaths(
        repo_root=repo_root,
        cache_dir=repo_root / ".local" / "cache",
        models_dir=repo_root / ".local" / "models",
        runs_dir=repo_root / ".local" / "runs",
    )
