"""Configuration for ModernBERT ANE conversion/inference experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModernBertConfig:
    model_id: str
    revision: str | None
    seq_len: int
    artifact: Path
    compute_unit: str
