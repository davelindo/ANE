"""ModernBERT export helpers.

This module intentionally starts as scaffolding. The implementation work here is
separate from the stabilized Nomic path and should land incrementally.
"""

from __future__ import annotations

try:
    from .config import ModernBertConfig
except ImportError:  # pragma: no cover - direct script/module execution fallback
    from config import ModernBertConfig


def export_to_onnx(config: ModernBertConfig) -> None:
    raise NotImplementedError(
        "ModernBERT export is not wired yet. Start by implementing a fixed-shape ONNX export path."
    )
