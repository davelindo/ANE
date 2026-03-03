#!/usr/bin/env python3
"""ModernBERT ANE pipeline scaffold.

Current status: command surface only. Conversion/execution is intentionally not
implemented yet so we can add features incrementally with explicit validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    from config import ModernBertConfig
    from export import export_to_onnx
else:
    from .config import ModernBertConfig
    from .export import export_to_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="answerdotai/ModernBERT-base")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(".local/models/modernbert/base/fp16.onnx"),
    )
    parser.add_argument("--compute-unit", default="cpu_and_ne")
    parser.add_argument(
        "command",
        choices=("convert", "smoke-test"),
        help="Scaffolded command surface.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ModernBertConfig(
        model_id=args.model_id,
        revision=args.revision,
        seq_len=args.seq_len,
        artifact=args.artifact,
        compute_unit=args.compute_unit,
    )

    if args.command == "convert":
        export_to_onnx(cfg)
        return 0

    if args.command == "smoke-test":
        raise NotImplementedError(
            "ModernBERT smoke-test is not wired yet. Implement runtime backend selection and parity checks."
        )

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
