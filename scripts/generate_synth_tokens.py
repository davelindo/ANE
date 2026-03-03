#!/usr/bin/env python3
"""Generate synthetic uint16 token data for quick ANE smoke/e2e runs."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output .bin file")
    parser.add_argument(
        "--count",
        type=int,
        default=8192,
        help="Number of synthetic tokens to generate",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Upper bound for token IDs (exclusive)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be > 0")
    if args.vocab_size <= 0:
        raise SystemExit("--vocab-size must be > 0")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        for i in range(args.count):
            f.write(struct.pack("<H", i % args.vocab_size))

    print(f"Wrote {args.output} with {args.count} synthetic tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
