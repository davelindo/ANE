#!/usr/bin/env python3
"""Extract pretokenized TinyStories data from zip to a local cache file."""

from __future__ import annotations

import argparse
import struct
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ZIP_PATH = Path.home() / "tiny_stories_data_pretokenized.zip"
DEFAULT_OUTPUT_PATH = REPO_ROOT / ".local" / "cache" / "tinystories_data00.bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--member",
        default="data00.bin",
        help="Zip entry to extract (default: data00.bin)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output_path.resolve()

    if output_path.exists() and not args.force:
        token_count = output_path.stat().st_size // 2
        print(
            f"{output_path} already exists ({token_count} tokens, {output_path.stat().st_size/1e6:.1f} MB)"
        )
        return 0

    if not args.zip_path.exists():
        raise SystemExit(f"Zip file not found: {args.zip_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {args.member} from {args.zip_path}...")

    with zipfile.ZipFile(args.zip_path, "r") as z:
        with z.open(args.member) as src, output_path.open("wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)

    token_count = output_path.stat().st_size // 2
    print(
        f"Written {output_path} ({token_count} tokens, {output_path.stat().st_size/1e6:.1f} MB)"
    )

    with output_path.open("rb") as f:
        first_ten = struct.unpack("<10H", f.read(20))
    print(f"First 10 tokens: {first_ten}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
