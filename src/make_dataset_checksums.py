#!/usr/bin/env python3
"""Create a deterministic SHA-256 manifest for pinned MVTec mirror bytes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/mvtec"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest_path = args.manifest or args.data_root / "samples.json"
    output_path = args.output or args.data_root / "SHA256SUMS.txt"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    relative_paths: set[str] = set()
    for sample in payload["samples"]:
        relative_paths.add(sample["filepath"])
        mask = sample.get("defect_mask", {}).get("mask_path")
        if mask:
            relative_paths.add(mask)

    ordered = sorted(relative_paths)

    def hash_relative(relative: str) -> str:
        path = args.data_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        return f"{sha256(path)}  {relative}"

    lines = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for index, line in enumerate(executor.map(hash_relative, ordered), start=1):
            lines.append(line)
            if index % 500 == 0:
                print(f"Hashed {index}/{len(ordered)} files", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(lines)} checksums "
        f"({len(payload['samples'])} images plus {len(lines) - len(payload['samples'])} masks) "
        f"to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
