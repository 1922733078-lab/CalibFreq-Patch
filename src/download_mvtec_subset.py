#!/usr/bin/env python3
"""Download a pinned category subset from the public Voxel51 MVTec AD mirror.

The mirror is a convenience source rather than an independently verified copy
of the official MVTec archive.  This downloader pins one immutable repository
revision, records the source-manifest hash, and writes image-level SHA-256
checksums so later runs can verify the exact bytes used in this study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from PIL import Image


REPO_ID = "Voxel51/mvtec-ad"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_REVISION = "30a183a3b96e3aef953f230784b123b719b09d97"


def load_categories(config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as handle:
        return list(yaml.safe_load(handle)["categories"])


def fetch_json(url: str) -> tuple[dict, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "freqpatch-lite/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def download_one(
    endpoint: str, revision: str, relative_path: str, root: Path
) -> tuple[str, str]:
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            with Image.open(destination) as image:
                image.verify()
            return relative_path, "skip"
        except Exception:
            destination.unlink(missing_ok=True)

    url = f"{endpoint}/datasets/{REPO_ID}/resolve/{revision}/{relative_path}"
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "freqpatch-lite/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(destination)
            return relative_path, "downloaded"
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Unreachable download failure: {relative_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/mvtec"))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    categories = set(load_categories(args.config))
    manifest_url = f"{args.endpoint}/datasets/{REPO_ID}/resolve/{args.revision}/samples.json"
    manifest, manifest_sha256 = fetch_json(manifest_url)
    selected = [sample for sample in manifest["samples"] if sample["category"]["label"] in categories]
    paths = {sample["filepath"] for sample in selected}
    paths.update(
        sample["defect_mask"]["mask_path"]
        for sample in selected
        if "defect_mask" in sample
    )
    local_manifest = {
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
        "mirror_revision": args.revision,
        "mirror_manifest_url": manifest_url,
        "mirror_manifest_sha256": manifest_sha256,
        "verification_scope": (
            "Pinned mirror manifest and downloaded-byte checksums; no byte-level comparison "
            "with the separately licensed official MVTec archive was performed."
        ),
        "license": "CC BY-NC-SA 4.0",
        "samples": selected,
    }
    with (args.output / "samples.json").open("w", encoding="utf-8") as handle:
        json.dump(local_manifest, handle, ensure_ascii=False, indent=2)

    counts = {category: sum(s["category"]["label"] == category for s in selected) for category in sorted(categories)}
    print(f"Selected {len(selected)} samples and {len(paths)} files: {counts}", flush=True)
    completed = 0
    downloaded = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_one, args.endpoint, args.revision, path, args.output)
            for path in sorted(paths)
        ]
        for future in as_completed(futures):
            _, status = future.result()
            completed += 1
            downloaded += status == "downloaded"
            if completed % 50 == 0 or completed == len(paths):
                print(f"[{completed}/{len(paths)}] downloaded={downloaded}", flush=True)

    checksum_path = args.output / "SHA256SUMS.txt"
    with checksum_path.open("w", encoding="utf-8") as handle:
        for relative_path in sorted(paths):
            digest = hashlib.sha256((args.output / relative_path).read_bytes()).hexdigest()
            handle.write(f"{digest}  {relative_path}\n")
    print(
        f"Pinned mirror download completed; wrote {len(paths)} byte checksums to {checksum_path}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
