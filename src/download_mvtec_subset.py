#!/usr/bin/env python3
"""Download a category subset of MVTec AD from the public Voxel51 mirror.

The mirror preserves all 5,354 original samples and pixel masks, but exposes
them with a FiftyOne manifest. This downloader reads that manifest, selects the
configured categories, and stores only the necessary images and masks. MVTec
AD is CC BY-NC-SA 4.0 and is used here solely for academic research.
"""

from __future__ import annotations

import argparse
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


def load_categories(config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as handle:
        return list(yaml.safe_load(handle)["categories"])


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "freqpatch-lite/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def download_one(endpoint: str, relative_path: str, root: Path) -> tuple[str, str]:
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            with Image.open(destination) as image:
                image.verify()
            return relative_path, "skip"
        except Exception:
            destination.unlink(missing_ok=True)

    url = f"{endpoint}/datasets/{REPO_ID}/resolve/main/{relative_path}"
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
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    categories = set(load_categories(args.config))
    manifest_url = f"{args.endpoint}/datasets/{REPO_ID}/resolve/main/samples.json"
    manifest = fetch_json(manifest_url)
    selected = [sample for sample in manifest["samples"] if sample["category"]["label"] in categories]
    paths = {sample["filepath"] for sample in selected}
    paths.update(
        sample["defect_mask"]["mask_path"]
        for sample in selected
        if "defect_mask" in sample
    )
    local_manifest = {
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
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
        futures = [executor.submit(download_one, args.endpoint, path, args.output) for path in sorted(paths)]
        for future in as_completed(futures):
            _, status = future.result()
            completed += 1
            downloaded += status == "downloaded"
            if completed % 50 == 0 or completed == len(paths):
                print(f"[{completed}/{len(paths)}] downloaded={downloaded}", flush=True)

    print("Dataset subset verified successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

