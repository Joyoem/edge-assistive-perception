#!/usr/bin/env python3
"""Fetch repositories, released weights and optionally ADE20K.

The script is intentionally idempotent: existing repositories are not reset and
existing non-empty files are not downloaded again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import gdown


REPOSITORIES = {
    "SeaFormer": "https://github.com/fudan-zvg/SeaFormer.git",
    "SCTNet": "https://github.com/xzz777/SCTNet.git",
    "BiSeNet": "https://github.com/CoinCheung/BiSeNet.git",
}

DEFAULT_SCTNET_URL = (
    "https://drive.google.com/file/d/"
    "1U5OXyFDBYd3bZLKIAg64BFU-zPDbaO6d/view?usp=sharing"
)
DEFAULT_BISENET_URL = (
    "https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/"
    "model_final_v2_ade20k.pth"
)
ADE20K_URL = (
    "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
)


def run(*args: str, cwd: Optional[Path] = None) -> str:
    completed = subprocess.run(
        list(args), cwd=cwd, check=True, text=True, stdout=subprocess.PIPE
    )
    return completed.stdout.strip()


def clone_if_missing(name: str, url: str, repos_dir: Path) -> Path:
    destination = repos_dir / name
    if (destination / ".git").is_dir():
        print(f"[keep] {destination}")
        return destination
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite non-git path: {destination}")
    print(f"[clone] {url}")
    run("git", "clone", "--filter=blob:none", url, str(destination))
    return destination


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"[keep] {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if "drive.google.com" in url:
        print(f"[gdown] {destination.name}")
        result = gdown.download(url=url, output=str(partial), fuzzy=True, quiet=False)
        if not result:
            raise RuntimeError(f"Google Drive download failed: {url}")
    else:
        print(f"[download] {destination.name}")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target)
    partial.replace(destination)


def discover_seaformer_url(repo: Path) -> str:
    """Find the Tiny ADE20K weight URL in the checked-out repository docs."""
    markdown_files = sorted(repo.rglob("*.md"))
    url_pattern = re.compile(r"https?://[^\s)>]+")
    candidates: List[Tuple[int, str, Path]] = []
    for path in markdown_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        for index, line in enumerate(lines):
            context = " ".join(lines[max(0, index - 3) : index + 4]).lower()
            if "tiny" not in context or "ade" not in context:
                continue
            for url in url_pattern.findall(line):
                score = int("drive.google" in url) + int("onedrive" in url)
                candidates.append((score, url.rstrip(".,"), path))
    if not candidates:
        raise RuntimeError(
            "Could not discover the SeaFormer-Tiny ADE20K weight link. "
            "Open the official SeaFormer model-zoo page and set "
            "BENCH_SEAFORMER_CKPT_URL in benchmark.env."
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    url = candidates[0][1]
    print(f"[discover] SeaFormer checkpoint link from {candidates[0][2]}")
    return url


def maybe_download_ade20k(root: Path, enabled: bool, storage_root: Path) -> None:
    if (root / "images" / "validation").is_dir() and (
        root / "annotations" / "validation"
    ).is_dir():
        print(f"[keep] ADE20K at {root}")
        return
    if not enabled:
        print(f"[skip] ADE20K not found at {root}; automatic download is disabled")
        return
    archive = storage_root / "downloads" / "ADEChallengeData2016.zip"
    download(ADE20K_URL, archive)
    root.parent.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive}")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(root.parent)
    extracted = root.parent / "ADEChallengeData2016"
    if extracted.resolve() != root.resolve():
        if root.exists():
            raise RuntimeError(f"Target already exists and is not usable: {root}")
        extracted.rename(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--ade20k-root", type=Path, required=True)
    parser.add_argument("--download-ade20k", choices=("0", "1"), default="0")
    parser.add_argument("--seaformer-url", default="")
    parser.add_argument("--sctnet-url", default=DEFAULT_SCTNET_URL)
    parser.add_argument("--bisenet-url", default=DEFAULT_BISENET_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repos_dir = args.storage_root / "repos"
    weights_dir = args.storage_root / "checkpoints"
    results_dir = args.storage_root / "results"
    for directory in (repos_dir, weights_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    repos = {
        name: clone_if_missing(name, url, repos_dir)
        for name, url in REPOSITORIES.items()
    }

    seaformer_url = args.seaformer_url or discover_seaformer_url(repos["SeaFormer"])
    sctnet_url = args.sctnet_url or DEFAULT_SCTNET_URL
    bisenet_url = args.bisenet_url or DEFAULT_BISENET_URL

    weights = {
        "seaformer_tiny": weights_dir / "seaformer_tiny_ade20k.pth",
        "sctnet_s": weights_dir / "SCTNet-S-ADE20K.pth",
        "bisenetv2": weights_dir / "model_final_v2_ade20k.pth",
    }
    download(seaformer_url, weights["seaformer_tiny"])
    download(sctnet_url, weights["sctnet_s"])
    download(bisenet_url, weights["bisenetv2"])
    maybe_download_ade20k(
        args.ade20k_root, args.download_ade20k == "1", args.storage_root
    )

    provenance = {
        "repositories": {
            name: {
                "url": REPOSITORIES[name],
                "commit": run("git", "rev-parse", "HEAD", cwd=path),
            }
            for name, path in repos.items()
        },
        "checkpoints": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in weights.items()
        },
        "checkpoint_urls": {
            "seaformer_tiny": seaformer_url,
            "sctnet_s": sctnet_url,
            "bisenetv2": bisenet_url,
        },
    }
    output = results_dir / "provenance.json"
    output.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"[done] provenance: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
