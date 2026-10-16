#!/usr/bin/env python3
"""Extract the final mIoU-like value from native evaluator logs.

Always inspect the raw logs once. Repository output formats change, and this
parser deliberately refuses to fabricate a number when no clear mIoU token is
present.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


MODELS = ("seaformer_tiny", "sctnet_s", "bisenetv2")


def extract(text: str):
    patterns = (
        r"mIoU\s*[:=|]\s*([0-9]+(?:\.[0-9]+)?)",
        r"mIoU[^\n]*?([0-9]+(?:\.[0-9]+)?)",
        r"mean\s+iou\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    )
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return matches[-1] if matches else "REVIEW_RAW_LOG"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.results_dir / "accuracy_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("model", "checkpoint_source", "ade20k_miou", "log")
        )
        writer.writeheader()
        for model in MODELS:
            log = args.results_dir / f"{model}_accuracy.log"
            text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
            writer.writerow(
                {
                    "model": model,
                    "checkpoint_source": (
                        "third_party" if model == "bisenetv2" else "official"
                    ),
                    "ade20k_miou": extract(text),
                    "log": str(log),
                }
            )
    print(output)


if __name__ == "__main__":
    main()

