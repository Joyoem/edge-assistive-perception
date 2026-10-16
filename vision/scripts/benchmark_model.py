#!/usr/bin/env python3
"""One-device latency/complexity benchmark for the three released checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except Exception:
        return "unknown"


class MMSegForward(nn.Module):
    def __init__(self, model: nn.Module, height: int, width: int):
        super().__init__()
        self.model = model
        self.meta = [
            {
                "ori_shape": (height, width, 3),
                "img_shape": (height, width, 3),
                "pad_shape": (height, width, 3),
                "scale_factor": 1.0,
                "flip": False,
            }
        ]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_decode(image, self.meta)


class BiSeNetForward(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image)
        return output[0] if isinstance(output, (tuple, list)) else output


def remove_pretrained_initializers(value: Any) -> None:
    """Disable only external initialization; checkpoint loading happens later."""
    if isinstance(value, dict):
        for key in list(value):
            if key in {"pretrained", "init_cfg"}:
                value[key] = None
            else:
                remove_pretrained_initializers(value[key])
    elif isinstance(value, list):
        for item in value:
            remove_pretrained_initializers(item)


def load_mmseg(args: argparse.Namespace) -> nn.Module:
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmseg.models import build_segmentor

    cfg = Config.fromfile(str(args.config))
    cfg.model.train_cfg = None
    remove_pretrained_initializers(cfg.model)

    # SCTNet's Transformer/auxiliary components are training-only. They must not
    # inflate deployed parameter and latency measurements.
    if args.kind == "mmseg_sct":
        cfg.model.pop("auxiliary_head", None)

    model = build_segmentor(
        cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg", None)
    )
    load_checkpoint(model, str(args.checkpoint), map_location="cpu", strict=False)
    return MMSegForward(model, args.height, args.width)


def load_bisenet(args: argparse.Namespace) -> nn.Module:
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    from configs import set_cfg_from_file
    from lib.models import model_factory

    cfg = set_cfg_from_file(str(args.config))
    model = model_factory[cfg.model_type](cfg.n_cats, aux_mode="eval")
    state = torch.load(str(args.checkpoint), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return BiSeNetForward(model)


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def estimate_macs(model: nn.Module, image: torch.Tensor) -> Tuple[Any, str]:
    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(model, image)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        flops = float(analysis.total())
        unsupported = dict(analysis.unsupported_ops())
        # fvcore counts one fused multiply-add as one flop in common conv paths;
        # label the value MACs_estimate to avoid pretending cross-tool identity.
        note = "fvcore estimate"
        if unsupported:
            note += f"; unsupported_ops={unsupported}"
        return flops / 1e9, note
    except Exception as error:
        return None, f"MAC estimate failed: {type(error).__name__}: {error}"


def time_model(
    model: nn.Module,
    image: torch.Tensor,
    precision: str,
    warmup: int,
    runs: int,
) -> Dict[str, float]:
    use_amp = precision == "fp16"
    autocast = torch.cuda.amp.autocast if use_amp else nullcontext

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmup):
            with autocast():
                model(image)
        torch.cuda.synchronize()

        timings = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast():
                model(image)
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))

    return {
        "latency_median_ms": statistics.median(timings),
        "latency_p95_ms": percentile(timings, 95),
        "latency_mean_ms": statistics.fmean(timings),
        "fps_from_median": 1000.0 / statistics.median(timings),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
    }


def append_csv(path: Path, record: Dict[str, Any]) -> None:
    fields = (
        "model",
        "checkpoint_source",
        "precision",
        "input_shape",
        "params_m",
        "trainable_params_m",
        "macs_g_estimate",
        "macs_note",
        "latency_median_ms",
        "latency_p95_ms",
        "latency_mean_ms",
        "fps_from_median",
        "peak_allocated_mib",
        "gpu",
        "torch_version",
        "repo_commit",
        "checkpoint_sha256",
        "warmup",
        "runs",
        "timestamp_utc",
    )
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--kind", choices=("mmseg", "mmseg_sct", "bisenet"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-source", choices=("official", "third_party"), required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the controlled latency run")
    for path in (args.repo, args.config, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    torch.manual_seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    model = load_bisenet(args) if args.kind == "bisenet" else load_mmseg(args)
    model.eval().to(device)
    image = torch.randn(1, 3, args.height, args.width, device=device)

    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    macs_g, macs_note = estimate_macs(model, image)
    timing = time_model(model, image, args.precision, args.warmup, args.runs)

    record: Dict[str, Any] = {
        "model": args.model_name,
        "checkpoint_source": args.checkpoint_source,
        "precision": args.precision,
        "input_shape": f"1x3x{args.height}x{args.width}",
        "params_m": params / 1e6,
        "trainable_params_m": trainable / 1e6,
        "macs_g_estimate": macs_g,
        "macs_note": macs_note,
        **timing,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "repo_commit": git_commit(args.repo),
        "checkpoint_sha256": sha256(args.checkpoint),
        "warmup": args.warmup,
        "runs": args.runs,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model_name}_{args.precision}_{args.height}x{args.width}"
    json_path = args.results_dir / f"{stem}.json"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    append_csv(args.results_dir / "latency_summary.csv", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

