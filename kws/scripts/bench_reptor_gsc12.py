import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--data-root", default=None)
ap.add_argument("--manifest", default=None)
ap.add_argument("--output", required=True)
ap.add_argument("--warmup", type=int, default=100)
args = ap.parse_args()

repo = Path(args.repo).resolve()
sys.path.insert(0, str(repo))

from reptor import reptor9
from reptor.data import GSCDataset, LABEL_TO_INDEX

device = torch.device("cuda")
torch.set_num_threads(1)

# --------------------------------------------------
# checkpoint
# --------------------------------------------------
ck = torch.load(
    args.checkpoint,
    map_location="cpu",
    weights_only=False
)

# Recover paths used during training whenever possible
saved_args = ck.get("args", {})

if args.data_root is None:
    data_root = Path(saved_args["data_root"])
else:
    data_root = Path(args.data_root)

if args.manifest is None:
    manifest_dir = Path(saved_args.get("manifest_dir", "manifests"))
    if not manifest_dir.is_absolute():
        manifest_dir = repo / manifest_dir
    manifest = manifest_dir / "gsc12_test.csv"
else:
    manifest = Path(args.manifest)

print("data root :", data_root)
print("manifest  :", manifest)

assert data_root.exists(), data_root
assert manifest.exists(), manifest

# --------------------------------------------------
# dataset / frontend
# --------------------------------------------------
ds = GSCDataset(
    data_root=data_root,
    manifest_path=manifest,
    training=False,
    specaugment=False,
)

print("test samples:", len(ds))
assert len(ds) == 4888, f"Expected 4888, got {len(ds)}"

# --------------------------------------------------
# model
# --------------------------------------------------
model = reptor9(num_classes=12)
model.load_state_dict(ck["model_state"])

model.eval()
model.switch_to_deploy()
model.cuda()

params = sum(p.numel() for p in model.parameters())

# --------------------------------------------------
# Warm-up with REAL sample
# --------------------------------------------------
row0 = ds.rows[0]
wav0 = ds._load(row0)
feat0 = ds.mfcc(wav0).squeeze(0).unsqueeze(0)

assert feat0.shape == (1, 40, 101), feat0.shape

x0 = feat0.cuda()

with torch.inference_mode():
    for _ in range(args.warmup):
        _ = model(x0)

torch.cuda.synchronize()

# --------------------------------------------------
# benchmark whole real GSC12 test set
#
# IMPORTANT:
# disk WAV loading is intentionally OUTSIDE timing.
#
# total compute =
# in-memory waveform
# -> MFCC CPU
# -> CPU→GPU
# -> model
# -> logits ready
# --------------------------------------------------
feature_ms = []
model_ms = []
total_ms = []

correct = 0

torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():

    for i, row in enumerate(ds.rows):

        # Do NOT include disk I/O in compute latency
        wav = ds._load(row)

        # End-to-end compute timing begins with waveform already in RAM
        wall_start = time.perf_counter()

        # CPU frontend
        f0 = time.perf_counter()
        feat = ds.mfcc(wav).squeeze(0).unsqueeze(0)
        f1 = time.perf_counter()

        feature_ms.append((f1 - f0) * 1000)

        # H2D transfer
        x = feat.cuda(non_blocking=False)

        # GPU model-only timing
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)

        ev_start.record()
        logits = model(x)
        ev_end.record()

        ev_end.synchronize()

        model_ms.append(ev_start.elapsed_time(ev_end))

        wall_end = time.perf_counter()
        total_ms.append((wall_end - wall_start) * 1000)

        pred = int(logits.argmax(dim=1).item())
        target = LABEL_TO_INDEX[row["label"]]

        correct += int(pred == target)

        if (i + 1) % 500 == 0:
            print(
                f"{i+1}/{len(ds)} "
                f"running_acc={100*correct/(i+1):.3f}%"
            )

accuracy = correct / len(ds)

peak_vram_mb = (
    torch.cuda.max_memory_allocated() / 1024**2
)

def stats(x):
    x = np.asarray(x)
    return {
        "mean": float(x.mean()),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
    }

result = {
    "benchmark": "GSC12 real-test compute benchmark",
    "model": "RepTor-9",
    "num_test_samples": len(ds),

    "accuracy": accuracy,

    "params_deploy": params,
    "macs": 1511000,

    "input_audio": "16 kHz / 1 s",
    "frontend": "40 MFCC / 30 ms window / 10 ms hop",
    "feature_shape": [1, 40, 101],

    "timing_definition": {
        "disk_io": "excluded",
        "feature": "CPU MFCC",
        "model": "GPU forward only",
        "total_compute":
            "in-memory waveform -> MFCC -> H2D -> logits ready",
    },

    "feature_cpu_ms": stats(feature_ms),
    "model_gpu_ms": stats(model_ms),
    "total_compute_ms": stats(total_ms),

    "peak_vram_mb": float(peak_vram_mb),
}

print("\n===== FINAL RESULT =====")
print(json.dumps(result, indent=2))

out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))

print("\nsaved:", out)
