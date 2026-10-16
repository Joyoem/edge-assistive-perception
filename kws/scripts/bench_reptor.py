import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--output", required=True)
p.add_argument("--warmup", type=int, default=100)
p.add_argument("--runs", type=int, default=1000)
args = p.parse_args()

sys.path.insert(0, args.repo)
from reptor import reptor9

torch.set_num_threads(1)
device = torch.device("cuda")

# ---------------- model ----------------
ck = torch.load(
    args.checkpoint,
    map_location="cpu",
    weights_only=False
)

model = reptor9(num_classes=12)
model.load_state_dict(ck["model_state"])
model.eval()
model.switch_to_deploy()
model.cuda()

params = sum(p.numel() for p in model.parameters())

# Exact RepTor frontend used in reproduction
mfcc = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=40,
    melkwargs={
        "n_fft": 512,
        "win_length": 480,
        "hop_length": 160,
        "n_mels": 40,
        "center": True,
    },
)

wave = torch.randn(1, 16000) * 0.01

# ---------------- frontend CPU latency ----------------
for _ in range(50):
    _ = mfcc(wave)

front_ms = []
for _ in range(args.runs):
    t0 = time.perf_counter()
    feat = mfcc(wave)
    front_ms.append((time.perf_counter() - t0) * 1000)

# should be [1,40,101]
feat = mfcc(wave)
print("feature shape:", tuple(feat.shape))
assert tuple(feat.shape) == (1, 40, 101)

x = feat.cuda()

# ---------------- GPU model latency ----------------
with torch.inference_mode():
    for _ in range(args.warmup):
        _ = model(x)
    torch.cuda.synchronize()

    model_ms = []

    for _ in range(args.runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = model(x)
        end.record()

        end.synchronize()
        model_ms.append(start.elapsed_time(end))

# ---------------- peak VRAM ----------------
torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    _ = model(x)
    torch.cuda.synchronize()

peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

# ---------------- waveform -> logits wall latency ----------------
e2e_ms = []

with torch.inference_mode():
    for _ in range(50):
        f = mfcc(wave).cuda()
        _ = model(f)
        torch.cuda.synchronize()

    for _ in range(300):
        t0 = time.perf_counter()

        f = mfcc(wave).cuda()
        _ = model(f)

        torch.cuda.synchronize()
        e2e_ms.append((time.perf_counter() - t0) * 1000)

result = {
    "model": "RepTor-9",
    "accuracy_gsc12": 0.9719721767594108,
    "params_deploy": params,
    "macs_reported": 1511000,
    "frontend": "40 MFCC, 30ms window, 10ms hop",
    "input_shape": list(x.shape),

    "feature_cpu_p50_ms": float(np.percentile(front_ms, 50)),
    "feature_cpu_p95_ms": float(np.percentile(front_ms, 95)),

    "model_gpu_p50_ms": float(np.percentile(model_ms, 50)),
    "model_gpu_p95_ms": float(np.percentile(model_ms, 95)),

    "audio_to_logits_p50_ms": float(np.percentile(e2e_ms, 50)),
    "audio_to_logits_p95_ms": float(np.percentile(e2e_ms, 95)),

    "peak_vram_mb": float(peak_vram_mb),
}

print(json.dumps(result, indent=2))

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(result, indent=2))
