import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr


ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--data-root", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--warmup", type=int, default=100)
args = ap.parse_args()


device = torch.device("cuda")
torch.set_num_threads(1)

data_root = Path(args.data_root)
manifest = Path(args.manifest)

assert data_root.exists(), data_root
assert manifest.exists(), manifest


# --------------------------------------------------
# load checkpoint
# --------------------------------------------------
model = nemo_asr.models.EncDecClassificationModel.restore_from(
    restore_path=args.checkpoint,
    map_location="cpu",
)

model.eval()
model.cuda()

labels = list(model.cfg.labels)
label_to_idx = {x: i for i, x in enumerate(labels)}

print("labels:", labels)

expected_labels = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go", "unknown", "silence"
]

assert labels == expected_labels, labels

params = sum(p.numel() for p in model.parameters())


# --------------------------------------------------
# load manifest
# --------------------------------------------------
with manifest.open() as f:
    rows = list(csv.DictReader(f))

print("test samples:", len(rows))

assert len(rows) == 4888, len(rows)


# RepTor manifest naming -> NeMo naming
def normalize_label(label):
    if label == "_unknown_":
        return "unknown"
    if label == "_silence_":
        return "silence"
    return label


# --------------------------------------------------
# WAV loader
#
# Same benchmark audio policy as RepTor:
# 16 kHz
# exactly 1 second
# silence = zeros
# disk I/O excluded from timing
# --------------------------------------------------
def load_wav(row):
    if row["kind"] == "silence":
        return torch.zeros(1, 16000, dtype=torch.float32)

    audio, sr = sf.read(
        data_root / row["path"],
        dtype="float32"
    )

    assert sr == 16000, sr

    x = torch.from_numpy(audio)

    if x.ndim == 1:
        x = x.unsqueeze(0)
    else:
        x = x.mean(dim=1).unsqueeze(0)

    n = x.shape[-1]

    if n < 16000:
        x = F.pad(x, (0, 16000 - n))
    elif n > 16000:
        x = x[..., :16000]

    return x


# --------------------------------------------------
# explicit NeMo forward
#
# frontend:
# waveform -> preprocessor -> optional crop/pad
#
# network:
# encoder -> decoder
# --------------------------------------------------
def frontend(x, length):
    processed, processed_len = model.preprocessor(
        input_signal=x,
        length=length
    )

    if model.crop_or_pad is not None:
        processed, processed_len = model.crop_or_pad(
            input_signal=processed,
            length=processed_len
        )

    return processed, processed_len


def network(processed, processed_len):
    encoded, encoded_len = model.encoder(
        audio_signal=processed,
        length=processed_len
    )

    logits = model.decoder(
        encoder_output=encoded
    )

    return logits


# --------------------------------------------------
# sanity test
# --------------------------------------------------
wav0 = load_wav(rows[0])
x0 = wav0.cuda()
len0 = torch.tensor([x0.shape[-1]], device=device)

with torch.inference_mode():
    processed0, plen0 = frontend(x0, len0)
    logits0 = network(processed0, plen0)

print("wave shape:", tuple(x0.shape))
print("feature shape:", tuple(processed0.shape))
print("sanity prediction:", labels[int(logits0.argmax(dim=1).item())])


# --------------------------------------------------
# warmup
# --------------------------------------------------
with torch.inference_mode():
    for _ in range(args.warmup):
        processed0, plen0 = frontend(x0, len0)
        _ = network(processed0, plen0)

torch.cuda.synchronize()


# --------------------------------------------------
# benchmark
# --------------------------------------------------
frontend_ms = []
model_ms = []
total_ms = []

correct = 0

torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():

    for i, row in enumerate(rows):

        # Disk read is OUTSIDE benchmark timing
        wav = load_wav(row)

        # ------------------------------------------
        # total:
        # CPU in-memory waveform
        # -> H2D
        # -> official NeMo frontend
        # -> encoder
        # -> decoder
        # -> logits
        # ------------------------------------------
        total_start = time.perf_counter()

        x = wav.cuda(non_blocking=False)
        length = torch.tensor(
            [x.shape[-1]],
            dtype=torch.long,
            device=device
        )

        # frontend GPU timing
        ev_f0 = torch.cuda.Event(enable_timing=True)
        ev_f1 = torch.cuda.Event(enable_timing=True)

        ev_f0.record()

        processed, processed_len = frontend(
            x,
            length
        )

        ev_f1.record()

        # model/network GPU timing
        ev_m0 = torch.cuda.Event(enable_timing=True)
        ev_m1 = torch.cuda.Event(enable_timing=True)

        ev_m0.record()

        logits = network(
            processed,
            processed_len
        )

        ev_m1.record()

        # wait until result is really ready
        torch.cuda.synchronize()

        total_end = time.perf_counter()

        frontend_ms.append(
            ev_f0.elapsed_time(ev_f1)
        )

        model_ms.append(
            ev_m0.elapsed_time(ev_m1)
        )

        total_ms.append(
            (total_end - total_start) * 1000
        )

        pred = int(logits.argmax(dim=1).item())

        target_label = normalize_label(row["label"])
        target = label_to_idx[target_label]

        correct += int(pred == target)

        if (i + 1) % 500 == 0:
            print(
                f"{i+1}/{len(rows)} "
                f"running_acc={100*correct/(i+1):.3f}%"
            )


accuracy = correct / len(rows)

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


# Save useful official frontend config
preprocessor_cfg = OmegaConf.to_container(
    model.cfg.preprocessor,
    resolve=True
)


result = {
    "benchmark": "GSC12 real-test compute benchmark",

    "model": "MatchboxNet-3x1x64",

    "checkpoint": str(args.checkpoint),

    "num_test_samples": len(rows),

    "accuracy": accuracy,

    "labels": labels,

    "params": params,

    "input_audio": "16 kHz / 1 s",

    "timing_definition": {
        "batch_size": 1,
        "precision": "FP32",
        "disk_io": "excluded",

        "frontend":
            "official NeMo preprocessor + crop/pad",

        "model":
            "encoder + decoder GPU forward",

        "total_compute":
            "in-memory CPU waveform -> H2D -> frontend -> encoder -> decoder -> logits",
    },

    "frontend_gpu_ms": stats(frontend_ms),

    "model_gpu_ms": stats(model_ms),

    "total_compute_ms": stats(total_ms),

    "peak_vram_mb": float(peak_vram_mb),

    "preprocessor_config": preprocessor_cfg,
}


print("\n===== FINAL RESULT =====")
print(json.dumps(result, indent=2))

out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)

out.write_text(
    json.dumps(result, indent=2)
)

print("\nsaved:", out)
