#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader


ap = argparse.ArgumentParser()

ap.add_argument(
    "--model",
    required=True,
    choices=["reptor9", "matchboxnet", "bcresnet8", "dscnn"]
)

ap.add_argument("--checkpoint", required=True)
ap.add_argument("--repo", default=None)

ap.add_argument("--data-root", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--output", required=True)

ap.add_argument("--batch-size", type=int, default=128)
ap.add_argument("--num-workers", type=int, default=2)

args = ap.parse_args()


DURATIONS_MS = [
    100, 150, 200, 250,
    300, 350, 400, 450,
    500, 550, 600, 650,
    700, 750, 800, 850,
    900, 950, 1000
]

device = torch.device("cuda")
torch.set_num_threads(1)

data_root = Path(args.data_root)
manifest = Path(args.manifest)


# ============================================================
# DATA
# ============================================================

with manifest.open() as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 4888


class GSC12(Dataset):

    def __len__(self):
        return len(rows)

    def __getitem__(self, idx):

        row = rows[idx]

        if row["kind"] == "silence":

            wav = torch.zeros(
                1, 16000,
                dtype=torch.float32
            )

        else:

            audio, sr = sf.read(
                data_root / row["path"],
                dtype="float32"
            )

            assert sr == 16000

            wav = torch.from_numpy(audio)

            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.mean(dim=1).unsqueeze(0)

            n = wav.shape[-1]

            if n < 16000:
                wav = F.pad(
                    wav,
                    (0, 16000-n)
                )

            elif n > 16000:
                wav = wav[..., :16000]

        return wav, row["label"], idx


loader = DataLoader(
    GSC12(),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=args.num_workers > 0
)


# ============================================================
# MODEL-SPECIFIC SETUP
# ============================================================

if args.model == "reptor9":

    sys.path.insert(0, args.repo)

    from reptor import reptor9

    model = reptor9(num_classes=12)

    ck = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        ck["model_state"]
    )

    model.eval()
    model.switch_to_deploy()
    model.cuda()

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={
            "n_fft":512,
            "win_length":480,
            "hop_length":160,
            "n_mels":40,
            "center":True,
        }
    ).cuda()

    label_to_idx = {
        "yes":0,
        "no":1,
        "up":2,
        "down":3,
        "left":4,
        "right":5,
        "on":6,
        "off":7,
        "stop":8,
        "go":9,
        "_unknown_":10,
        "_silence_":11
    }

    def forward_model(wav):

        feat = mfcc(wav).squeeze(1)

        return model(feat)


elif args.model == "matchboxnet":

    import nemo.collections.asr as nemo_asr

    model = (
        nemo_asr.models
        .EncDecClassificationModel
        .restore_from(
            restore_path=args.checkpoint,
            map_location="cpu"
        )
    )

    model.eval()
    model.cuda()

    labels = list(model.cfg.labels)

    label_to_idx = {
        "yes": labels.index("yes"),
        "no": labels.index("no"),
        "up": labels.index("up"),
        "down": labels.index("down"),
        "left": labels.index("left"),
        "right": labels.index("right"),
        "on": labels.index("on"),
        "off": labels.index("off"),
        "stop": labels.index("stop"),
        "go": labels.index("go"),
        "_unknown_": labels.index("unknown"),
        "_silence_": labels.index("silence"),
    }

    def forward_model(wav):

        # NeMo expects [B,T]
        signal = wav.squeeze(1)

        length = torch.full(
            (signal.shape[0],),
            signal.shape[-1],
            dtype=torch.long,
            device=device
        )

        processed, processed_len = (
            model.preprocessor(
                input_signal=signal,
                length=length
            )
        )

        if model.crop_or_pad is not None:

            processed, processed_len = (
                model.crop_or_pad(
                    input_signal=processed,
                    length=processed_len
                )
            )

        encoded, encoded_len = (
            model.encoder(
                audio_signal=processed,
                length=processed_len
            )
        )

        return model.decoder(
            encoder_output=encoded
        )


elif args.model == "bcresnet8":

    sys.path.insert(0, args.repo)

    from bcresnet import BCResNets
    from utils import Preprocess

    model = BCResNets(64)

    ck = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        ck["model_state"]
    )

    model.eval()
    model.cuda()

    frontend = Preprocess(
        None,
        device
    )

    label_to_idx = {
        "_silence_":0,
        "_unknown_":1,
        "down":2,
        "go":3,
        "left":4,
        "no":5,
        "off":6,
        "on":7,
        "right":8,
        "stop":9,
        "up":10,
        "yes":11
    }

    def forward_model(wav):

        dummy = torch.zeros(
            wav.shape[0],
            dtype=torch.long,
            device=device
        )

        feat = frontend(
            wav,
            labels=dummy,
            augment=False,
            is_train=False
        )

        return model(feat)


else:

    # Exact architecture from our DS-CNN training run

    class DSBlock(nn.Module):

        def __init__(self, c=64):

            super().__init__()

            self.dw = nn.Conv2d(
                c, c, 3,
                padding=1,
                groups=c,
                bias=False
            )

            self.bn1 = nn.BatchNorm2d(c)

            self.pw = nn.Conv2d(
                c, c, 1,
                bias=False
            )

            self.bn2 = nn.BatchNorm2d(c)

        def forward(self, x):

            x = F.relu(
                self.bn1(
                    self.dw(x)
                )
            )

            x = F.relu(
                self.bn2(
                    self.pw(x)
                )
            )

            return x


    class DSCNN(nn.Module):

        def __init__(self):

            super().__init__()

            self.conv = nn.Conv2d(
                1, 64,
                kernel_size=(10,4),
                stride=(2,2),
                bias=False
            )

            self.bn = nn.BatchNorm2d(64)

            self.drop1 = nn.Dropout(.2)

            self.blocks = nn.Sequential(
                DSBlock(),
                DSBlock(),
                DSBlock(),
                DSBlock()
            )

            self.drop2 = nn.Dropout(.4)

            self.pool = nn.AvgPool2d(
                (24,5)
            )

            self.fc = nn.Linear(
                64,12
            )

        def forward(self, x):

            x = F.pad(
                x,
                (1,1,4,5)
            )

            x = F.relu(
                self.bn(
                    self.conv(x)
                )
            )

            x = self.drop1(x)

            x = self.blocks(x)

            x = self.drop2(x)

            x = self.pool(x)

            return self.fc(
                x.flatten(1)
            )


    model = DSCNN()

    ck = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        ck["model_state"]
    )

    model.eval()
    model.cuda()

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=10,
        melkwargs={
            "n_fft":512,
            "win_length":480,
            "hop_length":320,
            "n_mels":40,
            "center":False
        }
    ).cuda()

    label_to_idx = {
        "yes":0,
        "no":1,
        "up":2,
        "down":3,
        "left":4,
        "right":5,
        "on":6,
        "off":7,
        "stop":8,
        "go":9,
        "_unknown_":10,
        "_silence_":11
    }

    def forward_model(wav):

        feat = (
            mfcc(wav)
            .squeeze(1)
            .transpose(-1,-2)
            .unsqueeze(1)
        )

        return model(feat)


# ============================================================
# BENCHMARK
# ============================================================

all_predictions = {
    str(t): np.zeros(
        len(rows),
        dtype=np.int16
    )
    for t in DURATIONS_MS
}

all_targets = np.zeros(
    len(rows),
    dtype=np.int16
)

curve = []


print(
    f"model={args.model} "
    f"samples={len(rows)}"
)


with torch.inference_mode():

    for duration_ms in DURATIONS_MS:

        cutoff = int(
            16000 *
            duration_ms /
            1000
        )

        correct = 0
        count = 0

        true_prob_sum = 0.0
        max_prob_sum = 0.0

        for wav, labels, indices in loader:

            wav = wav.to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # Prefix benchmark:
            # preserve original first t ms,
            # zero everything after it
            # ------------------------------------------------

            prefix = wav.clone()

            if cutoff < 16000:
                prefix[..., cutoff:] = 0.0

            logits = forward_model(
                prefix
            )

            probs = torch.softmax(
                logits,
                dim=1
            )

            targets = torch.tensor(
                [
                    label_to_idx[x]
                    for x in labels
                ],
                dtype=torch.long,
                device=device
            )

            pred = logits.argmax(1)

            b = len(labels)

            correct += (
                pred == targets
            ).sum().item()

            count += b

            true_prob_sum += (
                probs[
                    torch.arange(
                        b,
                        device=device
                    ),
                    targets
                ]
                .sum()
                .item()
            )

            max_prob_sum += (
                probs.max(1).values
                .sum()
                .item()
            )

            idx_np = indices.numpy()

            all_predictions[
                str(duration_ms)
            ][idx_np] = (
                pred.cpu().numpy()
            )

            if duration_ms == 1000:

                all_targets[
                    idx_np
                ] = targets.cpu().numpy()

        acc = correct / count

        true_conf = (
            true_prob_sum /
            count
        )

        max_conf = (
            max_prob_sum /
            count
        )

        rec = {
            "duration_ms":duration_ms,
            "accuracy":acc,
            "mean_true_class_probability":
                true_conf,
            "mean_max_probability":
                max_conf
        }

        curve.append(rec)

        print(
            f"{duration_ms:4d} ms "
            f"acc={acc*100:6.2f}% "
            f"true_p={true_conf:.4f} "
            f"max_p={max_conf:.4f}",
            flush=True
        )


# ============================================================
# SUMMARY METRICS
# ============================================================

full_acc = curve[-1]["accuracy"]


def first_threshold(ratio):

    threshold = ratio * full_acc

    for rec in curve:

        if rec["accuracy"] >= threshold:
            return rec["duration_ms"]

    return None


def first_sustained_threshold(ratio):

    threshold = ratio * full_acc

    for i, rec in enumerate(curve):

        if all(
            x["accuracy"] >= threshold
            for x in curve[i:]
        ):
            return rec["duration_ms"]

    return None


t90 = first_threshold(.90)
t95 = first_threshold(.95)

t90_sustained = (
    first_sustained_threshold(.90)
)

t95_sustained = (
    first_sustained_threshold(.95)
)


# ============================================================
# SAMPLE-LEVEL EARLIEST STABLE CORRECT DECISION
#
# earliest t where:
# prediction == ground truth
# AND every later prefix remains correct
#
# Only meaningful as clip-prefix discovery metric here.
# ============================================================

pred_matrix = np.stack(
    [
        all_predictions[str(t)]
        for t in DURATIONS_MS
    ],
    axis=0
)

stable_times = []

for sample_idx in range(
    len(rows)
):

    target = all_targets[
        sample_idx
    ]

    stable_t = None

    for i,t in enumerate(
        DURATIONS_MS
    ):

        later = pred_matrix[
            i:,
            sample_idx
        ]

        if np.all(
            later == target
        ):
            stable_t = t
            break

    if stable_t is not None:
        stable_times.append(
            stable_t
        )


stable_times_np = np.asarray(
    stable_times
)


stable_summary = {

    "num_samples_with_stable_correct_decision":
        len(stable_times),

    "fraction":
        len(stable_times) /
        len(rows),

    "median_ms":
        float(
            np.median(
                stable_times_np
            )
        )
        if len(stable_times)
        else None,

    "p75_ms":
        float(
            np.percentile(
                stable_times_np,75
            )
        )
        if len(stable_times)
        else None,

    "p90_ms":
        float(
            np.percentile(
                stable_times_np,90
            )
        )
        if len(stable_times)
        else None,
}


# ============================================================
# PER-CLASS CURVES
# ============================================================

idx_to_label = {
    v:k
    for k,v in label_to_idx.items()
}

per_class = {}

for cls_idx,label in idx_to_label.items():

    mask = (
        all_targets ==
        cls_idx
    )

    if not mask.any():
        continue

    per_class[label] = []

    for t in DURATIONS_MS:

        pred = all_predictions[
            str(t)
        ]

        acc = (
            pred[mask] ==
            cls_idx
        ).mean()

        per_class[label].append({
            "duration_ms":t,
            "accuracy":float(acc)
        })


result = {

    "benchmark":
        "GSC12 clip-prefix discovery benchmark",

    "warning":
        "Prefix duration is measured from clip start, not keyword onset.",

    "model":
        args.model,

    "num_samples":
        len(rows),

    "durations_ms":
        DURATIONS_MS,

    "full_clip_accuracy":
        full_acc,

    "T90_relative_to_full_ms":
        t90,

    "T95_relative_to_full_ms":
        t95,

    "T90_sustained_ms":
        t90_sustained,

    "T95_sustained_ms":
        t95_sustained,

    "curve":
        curve,

    "stable_correct_decision":
        stable_summary,

    "per_class":
        per_class
}


out = Path(args.output)

out.parent.mkdir(
    parents=True,
    exist_ok=True
)

out.write_text(
    json.dumps(
        result,
        indent=2
    )
)


# raw predictions for later analysis
np.savez_compressed(
    out.with_suffix(".npz"),

    durations_ms=np.asarray(
        DURATIONS_MS
    ),

    predictions=pred_matrix,

    targets=all_targets,

    sample_ids=np.asarray(
        [
            r["sample_id"]
            for r in rows
        ]
    )
)


print("\n===== SUMMARY =====")

print(
    json.dumps(
        {
            "model":args.model,
            "full_accuracy":
                full_acc,
            "T90_ms":
                t90,
            "T95_ms":
                t95,
            "T90_sustained_ms":
                t90_sustained,
            "T95_sustained_ms":
                t95_sustained,
            "stable_correct_decision":
                stable_summary,
        },
        indent=2
    )
)

print("\nsaved:", out)
