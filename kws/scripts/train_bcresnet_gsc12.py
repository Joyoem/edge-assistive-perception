#!/usr/bin/env python3
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# args
# --------------------------------------------------
ap = argparse.ArgumentParser()

ap.add_argument("--repo", required=True)
ap.add_argument("--data-root", required=True)
ap.add_argument("--manifest-dir", required=True)
ap.add_argument("--output-dir", required=True)

ap.add_argument("--tau", type=float, default=8.0)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--batch-size", type=int, default=100)

ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--weight-decay", type=float, default=1e-3)
ap.add_argument("--momentum", type=float, default=0.9)
ap.add_argument("--warmup-epochs", type=int, default=5)

ap.add_argument("--seed", type=int, default=1337)
ap.add_argument("--num-workers", type=int, default=0)

args = ap.parse_args()


repo = Path(args.repo).resolve()
sys.path.insert(0, str(repo))

from bcresnet import BCResNets
from utils import Preprocess


# --------------------------------------------------
# seed
# --------------------------------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda")
torch.set_num_threads(1)

data_root = Path(args.data_root)
manifest_dir = Path(args.manifest_dir)
output_dir = Path(args.output_dir)

run_dir = output_dir / f"bcresnet8_seed{args.seed}_ep{args.epochs}"
run_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# official BC-ResNet label mapping
# --------------------------------------------------
LABEL_TO_INDEX = {
    "_silence_": 0,
    "_unknown_": 1,
    "down": 2,
    "go": 3,
    "left": 4,
    "no": 5,
    "off": 6,
    "on": 7,
    "right": 8,
    "stop": 9,
    "up": 10,
    "yes": 11,
}


# --------------------------------------------------
# fixed manifest dataset
# --------------------------------------------------
class ManifestGSC12(Dataset):

    def __init__(self, manifest):
        with Path(manifest).open() as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        if row["kind"] == "silence":
            x = torch.zeros(1, 16000, dtype=torch.float32)

        else:
            wav_path = data_root / row["path"]

            audio, sr = sf.read(
                wav_path,
                dtype="float32"
            )

            if sr != 16000:
                raise RuntimeError(
                    f"{wav_path}: expected 16kHz, got {sr}"
                )

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

        y = LABEL_TO_INDEX[row["label"]]

        return x, y


train_ds = ManifestGSC12(
    manifest_dir / "gsc12_train.csv"
)

val_ds = ManifestGSC12(
    manifest_dir / "gsc12_val.csv"
)

test_ds = ManifestGSC12(
    manifest_dir / "gsc12_test.csv"
)


assert len(train_ds) == 36922
assert len(val_ds) == 4443
assert len(test_ds) == 4888


train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False,
)

val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

test_loader = DataLoader(
    test_ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)


print(
    f"train/val/test = "
    f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}"
)


# --------------------------------------------------
# official BC-ResNet-8 frontend
# --------------------------------------------------
noise_dir = str(
    data_root / "_background_noise_"
)

# tau=8 -> specaugment enabled
frequency_masking_para = {
    1: 0,
    1.5: 1,
    2: 3,
    3: 5,
    6: 7,
    8: 7,
}

preprocess_train = Preprocess(
    noise_dir,
    device,
    specaug=args.tau >= 1.5,
    frequency_masking_para=frequency_masking_para[args.tau],
)

preprocess_eval = Preprocess(
    noise_dir,
    device,
)

# Performance-only optimization:
# official Preprocess otherwise calls .to(device) for a randomly selected
# background-noise tensor inside the per-sample Python loop.
# Cache identical noise tensors on GPU once; augmentation semantics unchanged.
preprocess_train.background_noise = [
    x.to(device) for x in preprocess_train.background_noise
]
preprocess_eval.background_noise = preprocess_train.background_noise

print(
    f"cached {len(preprocess_train.background_noise)} "
    f"background-noise tensors on GPU",
    flush=True
)


# --------------------------------------------------
# model
#
# Official:
# BCResNets(int(tau * 8))
# tau=8 -> base_c=64
# --------------------------------------------------
model = BCResNets(
    int(args.tau * 8)
).to(device)

params = sum(
    p.numel()
    for p in model.parameters()
)

print("model: BC-ResNet-8")
print("params:", params)


# --------------------------------------------------
# optimizer: official recipe
# --------------------------------------------------
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.0,
    weight_decay=args.weight_decay,
    momentum=args.momentum,
)

steps_per_epoch = len(train_loader)

warmup_steps = (
    steps_per_epoch *
    args.warmup_epochs
)

total_steps = (
    steps_per_epoch *
    args.epochs
)


def get_lr(step):

    if step < warmup_steps:
        return (
            args.lr *
            step /
            warmup_steps
        )

    progress = (
        step - warmup_steps
    ) / max(
        1,
        total_steps - warmup_steps
    )

    return 0.5 * args.lr * (
        1.0 +
        np.cos(np.pi * progress)
    )


# --------------------------------------------------
# evaluation
# --------------------------------------------------
@torch.no_grad()
def evaluate(loader, validation=False):

    model.eval()

    correct = 0
    count = 0
    loss_sum = 0.0

    for inputs, labels in loader:

        inputs = inputs.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # Official validation behavior:
        # augment=True, is_train=False
        #
        # This mainly turns synthetic silence
        # into background-noise examples.
        #
        # Final test uses augment=False.
        inputs = preprocess_eval(
            inputs,
            labels=labels,
            is_train=False,
            augment=validation,
        )

        logits = model(inputs)

        loss = F.cross_entropy(
            logits,
            labels
        )

        batch = labels.size(0)

        loss_sum += (
            loss.item() * batch
        )

        correct += (
            logits.argmax(1) == labels
        ).sum().item()

        count += batch

    return (
        loss_sum / count,
        correct / count
    )


# --------------------------------------------------
# training
# --------------------------------------------------
best_val = -1.0
history = []
global_step = 0

start_all = time.perf_counter()


for epoch in range(args.epochs):

    model.train()

    epoch_start = time.perf_counter()

    train_correct = 0
    train_count = 0
    train_loss_sum = 0.0

    for inputs, labels in train_loader:

        global_step += 1

        lr = get_lr(global_step)

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        inputs = inputs.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        inputs = preprocess_train(
            inputs,
            labels,
            augment=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(inputs)

        loss = F.cross_entropy(
            logits,
            labels
        )

        loss.backward()
        optimizer.step()

        batch = labels.size(0)

        train_loss_sum += (
            loss.item() * batch
        )

        train_correct += (
            logits.argmax(1) == labels
        ).sum().item()

        train_count += batch


    train_loss = (
        train_loss_sum /
        train_count
    )

    train_acc = (
        train_correct /
        train_count
    )


    val_loss, val_acc = evaluate(
        val_loader,
        validation=True
    )

    seconds = (
        time.perf_counter() -
        epoch_start
    )

    rec = {
        "epoch": epoch + 1,
        "lr": lr,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "seconds": seconds,
    }

    history.append(rec)


    print(
        f"epoch "
        f"{epoch+1:03d}/{args.epochs:03d} "
        f"lr={lr:.6f} "
        f"train="
        f"{train_loss:.4f}/"
        f"{train_acc*100:.2f}% "
        f"val="
        f"{val_loss:.4f}/"
        f"{val_acc*100:.2f}% "
        f"time={seconds:.1f}s",
        flush=True
    )


    ckpt = {
        "model": "BC-ResNet-8",
        "tau": args.tau,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch + 1,
        "val_acc": val_acc,
        "params": params,
        "labels": LABEL_TO_INDEX,
        "args": vars(args),
    }


    torch.save(
        ckpt,
        run_dir / "last.pt"
    )


    if val_acc > best_val:

        best_val = val_acc

        torch.save(
            ckpt,
            run_dir / "best.pt"
        )


    (
        run_dir /
        "history.json"
    ).write_text(
        json.dumps(
            history,
            indent=2
        )
    )


# --------------------------------------------------
# final fixed GSC12 test
# --------------------------------------------------
best_ck = torch.load(
    run_dir / "best.pt",
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    best_ck["model_state"]
)

test_loss, test_acc = evaluate(
    test_loader,
    validation=False
)


result = {
    "model": "BC-ResNet-8",
    "seed": args.seed,
    "epochs": args.epochs,
    "best_epoch": best_ck["epoch"],
    "best_val_acc": best_val,
    "test_loss": test_loss,
    "test_acc": test_acc,
    "params": params,
    "train_samples": len(train_ds),
    "val_samples": len(val_ds),
    "test_samples": len(test_ds),
    "total_seconds":
        time.perf_counter() -
        start_all,
}


print("\n===== FINAL TEST =====")
print(
    json.dumps(
        result,
        indent=2
    )
)


(
    run_dir /
    "test_result.json"
).write_text(
    json.dumps(
        result,
        indent=2
    )
)

print(
    "\nrun:",
    run_dir
)
