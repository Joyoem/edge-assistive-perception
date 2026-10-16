#!/usr/bin/env python3
import argparse, csv, json, random, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--manifest-dir", required=True)
ap.add_argument("--output-dir", required=True)
ap.add_argument("--epochs", type=int, default=36)
ap.add_argument("--batch-size", type=int, default=100)
ap.add_argument("--num-workers", type=int, default=2)
ap.add_argument("--seed", type=int, default=1337)
args = ap.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

torch.set_num_threads(1)
device = torch.device("cuda")

data_root = Path(args.data_root)
manifest_dir = Path(args.manifest_dir)

run_dir = Path(args.output_dir) / f"dscnn_mlperfstyle_seed{args.seed}_ep{args.epochs}"
run_dir.mkdir(parents=True, exist_ok=True)

LABELS = (
    "yes","no","up","down","left","right",
    "on","off","stop","go","_unknown_","_silence_"
)
LABEL_TO_INDEX = {x:i for i,x in enumerate(LABELS)}


class GSC12(Dataset):
    def __init__(self, manifest):
        with open(manifest) as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]

        if row["kind"] == "silence":
            x = torch.zeros(1, 16000, dtype=torch.float32)
        else:
            a, sr = sf.read(data_root / row["path"], dtype="float32")
            assert sr == 16000

            x = torch.from_numpy(a)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            else:
                x = x.mean(dim=1).unsqueeze(0)

            if x.shape[-1] < 16000:
                x = F.pad(x, (0, 16000-x.shape[-1]))
            elif x.shape[-1] > 16000:
                x = x[..., :16000]

        return x, LABEL_TO_INDEX[row["label"]]


train_ds = GSC12(manifest_dir/"gsc12_train.csv")
val_ds   = GSC12(manifest_dir/"gsc12_val.csv")
test_ds  = GSC12(manifest_dir/"gsc12_test.csv")

assert (len(train_ds),len(val_ds),len(test_ds)) == (36922,4443,4888)

def loader(ds, shuffle):
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

train_loader = loader(train_ds, True)
val_loader   = loader(val_ds, False)
test_loader  = loader(test_ds, False)

print(f"train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")


# MLPerf Tiny default frontend dimensions:
# 1 sec / 16 kHz / 30 ms window / 20 ms stride / 10 MFCC
# -> 49 x 10
mfcc = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=10,
    melkwargs={
        "n_fft":512,
        "win_length":480,
        "hop_length":320,
        "n_mels":40,
        "center":False,
    },
).to(device)

# cache background noise on GPU once
noise_files = sorted((data_root/"_background_noise_").glob("*.wav"))
noise_bank = []

for p in noise_files:
    a, sr = sf.read(p, dtype="float32")
    assert sr == 16000
    x = torch.from_numpy(a)

    if x.ndim > 1:
        x = x.mean(dim=1)

    noise_bank.append(x.to(device))

print(f"cached {len(noise_bank)} background-noise tensors on GPU")


def augment_wave_batch(x, y):
    B = x.shape[0]

    for i in range(B):
        # ±100 ms time shift
        shift = random.randint(-1600, 1600)

        if shift > 0:
            x[i] = torch.cat([
                torch.zeros(1, shift, device=device),
                x[i, :, :-shift]
            ], dim=-1)

        elif shift < 0:
            s = -shift
            x[i] = torch.cat([
                x[i, :, s:],
                torch.zeros(1, s, device=device)
            ], dim=-1)

        # silence = noise; others get noise with p=.8
        is_silence = int(y[i]) == LABEL_TO_INDEX["_silence_"]

        if noise_bank and (is_silence or random.random() < 0.8):
            n = random.choice(noise_bank)

            if n.numel() >= 16000:
                start = random.randint(0, n.numel()-16000)
                n = n[start:start+16000]
            else:
                n = F.pad(n, (0, 16000-n.numel()))

            amp = random.uniform(0, 1.0 if is_silence else 0.1)

            if is_silence:
                x[i,0] = amp*n
            else:
                x[i,0] = (x[i,0] + amp*n).clamp(-1,1)

    return x


class DSBlock(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.dw = nn.Conv2d(c,c,3,padding=1,groups=c,bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.pw = nn.Conv2d(c,c,1,bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self,x):
        x = F.relu(self.bn1(self.dw(x)))
        x = F.relu(self.bn2(self.pw(x)))
        return x


class DSCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            1,64,
            kernel_size=(10,4),
            stride=(2,2),
            bias=False
        )
        self.bn = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout(0.2)

        self.blocks = nn.Sequential(
            DSBlock(), DSBlock(), DSBlock(), DSBlock()
        )

        self.drop2 = nn.Dropout(0.4)
        self.pool = nn.AvgPool2d((24,5))
        self.fc = nn.Linear(64,12)

    def forward(self,x):
        # Keras SAME padding for 49x10 -> 25x5
        x = F.pad(x, (1,1,4,5))
        x = F.relu(self.bn(self.conv(x)))
        x = self.drop1(x)
        x = self.blocks(x)
        x = self.drop2(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


model = DSCNN().to(device)

params = sum(p.numel() for p in model.parameters())
print("model: MLPerf-Tiny-style DS-CNN")
print("params:", params)

# sanity frontend
with torch.no_grad():
    z = torch.zeros(1,16000,device=device)
    feat = mfcc(z).unsqueeze(1)
    print("feature shape:", tuple(feat.shape))
    assert feat.shape[-2:] == (10,49)
    feat = feat.transpose(-1,-2)
    print("model input:", tuple(feat.shape))
    assert feat.shape == (1,1,49,10)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-4
)

def set_lr(epoch):
    if epoch < 12:
        lr = 5e-4
    elif epoch < 24:
        lr = 1e-4
    elif epoch < 36:
        lr = 2e-5
    else:
        lr = 1e-5

    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(dl):
    model.eval()

    correct = count = 0
    loss_sum = 0.0

    for wav,y in dl:
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        feat = mfcc(wav).squeeze(1).transpose(-1,-2).unsqueeze(1)
        logits = model(feat)

        loss = F.cross_entropy(logits,y)

        b = y.size(0)
        loss_sum += loss.item()*b
        correct += (logits.argmax(1)==y).sum().item()
        count += b

    return loss_sum/count, correct/count


best = -1
history = []
t_all = time.perf_counter()

for ep in range(args.epochs):
    lr = set_lr(ep)
    model.train()

    t0 = time.perf_counter()
    loss_sum = correct = count = 0

    for wav,y in train_loader:
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        wav = augment_wave_batch(wav,y)

        feat = mfcc(wav).squeeze(1).transpose(-1,-2).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        logits = model(feat)
        loss = F.cross_entropy(logits,y)

        loss.backward()
        optimizer.step()

        b = y.size(0)
        loss_sum += loss.item()*b
        correct += (logits.argmax(1)==y).sum().item()
        count += b

    tr_loss = loss_sum/count
    tr_acc = correct/count

    val_loss,val_acc = evaluate(val_loader)
    sec = time.perf_counter()-t0

    print(
        f"epoch {ep+1:03d}/{args.epochs:03d} "
        f"lr={lr:.6g} "
        f"train={tr_loss:.4f}/{tr_acc*100:.2f}% "
        f"val={val_loss:.4f}/{val_acc*100:.2f}% "
        f"time={sec:.1f}s",
        flush=True
    )

    rec = {
        "epoch":ep+1,
        "lr":lr,
        "train_loss":tr_loss,
        "train_acc":tr_acc,
        "val_loss":val_loss,
        "val_acc":val_acc,
        "seconds":sec
    }
    history.append(rec)

    ck = {
        "model_state":model.state_dict(),
        "epoch":ep+1,
        "val_acc":val_acc,
        "params":params,
        "labels":LABELS,
    }

    torch.save(ck,run_dir/"last.pt")

    if val_acc > best:
        best = val_acc
        torch.save(ck,run_dir/"best.pt")

    (run_dir/"history.json").write_text(json.dumps(history,indent=2))


best_ck = torch.load(run_dir/"best.pt",map_location=device,weights_only=False)
model.load_state_dict(best_ck["model_state"])

test_loss,test_acc = evaluate(test_loader)

result = {
    "model":"MLPerf-Tiny-style DS-CNN",
    "epochs":args.epochs,
    "best_epoch":best_ck["epoch"],
    "best_val_acc":best,
    "test_acc":test_acc,
    "test_loss":test_loss,
    "params":params,
    "train_samples":len(train_ds),
    "val_samples":len(val_ds),
    "test_samples":len(test_ds),
    "total_seconds":time.perf_counter()-t_all
}

print("\n===== FINAL TEST =====")
print(json.dumps(result,indent=2))

(run_dir/"test_result.json").write_text(json.dumps(result,indent=2))
