#!/usr/bin/env python3
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["bcresnet8","dscnn"], required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--bcresnet-repo")
ap.add_argument("--data-root", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--warmup", type=int, default=100)
args = ap.parse_args()

device = torch.device("cuda")
torch.set_num_threads(1)

root = Path(args.data_root)

with open(args.manifest) as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 4888


def load_wav(row):
    if row["kind"] == "silence":
        return torch.zeros(1,16000,dtype=torch.float32)

    a,sr = sf.read(root/row["path"],dtype="float32")
    assert sr == 16000

    x = torch.from_numpy(a)

    if x.ndim == 1:
        x=x.unsqueeze(0)
    else:
        x=x.mean(1).unsqueeze(0)

    if x.shape[-1] < 16000:
        x=F.pad(x,(0,16000-x.shape[-1]))
    elif x.shape[-1] > 16000:
        x=x[...,:16000]

    return x


# ==================================================
# BC-ResNet
# ==================================================
if args.model == "bcresnet8":

    sys.path.insert(0,args.bcresnet_repo)

    from bcresnet import BCResNets
    from utils import Preprocess

    model = BCResNets(64)

    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model.load_state_dict(ck["model_state"])

    model.eval().cuda()

    frontend_obj=Preprocess(
        None,
        device
    )

    label_map={
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

    def frontend(wav_gpu):
        dummy=torch.zeros(
            wav_gpu.shape[0],
            dtype=torch.long,
            device=device
        )

        return frontend_obj(
            wav_gpu,
            labels=dummy,
            augment=False,
            is_train=False
        )

    def network(feat):
        return model(feat)


# ==================================================
# DS-CNN
# ==================================================
else:

    class DSBlock(nn.Module):
        def __init__(self,c=64):
            super().__init__()
            self.dw=nn.Conv2d(
                c,c,3,padding=1,
                groups=c,bias=False
            )
            self.bn1=nn.BatchNorm2d(c)

            self.pw=nn.Conv2d(
                c,c,1,bias=False
            )
            self.bn2=nn.BatchNorm2d(c)

        def forward(self,x):
            x=F.relu(self.bn1(self.dw(x)))
            x=F.relu(self.bn2(self.pw(x)))
            return x


    class DSCNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv=nn.Conv2d(
                1,64,
                (10,4),
                stride=(2,2),
                bias=False
            )

            self.bn=nn.BatchNorm2d(64)
            self.drop1=nn.Dropout(.2)

            self.blocks=nn.Sequential(
                DSBlock(),
                DSBlock(),
                DSBlock(),
                DSBlock()
            )

            self.drop2=nn.Dropout(.4)
            self.pool=nn.AvgPool2d((24,5))
            self.fc=nn.Linear(64,12)

        def forward(self,x):
            x=F.pad(x,(1,1,4,5))
            x=F.relu(self.bn(self.conv(x)))
            x=self.drop1(x)
            x=self.blocks(x)
            x=self.drop2(x)
            x=self.pool(x)
            return self.fc(x.flatten(1))


    model=DSCNN()

    ck=torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        ck["model_state"]
    )

    model.eval().cuda()

    mfcc=torchaudio.transforms.MFCC(
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

    label_map={
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

    def frontend(wav_gpu):

        feat=mfcc(wav_gpu)

        return (
            feat
            .squeeze(1)
            .transpose(-1,-2)
            .unsqueeze(1)
        )

    def network(feat):
        return model(feat)


params=sum(
    p.numel()
    for p in model.parameters()
)

print("model:",args.model)
print("params:",params)
print("samples:",len(rows))


# ==================================================
# warmup
# ==================================================
wav0=load_wav(rows[0]).unsqueeze(0).cuda()

with torch.inference_mode():
    for _ in range(args.warmup):
        f=frontend(wav0)
        _=network(f)

torch.cuda.synchronize()


# ==================================================
# real GSC12 benchmark
# ==================================================
front_ms=[]
model_ms=[]
total_ms=[]

correct=0

torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():

    for i,row in enumerate(rows):

        # disk I/O excluded
        wav=load_wav(row).unsqueeze(0)

        total_start=time.perf_counter()

        wav_gpu=wav.cuda(
            non_blocking=False
        )

        # frontend
        f0=torch.cuda.Event(
            enable_timing=True
        )
        f1=torch.cuda.Event(
            enable_timing=True
        )

        f0.record()

        feat=frontend(wav_gpu)

        f1.record()


        # model
        m0=torch.cuda.Event(
            enable_timing=True
        )
        m1=torch.cuda.Event(
            enable_timing=True
        )

        m0.record()

        logits=network(feat)

        m1.record()

        torch.cuda.synchronize()

        total_end=time.perf_counter()

        front_ms.append(
            f0.elapsed_time(f1)
        )

        model_ms.append(
            m0.elapsed_time(m1)
        )

        total_ms.append(
            (total_end-total_start)*1000
        )

        pred=int(
            logits.argmax(1).item()
        )

        target=label_map[
            row["label"]
        ]

        correct += int(
            pred==target
        )

        if (i+1)%500==0:
            print(
                f"{i+1}/4888 "
                f"acc={100*correct/(i+1):.3f}%",
                flush=True
            )


def stats(x):
    x=np.asarray(x)

    return {
        "mean":float(x.mean()),
        "p50":float(np.percentile(x,50)),
        "p95":float(np.percentile(x,95)),
        "p99":float(np.percentile(x,99))
    }


result={
    "model":args.model,
    "samples":len(rows),
    "accuracy":correct/len(rows),
    "params":params,

    "frontend_gpu_ms":
        stats(front_ms),

    "model_gpu_ms":
        stats(model_ms),

    "total_compute_ms":
        stats(total_ms),

    "peak_vram_mb":
        torch.cuda.max_memory_allocated()/1024**2,

    "timing":{
        "batch":1,
        "precision":"FP32",
        "disk_io":"excluded",
        "total":
        "CPU waveform -> H2D -> frontend -> model -> logits"
    }
}

print("\n===== FINAL RESULT =====")
print(json.dumps(result,indent=2))

Path(args.output).write_text(
    json.dumps(result,indent=2)
)
