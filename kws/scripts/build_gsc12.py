#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, random
from collections import Counter
from pathlib import Path

TARGET_WORDS = ("yes","no","up","down","left","right","on","off","stop","go")
PAPER_TOTALS = {"train":36922,"val":4443,"test":4888}
EXPECTED_WANTED = {"train":30769,"val":3703,"test":4074}
EXPECTED_RAW_TOTAL = 105829
SEED_OFFSET = {"train":0,"val":1,"test":2}


def _read_lines(path: Path):
    return {x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}


def collect(data_root: Path):
    val = _read_lines(data_root / "validation_list.txt")
    test = _read_lines(data_root / "testing_list.txt")
    out = {"train":[],"val":[],"test":[]}
    for wav in sorted(data_root.glob("*/*.wav")):
        label = wav.parent.name
        if label == "_background_noise_":
            continue
        rel = wav.relative_to(data_root).as_posix()
        split = "val" if rel in val else "test" if rel in test else "train"
        out[split].append((rel,label))
    return out


def build_rows(items, split: str, seed: int):
    wanted = sorted((p,l) for p,l in items if l in TARGET_WORDS)
    unknown_pool = sorted((p,l) for p,l in items if l not in TARGET_WORDS)
    if len(wanted) != EXPECTED_WANTED[split]:
        raise RuntimeError(f"{split}: expected {EXPECTED_WANTED[split]} wanted samples, got {len(wanted)}")
    remainder = PAPER_TOTALS[split] - len(wanted)
    n_unknown = (remainder + 1)//2
    n_silence = remainder//2
    rng = random.Random(seed + SEED_OFFSET[split])
    sampled_unknown = sorted(rng.sample(unknown_pool, n_unknown))
    rows = []
    for p,l in wanted:
        rows.append({"path":p,"label":l,"source_label":l,"kind":"keyword","sample_id":p})
    for p,l in sampled_unknown:
        rows.append({"path":p,"label":"_unknown_","source_label":l,"kind":"unknown","sample_id":p})
    for i in range(n_silence):
        rows.append({"path":"","label":"_silence_","source_label":"_silence_","kind":"silence","sample_id":f"silence_{split}_{i:05d}"})
    assert len(rows) == PAPER_TOTALS[split]
    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path","label","source_label","kind","sample_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("manifests"))
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    root = args.data_root.expanduser().resolve()
    for p in [root/"validation_list.txt", root/"testing_list.txt", root/"_background_noise_"]:
        if not p.exists(): raise FileNotFoundError(p)
    splits = collect(root)
    raw_total = sum(map(len, splits.values()))
    if raw_total != EXPECTED_RAW_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_RAW_TOTAL} raw utterances, got {raw_total}")
    print(f"GSC root: {root}")
    print(f"Raw utterances: {raw_total}")
    for split in ("train","val","test"):
        rows = build_rows(splits[split], split, args.seed)
        path = args.output_dir / f"gsc12_{split}.csv"
        write_csv(rows, path)
        kinds = Counter(r["kind"] for r in rows)
        print(f"\n{split.upper()}: total={len(rows)} keyword={kinds['keyword']} unknown={kinds['unknown']} silence={kinds['silence']}")
        print(path)
    print("\nPASS: manifests match RepTor paper split totals exactly.")
    print("NOTE: unknown seed/rounding are explicit reproduction choices because the paper does not specify them.")

if __name__ == "__main__": main()
