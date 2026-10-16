# Edge Assistive Perception

Preliminary component benchmarks for a low-latency speech-to-intent + semantic-segmentation system, with a planned **4 KWS × 4 vision = 16-pipeline** benchmark.

**Current stage:** separately benchmarked KWS and vision models on RTX 3060. Combined 16-pipeline and Jetson measurements are **not yet available**. A Docker container on RTX 3060 is not a Jetson hardware emulator.

## KWS — Google Speech Commands v2, GSC12

Common evaluation test manifest: 4,888 samples (train 36,922; validation 4,443). Batch 1, FP32, RTX 3060; latencies are waveform-to-logits compute excluding WAV decoding and waiting for live audio.

| Model | Test accuracy | Parameters | p50 total | p95 total | Peak allocated GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| DS-CNN (MLPerf-Tiny-style reimplementation) | 93.82% | 23,180 | 1.326 ms | 3.431 ms | 9.40 MiB |
| RepTor-9 (deploy) | 97.20% | 62,076 | **1.082 ms** | **1.096 ms** | 9.40 MiB |
| MatchboxNet-3×1×64 | 98.16% | 74,892 | 2.365 ms | 3.014 ms | 9.77 MiB |
| BC-ResNet-8 | **98.65%** | 321,068 | 3.976 ms | 9.968 ms | 12.12 MiB |

**Caution:** Each model retains its own acoustic frontend and training/checkpoint recipe. RepTor uses a CPU feature stage while the other benchmark frontends are GPU-based. MatchboxNet uses a pretrained NeMo checkpoint, while the other models were trained/reproduced for this benchmark. These are operational system comparisons, not controlled identical-frontend/training ablations. No claim of real-time streaming detection delay is made.

## Vision — ADE20K checkpoint evaluations

| Model | Validation mIoU |
| --- | ---: |
| EfficientViT-B1 | 42.843% |
| SegFormer-B0 | 37.41% |
| SCTNet-S | 37.91% |
| SeaFormer-Tiny | ~36.48% |

This table reports model checkpoint evaluation only, not end-to-end pipeline speed or quality. Original logs and scripts are archived in `vision/` when available.

## Organization

- `vision/`: checkpoint-evaluation scripts/configs, original logs, results, patches.
- `kws/`: model training/evaluation and inference-latency scripts, original JSON/CSV/NPZ, GSC12 split manifests.
- `pipeline/`: planned multimodal integration and 16-combination benchmarking.
- `environments/`: available conda environment exports.
- `docs/benchmark_protocol.md`: evaluation definitions and limitations.

## Status

- Completed: KWS K1 (4 models); ADE20K vision checkpoint evaluation (4 models).
- Exploratory: GSC12 K2a *clip-prefix* / temporal occlusion (not keyword-onset aligned).
- Pending: joint KWS–vision pipeline evaluation, Jetson measurements, streaming/onset-aligned latency, and optional throat-microphone comparisons.

**Data/weights:** Datasets, audio, trained checkpoints and source-model repositories are not redistributed. Keep original upstream project attribution and license requirements. Exact source revisions should be recorded under `docs/`.
