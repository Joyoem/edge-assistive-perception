# Preliminary benchmark protocol (2026-10-16)

## KWS K1: full-clip classification / compute

- Dataset: Google Speech Commands v2, GSC12 labels (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, `unknown`, `silence`). Common recorded train/val/test manifest counts: 36,922 / 4,443 / 4,888.
- Main metric: top-1 classification accuracy on the fixed 4,888-example test manifest.
- Compute: NVIDIA RTX 3060, FP32, batch 1; include waveform transfer to GPU, acoustic frontend, and network forward; exclude WAV disk decoding, audio capture/buffering and downstream segmentation.
- Report total p50, p95, p99, frontend/model stage timings and allocated GPU memory. These numbers are not system-level peak RAM/Jetson RAM.
- Source recipes differ. MatchboxNet uses an NVIDIA pretrained NeMo checkpoint; other KWS models are local reproductions/retrainings. Architectural/front-end comparisons are informative but not identical-protocol ablations.
- The MLPerf-Tiny-style DS-CNN is a *reimplementation*, not an identical reproduction of TensorFlow's official microfrontend.
- Do not infer early speech-decision latency from ≈1–4ms network compute alone.

## KWS K2a: temporal occlusion

For each nominal 1s clip, preserve its first t ms and replace the remainder with zeros; evaluate the original full-clip KWS classifier. The measured curves characterize robustness to removing later acoustic input.

- Intended: accuracy-vs-clip-prefix duration, per-class recall, model confidence diagnostics.
- Exploratory only: relative-to-full-accuracy T90/T95, offline earliest stable correct prediction.
- Not established: onset/offset-aligned true earliest decision, streaming false accepts per hour, or application response latency.
- Verify accuracy at 1000ms reproduces K1 before interpreting curves. Validation data must be used to select online thresholds.

## Vision V1: checkpoints

ADE20K validation semantic segmentation; report class-count, resolution, preprocessing, checkpoint provenance, mIoU, and original evaluation commands alongside logs. The table in the project README is a checkpoint result, not a vision latency benchmark.

## Planned pipeline P1

16 combinations across four KWS and four segmentation models. Define which stage is triggered by speech (sequential or continuous vision). Measure end-to-end event timing using instrumented timestamps, segmentation FPS, peak shared-process/device memory, confidence/false intent effects and functional mask output. RTX 3060 integration is not a substitute for a real Jetson 8GB benchmark. Command-to-mask semantics require a documented proxy mapping or object-word speech dataset.

## Preservation checklist

Archive evaluation/training scripts, SLURM scripts, original text logs, CSV/JSON/NPZ, split manifests, environment exports, and source repository revisions. Do not upload raw speech/audio, ADE20K image files, checkpoints, credentials, or cached model weights. Check copied artifacts against source before committing. Record any missing artifacts explicitly.
