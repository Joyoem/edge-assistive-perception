# KWS benchmark

- Dataset: Google Speech Commands v2 (GSC12 setup), with common manifests under `splits/`.
- Evaluated classes: 10 target command words + unknown + silence.
- Models: RepTor-9, MatchboxNet-3×1×64, BC-ResNet-8, and a MLPerf-Tiny-*style* PyTorch DS-CNN reimplementation.
- K1 metrics: full-clip accuracy, parameter count, batch-1 FP32 waveform-to-logits latency (p50/p95/p99) and allocated GPU memory on RTX 3060. Disk WAV reading is not timed.
- K2a: exploration of prefix preservation followed by zero padding; duration is relative to clip start, **not keyword onset**. Offline stable decisions are **not online trigger latencies**.
- Results: original model-specific JSON and prefix-prediction NPZ are retained under `results/` where available.

Check `docs/benchmark_protocol.md` for interpretation limits. Model-specific frontends/training protocols differ; avoid presenting all four as fully controlled identical-training comparisons.
