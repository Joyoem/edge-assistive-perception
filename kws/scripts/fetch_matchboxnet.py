from pathlib import Path
import nemo.collections.asr as nemo_asr

name = "commandrecognition_en_matchboxnet3x1x64_v2_subset_task"
out = Path.home() / "kws_benchmark/checkpoints/matchboxnet3x1x64_gscv2_subset.nemo"
out.parent.mkdir(parents=True, exist_ok=True)

model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
    model_name=name
)

model.eval()

print("\n===== MODEL =====")
print(type(model))

print("\n===== LABELS =====")
print(list(model.cfg.labels))

print("\n===== PARAMS =====")
print(sum(p.numel() for p in model.parameters()))

model.save_to(str(out))

print("\nsaved:", out)
