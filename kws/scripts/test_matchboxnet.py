import torch
import nemo.collections.asr as nemo_asr

ckpt = "/home/zhaoyan/kws_benchmark/checkpoints/matchboxnet3x1x64_gscv2_subset.nemo"

wav = "/data/zhaoyan/datasets/speech_commands_v0.02/down/022cd682_nohash_0.wav"

model = nemo_asr.models.EncDecClassificationModel.restore_from(
    restore_path=ckpt
)

model.eval()
model.cuda()

print("labels:", list(model.cfg.labels))

pred = model.transcribe(
    [wav],
    batch_size=1,
    logprobs=False
)

print("wav :", wav)
print("pred:", pred)
