from pathlib import Path
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess


model_dir = str(Path(__file__).resolve().parent / "model" / "SenseVoiceSmall-onnx")

model = SenseVoiceSmall(model_dir, batch_size=1, quantize=True)

# inference
wav_or_scp = ["en.mp3"]

res = model(wav_or_scp, language="auto", textnorm="withitn")
print([rich_transcription_postprocess(i) for i in res])
