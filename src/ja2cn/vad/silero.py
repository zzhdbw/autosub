import importlib.resources as resources
from pathlib import Path

import soundfile as sf
from silero_vad import get_speech_timestamps
from silero_vad.utils_vad import OnnxWrapper

from ja2cn.core import BaseVAD


class SileroVAD(BaseVAD):
    """Voice Activity Detection using Silero VAD (language-agnostic).

    Loads the ONNX model from the project ``model_dir/silero_vad/`` if available,
    otherwise falls back to the bundled copy shipped with the ``silero-vad`` pip
    package — no network access at runtime.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_dir: str | None = None,
    ):
        model_path = self._resolve_model(model_dir or "model")
        self.model = OnnxWrapper(str(model_path), force_onnx_cpu=True)

    @staticmethod
    def _resolve_model(model_dir: str) -> Path:
        candidate = Path(model_dir) / "silero_vad" / "silero_vad.onnx"
        if candidate.exists():
            return candidate
        return resources.files("silero_vad.data").joinpath("silero_vad.onnx")

    def detect(
        self, audio_path: str, min_duration_ms: int = 1000,
    ) -> list[dict]:
        """Run VAD on audio file, return list of {start_ms, end_ms} segments.

        Segments shorter than *min_duration_ms* are filtered out.
        """
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        timestamps = get_speech_timestamps(
            audio, self.model, sampling_rate=sr,
            return_seconds=True,
        )

        segments = [
            {"start": int(t["start"] * 1000), "end": int(t["end"] * 1000)}
            for t in timestamps
            if (t["end"] - t["start"]) * 1000 >= min_duration_ms
        ]
        return segments
