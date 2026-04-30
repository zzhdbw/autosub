import json
from pathlib import Path

import soundfile as sf
from silero_vad import get_speech_timestamps, load_silero_vad


class VADDetector:
    """Voice Activity Detection using Silero VAD (language-agnostic)."""

    def __init__(
        self,
        device: str = "cpu",
        model_dir: str | None = None,
    ):
        self.model = load_silero_vad(onnx=True)

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

    def detect_and_save(
        self, audio_path: str, output_path: str, min_duration_ms: int = 1000,
    ) -> list[dict]:
        """Run VAD and dump segments to JSON, return them."""
        segments = self.detect(audio_path, min_duration_ms=min_duration_ms)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return segments
