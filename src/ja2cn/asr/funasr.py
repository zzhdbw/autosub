import re

import numpy as np
import soundfile as sf
from funasr import AutoModel

from ja2cn.core import BaseASR


def _clean_text(text: str) -> str:
    """Strip SenseVoice special tokens like <|ja|>, <|NEUTRAL|>, etc."""
    return re.sub(r"<\|[^|]+\|>", "", text).strip()


class FunasrASR(BaseASR):
    """Japanese ASR with SenseVoiceSmall via FunASR."""

    def __init__(
        self,
        device: str = "cpu",
        model_dir: str | None = None,
    ):
        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            device=device,
            model_dir=model_dir,
            disable_update=True,
        )

    def _run_asr(self, audio: np.ndarray, sr: int) -> str:
        """Run ASR on a single audio array, return cleaned text."""
        result = self.model.generate(input=audio, fs=sr, batch_size_s=0)

        text = ""
        if isinstance(result, list) and len(result) > 0:
            text = _clean_text(result[0].get("text", ""))
        elif isinstance(result, dict):
            text = _clean_text(result.get("text", ""))

        return text

    def recognize(
        self,
        audio_path: str,
        chunk_duration_ms: int = 10000,
        vad_segments: list[dict] | None = None,
    ) -> list[dict]:
        """Run ASR on audio.

        If *vad_segments* is provided, process each VAD segment individually
        so every speech segment becomes one subtitle entry (no merging).

        Otherwise fall back to fixed-duration chunking (original behaviour).

        Returns list of ``{start, end, text}`` in milliseconds.
        """
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        if vad_segments:
            return self._recognize_with_vad(audio, sr, vad_segments)
        return self._recognize_fixed_chunks(audio, sr, chunk_duration_ms)

    def _recognize_with_vad(
        self, audio: np.ndarray, sr: int, vad_segments: list[dict]
    ) -> list[dict]:
        """Run ASR on each VAD segment independently."""
        segments: list[dict] = []
        for seg in vad_segments:
            start_s = seg["start"] / 1000
            end_s = seg["end"] / 1000
            start_idx = int(start_s * sr)
            end_idx = int(end_s * sr)
            chunk = audio[start_idx:end_idx]

            if len(chunk) < sr * 0.1:
                continue

            text = self._run_asr(chunk, sr)
            if text:
                segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text,
                })

        return segments

    def _recognize_fixed_chunks(
        self, audio: np.ndarray, sr: int, chunk_duration_ms: int
    ) -> list[dict]:
        """Fixed-duration chunking with overlap (original behaviour)."""
        chunk_len = int(sr * chunk_duration_ms / 1000)
        overlap = int(sr * 0.5)
        step = chunk_len - overlap

        segments: list[dict] = []
        for i in range(0, len(audio), step):
            start_ms = int(i / sr * 1000)
            end_ms = int(min(i + chunk_len, len(audio)) / sr * 1000)
            chunk = audio[i : i + chunk_len]

            if len(chunk) < sr * 0.3:
                continue

            text = self._run_asr(chunk, sr)
            if text:
                segments.append({
                    "start": start_ms,
                    "end": end_ms,
                    "text": text,
                })

        return segments
