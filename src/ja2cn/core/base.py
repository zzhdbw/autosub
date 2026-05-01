"""Abstract base classes for VAD, ASR, and Translator pipelines."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger


# ── VAD ─────────────────────────────────────────────────────────────────

class BaseVAD(ABC):
    """Voice Activity Detection base class."""

    @abstractmethod
    def detect(self, audio_path: str, **kwargs) -> list[dict]:
        """Run VAD on *audio_path*, return ``[{start, end}, …]`` in milliseconds."""
        ...

    def detect_and_save(
        self, audio_path: str, output_path: str, **kwargs
    ) -> list[dict]:
        """Run VAD and dump segments to JSON, return them."""
        segments = self.detect(audio_path, **kwargs)
        _write_json(output_path, segments)
        return segments


# ── ASR ─────────────────────────────────────────────────────────────────

class BaseASR(ABC):
    """Speech Recognition base class."""

    @abstractmethod
    def recognize(self, audio_path: str, **kwargs) -> list[dict]:
        """Run ASR on *audio_path*, return ``[{start, end, text}, …]`` in ms."""
        ...

    def recognize_and_save(
        self, audio_path: str, output_path: str, **kwargs
    ) -> list[dict]:
        """Run ASR and dump segments to JSON, return them."""
        segments = self.recognize(audio_path, **kwargs)
        _write_json(output_path, segments)
        return segments


# ── Translator ──────────────────────────────────────────────────────────

class BaseTranslator(ABC):
    """Translation base class."""

    @abstractmethod
    def translate(self, text: str) -> str:
        """Translate a single string."""
        ...

    def translate_segments(
        self, segments: list[dict], output_path: str | None = None
    ) -> list[dict]:
        """Translate all segments, optionally save to JSON, return enriched list."""
        translated: list[dict] = []
        for idx, seg in enumerate(segments, 1):
            result = self.translate(seg["text"])
            logger.info(
                "[{}/{}] 日: {} → 中: {}",
                idx, len(segments), seg["text"], result,
            )
            translated.append({**seg, "translation": result})

        if output_path:
            _write_json(output_path, translated)

        return translated


# ── shared helpers ──────────────────────────────────────────────────────

def _write_json(path: str, data: list) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
