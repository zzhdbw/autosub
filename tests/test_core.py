import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from autosub.core.base import BaseASR, BaseTranslator, BaseVAD, _write_json


class TestWriteJson:
    def test_writes_json_file(self):
        data = [{"a": 1}, {"b": 2}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.json")
            _write_json(path, data)
            result = json.loads(Path(path).read_text(encoding="utf-8"))
            assert result == data

    def test_creates_parent_dirs(self):
        data = [{"x": "y"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "deep" / "nested" / "out.json")
            _write_json(path, data)
            assert Path(path).exists()


@dataclass
class _FakeVAD(BaseVAD):
    _segments: list[dict]

    def detect(self, audio_path: str, **kwargs) -> list[dict]:
        return self._segments


@dataclass
class _FakeASR(BaseASR):
    _segments: list[dict]

    def recognize(self, audio_path: str, **kwargs) -> list[dict]:
        return self._segments


@dataclass
class _FakeTranslator(BaseTranslator):
    def translate(self, text: str) -> str:
        return text.upper()


class TestBaseVAD:
    def test_detect_and_save(self):
        segs = [{"start": 0, "end": 100}]
        vad = _FakeVAD(segs)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "vad.json")
            result = vad.detect_and_save("fake.wav", path)
            assert result == segs
            assert Path(path).exists()


class TestBaseASR:
    def test_recognize_and_save(self):
        segs = [{"start": 0, "end": 100, "text": "hello"}]
        asr = _FakeASR(segs)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "asr.json")
            result = asr.recognize_and_save("fake.wav", path)
            assert result == segs
            assert Path(path).exists()


class TestBaseTranslator:
    def test_translate_segments(self):
        segments = [
            {"start": 0, "end": 100, "text": "hello"},
            {"start": 200, "end": 300, "text": "world"},
        ]
        t = _FakeTranslator()
        result = t.translate_segments(segments)
        assert len(result) == 2
        assert result[0]["translation"] == "HELLO"
        assert result[1]["translation"] == "WORLD"

    def test_translate_segments_saves_json(self):
        segments = [{"start": 0, "end": 100, "text": "hi"}]
        t = _FakeTranslator()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "translated.json")
            t.translate_segments(segments, path)
            assert Path(path).exists()
            saved = json.loads(Path(path).read_text(encoding="utf-8"))
            assert saved[0]["translation"] == "HI"
