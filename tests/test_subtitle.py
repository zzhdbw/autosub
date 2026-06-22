import tempfile
from pathlib import Path

import pytest

from autosub.utils.subtitle import _fmt_ts, generate_srt


class TestFmtTs:
    def test_zero(self):
        assert _fmt_ts(0) == "00:00:00,000"

    def test_seconds(self):
        assert _fmt_ts(1000) == "00:00:01,000"

    def test_minutes(self):
        assert _fmt_ts(65000) == "00:01:05,000"

    def test_hours(self):
        assert _fmt_ts(3723000) == "01:02:03,000"

    def test_milliseconds(self):
        assert _fmt_ts(1234) == "00:00:01,234"

    def test_negative_clamps_to_zero(self):
        assert _fmt_ts(-500) == "00:00:00,000"


class TestGenerateSrt:
    def test_basic_generation(self, sample_segments):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.srt"
            result = generate_srt(sample_segments, str(out))

            assert result == str(out)
            content = out.read_text(encoding="utf-8")
            assert "00:00:00,000 --> 00:00:02,500" in content
            assert "你好" in content
            assert "00:00:03,000 --> 00:00:05,800" in content
            assert "你好吗" in content

    def test_creates_parent_dir(self, sample_segments):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sub" / "test.srt"
            generate_srt(sample_segments, str(out))
            assert out.exists()

    def test_falls_back_to_text(self):
        seg = [{"start": 0, "end": 1000, "text": "日本語"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.srt"
            generate_srt(seg, str(out))
            content = out.read_text(encoding="utf-8")
            assert "日本語" in content

    def test_empty_segments_raises(self):
        with pytest.raises(ValueError, match="No segments"):
            generate_srt([], "out.srt")

    def test_min_duration_clamp(self):
        seg = [{"start": 0, "end": 100, "text": "短い"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.srt"
            generate_srt(seg, str(out))
            content = out.read_text(encoding="utf-8")
            # end should be clamped to start + 500ms
            assert "00:00:00,500" in content

    def test_skips_empty_text(self):
        seg = [
            {"start": 0, "end": 1000, "text": ""},
            {"start": 1000, "end": 2000, "text": "hello"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.srt"
            generate_srt(seg, str(out))
            content = out.read_text(encoding="utf-8")
            # empty-text segment is skipped, non-empty gets its original index
            assert "hello" in content
            assert "00:00:01,000 --> 00:00:02,000" in content
