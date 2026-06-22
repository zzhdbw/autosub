import pytest

from autosub.model_manager import _fmt_size, status


class TestFmtSize:
    def test_bytes(self):
        assert _fmt_size(0) == "0.0 B"
        assert _fmt_size(500) == "500.0 B"

    def test_kb(self):
        assert _fmt_size(2048) == "2.0 KB"

    def test_mb(self):
        assert _fmt_size(2 * 1024 * 1024) == "2.0 MB"

    def test_gb(self):
        assert _fmt_size(3 * 1024 * 1024 * 1024) == "3.0 GB"


class TestStatus:
    def test_returns_valid_status(self):
        for key in ("silero_vad", "hy_mt", "sensevoice"):
            assert status(key) in ("ok", "missing", "partial")

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            status("nonexistent_model")
