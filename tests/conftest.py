import pytest


@pytest.fixture
def sample_segments():
    return [
        {"start": 0, "end": 2500, "text": "こんにちは", "translation": "你好"},
        {"start": 3000, "end": 5800, "text": "元気ですか", "translation": "你好吗"},
    ]
