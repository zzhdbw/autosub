from pathlib import Path

import av
import numpy as np
import soundfile as sf
from loguru import logger


def extract_audio(video_path: str, output_dir: str, sample_rate: int = 16000) -> str:
    """Extract audio from video as 16 kHz mono WAV via PyAV."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}.wav"

    container = av.open(str(video_path))

    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if audio_stream is None:
        container.close()
        raise ValueError(f"No audio stream found in {video_path}")

    resampler = av.AudioResampler(
        format="s16p",
        layout="mono",
        rate=sample_rate,
    )

    frames: list[np.ndarray] = []
    for frame in container.decode(audio=0):
        resampled = resampler.resample(frame)
        for r in resampled:
            frames.append(r.to_ndarray())

    container.close()

    if not frames:
        raise ValueError("No audio data decoded from video")

    audio = np.concatenate(frames, axis=1).T  # (channels, samples) → (samples, channels)
    sf.write(str(output_path), audio, sample_rate, subtype="PCM_16")

    logger.debug("Audio extracted: {} ({} Hz, mono)", output_path, sample_rate)
    return str(output_path)
