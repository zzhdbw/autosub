import subprocess
import sys
from pathlib import Path

from loguru import logger


def extract_audio(video_path: str, output_dir: str, sample_rate: int = 16000) -> str:
    """Extract audio from video as 16 kHz mono WAV via ffmpeg."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}.wav"

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-y",
        str(output_path),
    ]

    logger.debug("Running ffmpeg: {}", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        logger.error("ffmpeg not found. Install it:")
        logger.error("  macOS: brew install ffmpeg")
        logger.error("  Ubuntu/Debian: sudo apt install ffmpeg")
        logger.error("  Windows: winget install ffmpeg")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        logger.error("ffmpeg error: {}", exc.stderr.decode(errors="replace"))
        sys.exit(1)

    logger.debug("Audio extracted: {} ({} Hz, mono)", output_path, sample_rate)
    return str(output_path)
