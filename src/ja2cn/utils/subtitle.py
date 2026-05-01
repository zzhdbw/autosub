from pathlib import Path


def _fmt_ts(ms: int) -> str:
    """Convert milliseconds → SRT timecode (HH:MM:SS,mmm)."""
    ms = max(0, ms)
    total_sec = ms / 1000
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    mill = int(ms % 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{mill:03d}"


def generate_srt(
    segments: list[dict],
    output_path: str,
    min_duration_ms: int = 500,
) -> str:
    """Build an SRT file from translated segments.

    Each segment must contain ``start``, ``end`` (ms), and at least one of
    ``translation`` or ``text``.

    Returns the output file path.
    """
    if not segments:
        raise ValueError("No segments to generate subtitles from")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for idx, seg in enumerate(segments, 1):
        start = max(0, seg["start"])
        end = max(start + min_duration_ms, seg["end"])

        text = seg.get("translation") or seg.get("text", "")
        if not text:
            continue

        lines.append(str(idx))
        lines.append(f"{_fmt_ts(start)} --> {_fmt_ts(end)}")
        lines.append(text)
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)
