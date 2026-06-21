"""Model download utilities for AutoSub.

Uses direct HTTP downloads (``requests``) — no subprocess / CLI dependency,
works reliably inside PyInstaller bundles.
"""

from pathlib import Path as _Path
from typing import Callable as _Callable

import requests as _requests

MODEL_DIR = _Path(__file__).resolve().parents[2] / "model"

# Base URL for ModelScope raw file downloads
_MS_URL = "https://modelscope.cn/models/{model}/resolve/{branch}/{file}"

SENSEVOICE_FILES: list[dict] = [
    {"file": "model_quant.onnx", "branch": "master"},
    {"file": "am.mvn", "branch": "master"},
    {"file": "config.yaml", "branch": "master"},
    {"file": "configuration.json", "branch": "master"},
    {"file": "tokens.json", "branch": "master"},
    {"file": "README.md", "branch": "master"},
]

SENSEVOICE_BPE = {
    "url": _MS_URL.format(
        model="iic/SenseVoiceSmall",
        branch="master",
        file="chn_jpn_yue_eng_ko_spectok.bpe.model",
    ),
    "dest": "chn_jpn_yue_eng_ko_spectok.bpe.model",
}

MODELS: dict[str, dict] = {
    "silero_vad": {
        "name": "Silero VAD",
        "description": "Voice Activity Detection (2.2 MB)",
        "dest": MODEL_DIR / "silero_vad" / "silero_vad.onnx",
        "files": [{
            "url": "https://github.com/snakers4/silero-vad/raw/master/"
                   "src/site-packages/silero_vad/data/silero_vad.onnx",
            "dest": MODEL_DIR / "silero_vad" / "silero_vad.onnx",
        }],
    },
    "sensevoice": {
        "name": "SenseVoiceSmall (ASR)",
        "description": "Speech Recognition (230 MB)",
        "dest": MODEL_DIR / "SenseVoiceSmall-onnx",
        "files": None,  # resolved at download time
    },
    "hy_mt": {
        "name": "HY-MT1.5 GGUF (Translate)",
        "description": "Japanese\u2192Chinese Translation (1.1 GB)",
        "dest": MODEL_DIR / "HY-MT1.5-1.8B-GGUF" / "HY-MT1.5-1.8B-Q4_K_M.gguf",
        "files": [{
            "url": _MS_URL.format(
                model="Tencent-Hunyuan/HY-MT1.5-1.8B-GGUF",
                branch="main",
                file="HY-MT1.5-1.8B-Q4_K_M.gguf",
            ),
            "dest": MODEL_DIR / "HY-MT1.5-1.8B-GGUF" / "HY-MT1.5-1.8B-Q4_K_M.gguf",
        }],
    },
}


def status(key: str) -> str:
    """Return ``\"ok\"``, ``\"partial\"``, or ``\"missing\"``."""
    info = MODELS[key]
    if key == "sensevoice":
        d = info["dest"]
        required = ["model_quant.onnx", "chn_jpn_yue_eng_ko_spectok.bpe.model"]
        if d.is_dir() and all((d / f).exists() for f in required):
            return "ok"
        return "missing" if not d.exists() else "partial"
    dest = info["dest"]
    return "ok" if dest.exists() and dest.stat().st_size > 1000 else "missing"


def download(
    key: str,
    log_cb: _Callable[[str], None],
    prog_cb: _Callable[[float], None],
) -> None:
    """Download a model by key, reporting progress via callbacks."""
    info = MODELS[key]
    log_cb(f"Downloading {info['name']} ...\n")

    try:
        if key == "sensevoice":
            _download_sensevoice(info, log_cb, prog_cb)
        else:
            for f in info["files"]:
                f["dest"].parent.mkdir(parents=True, exist_ok=True)
                _download_file(f["url"], f["dest"], log_cb, prog_cb)

        log_cb(f"\u2713 {info['name']} download complete!\n")
        prog_cb(100)
    except Exception as exc:
        log_cb(f"ERROR: {exc}\n")
        raise


# ── Internal helpers ──────────────────────────────────────────────────


def _resolve_sensevoice_files(dest_dir: _Path) -> list[dict]:
    """Build the file list for SenseVoiceSmall-onnx download."""
    files = []
    for f in SENSEVOICE_FILES:
        files.append({
            "url": _MS_URL.format(
                model="iic/SenseVoiceSmall-onnx",
                branch=f["branch"],
                file=f["file"],
            ),
            "dest": dest_dir / f["file"],
        })
    files.append({
        "url": SENSEVOICE_BPE["url"],
        "dest": dest_dir / SENSEVOICE_BPE["dest"],
    })
    return files


def _download_sensevoice(
    info: dict,
    log_cb: _Callable[[str], None],
    prog_cb: _Callable[[float], None],
) -> None:
    dest_dir = info["dest"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = _resolve_sensevoice_files(dest_dir)

    total_bytes = 0
    sizes: list[int] = []
    log_cb("  Resolving file sizes ...\n")
    for f in files:
        try:
            resp = _requests.head(f["url"], allow_redirects=True, timeout=15)
            sz = int(resp.headers.get("content-length", 0))
        except Exception:
            sz = 0
        sizes.append(sz)
        total_bytes += sz

    log_cb(f"  Total: {_fmt_size(total_bytes)}\n")

    downloaded = 0
    for i, f in enumerate(files):
        fname = f["dest"].name
        f["dest"].parent.mkdir(parents=True, exist_ok=True)
        log_cb(f"  [{i + 1}/{len(files)}] {fname} ...\n")

        def _local_prog(pct: float) -> None:
            file_done = int(pct / 100 * sizes[i]) if sizes[i] > 0 else 0
            cumulative = downloaded + file_done
            if total_bytes > 0:
                prog_cb(cumulative / total_bytes * 100)

        _download_file(f["url"], f["dest"], log_cb, _local_prog)
        downloaded += sizes[i]

    prog_cb(100)


def _download_file(
    url: str,
    dest: _Path,
    log_cb: _Callable[[str], None],
    prog_cb: _Callable[[float], None],
) -> None:
    """Download a single file with progress reporting."""
    resp = _requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    done = 0
    TEMP_SUFFIX = ".autosub_part"

    temp_path = dest.with_suffix(dest.suffix + TEMP_SUFFIX)
    with open(temp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                done += len(chunk)
                if total:
                    prog_cb(done / total * 100)
    if not total:
        prog_cb(100)

    temp_path.rename(dest)


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
