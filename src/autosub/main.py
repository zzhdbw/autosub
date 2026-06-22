import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from autosub.translator import LlamaCppTranslator, TransformersTranslator
from autosub.asr import SenseVoiceASR
from autosub.utils.audio import extract_audio
from autosub.utils.subtitle import generate_srt
from autosub.vad import SileroVAD


def setup_logging(verbose: bool) -> None:
    logger.remove()
    fmt = (
        "{time:HH:mm:ss.SSS} | {level:<7} | {message}"
        if verbose
        else "{time:HH:mm:ss} | {level:<7} | {message}"
    )
    logger.add(sys.stderr, format=fmt, level="DEBUG" if verbose else "INFO")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Japanese video audio → Chinese subtitles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  autosub movie.mp4                          # full pipeline\n"
            "  autosub movie.mp4 --skip-vad               # re-run ASR with existing VAD\n"
            "  autosub movie.mp4 --skip-vad --skip-asr    # translate only\n"
            "  autosub movie.mp4 --skip-translate         # ASR-only (Japanese subs)\n"
        ),
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .srt path (default: output/<input>.srt)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Working directory for intermediate files (default: output/)",
    )
    parser.add_argument(
        "--model-dir",
        default="model",
        help="Root directory for models (default: model/)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu"],
        help="Device to run models on (default: cpu)",
    )
    parser.add_argument(
        "--prompt-template",
        default="将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}",
        help="Prompt template for translation model. Use {text} as placeholder",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling (default: 20)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.6,
        help="Top-p sampling (default: 0.6)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Repetition penalty (default: 1.05)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=10000,
        help="ASR chunk duration in ms (default: 10000; only used without VAD)",
    )
    parser.add_argument(
        "--backend",
        choices=["llamacpp", "transformers"],
        default="llamacpp",
        help="Translation backend (default: llamacpp)",
    )
    parser.add_argument(
        "--gguf-model",
        default="model/Hy-MT2-1.8B-1.25Bit-GGUF/Hy-MT2-1.8B-1.25Bit.gguf",
        help="Path to GGUF model file (default: model/...1.25Bit.gguf)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    parser.add_argument(
        "--skip-vad",
        action="store_true",
        help="Skip VAD; load existing VAD segments from output dir",
    )
    parser.add_argument(
        "--skip-asr",
        action="store_true",
        help="Skip ASR; load existing segments JSON from output dir",
    )
    parser.add_argument(
        "--skip-translate",
        action="store_true",
        help="Skip translation; generate SRT from raw ASR text",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: {}", input_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    output_srt = args.output or str(output_dir / f"{stem}.srt")
    vad_json = str(output_dir / f"{stem}_vad.json")
    segments_json = str(output_dir / f"{stem}_segments.json")
    translated_json = str(output_dir / f"{stem}_translated.json")

    # ── Step 1: Audio extraction ──────────────────────────────────────
    logger.info("Step 1/4: Extracting audio …")
    audio_path = extract_audio(str(input_path), str(output_dir))
    logger.info("  → {}", audio_path)

    # ── Step 2: VAD ───────────────────────────────────────────────────
    logger.info("Step 2/4: Voice activity detection (Silero VAD) …")
    if args.skip_vad:
        logger.info("  (--skip-vad: loading existing VAD segments)")
        vad_segments = json.loads(Path(vad_json).read_text(encoding="utf-8"))
    else:
        detector = SileroVAD(
            device=args.device,
            model_dir=str(model_dir),
        )
        vad_segments = detector.detect_and_save(audio_path, vad_json)

    logger.info("  → {} speech segment(s) found", len(vad_segments))

    # ── Step 3: ASR ───────────────────────────────────────────────────
    logger.info("Step 3/4: Speech recognition (SenseVoiceSmall) …")
    if args.skip_asr:
        logger.info("  (--skip-asr: loading existing segments)")
        segments = json.loads(Path(segments_json).read_text(encoding="utf-8"))
    else:
        recognizer = SenseVoiceASR(
            device=args.device,
            model_dir=model_dir / "SenseVoiceSmall-onnx",
        )
        segments = recognizer.recognize_and_save(
            audio_path,
            segments_json,
            chunk_duration_ms=args.chunk_ms,
            vad_segments=vad_segments,
        )

    logger.info("  → {} segment(s) recognised", len(segments))

    # Free ASR model memory before loading the translator
    if not args.skip_asr:
        del recognizer

    if args.skip_translate:
        logger.info("  (--skip-translate: generating SRT from ASR text)")
        srt_path = generate_srt(segments, output_srt)
        logger.info("  ✓ Subtitles → {}", srt_path)
        return

    # ── Step 4: Translation ───────────────────────────────────────────
    if args.backend == "transformers":
        logger.info("Step 4/4: Translating (Transformers) …")
        translator = TransformersTranslator(
            device=args.device,
            model_dir=str(model_dir),
            prompt_template=args.prompt_template,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    else:
        logger.info("Step 4/4: Translating (llama.cpp / {}) …", args.gguf_model)
        translator = LlamaCppTranslator(
            model_path=args.gguf_model,
            prompt_template=args.prompt_template,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    translated = translator.translate_segments(segments, translated_json)

    # ── Step 5: SRT ───────────────────────────────────────────────────
    logger.info("Generating subtitle file …")
    srt_path = generate_srt(translated, output_srt)
    logger.info("  ✓ Subtitles → {}", srt_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
