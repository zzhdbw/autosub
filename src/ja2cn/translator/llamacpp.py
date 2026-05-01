import os
from pathlib import Path

from loguru import logger

from ja2cn.core import BaseTranslator

DEFAULT_PROMPT_TEMPLATE = (
    "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 20
DEFAULT_TOP_P = 0.6
DEFAULT_REPETITION_PENALTY = 1.05


class LlamaCppTranslator(BaseTranslator):
    """Japanese-to-Chinese translation via llama.cpp (GGUF).

    Uses the GGUF model's native tokenizer — no HuggingFace Transformers dependency.
    Prompt format and inference parameters follow the official HY-MT1.5 recommendations.
    """

    def __init__(
        self,
        model_path: str,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    ):
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        resolved = _resolve_gguf(model_path)
        logger.info("Loading GGUF model: {} …", resolved)
        logger.info(
            "  params: temp={}, top_k={}, top_p={}, repeat_penalty={}",
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        )
        from llama_cpp import Llama

        self.model = Llama(
            model_path=resolved,
            n_ctx=n_ctx,
            n_threads=n_threads or max(os.cpu_count() or 4, 4),
            verbose=False,
        )
        logger.info(
            "  ✓ model loaded (vocab={}, threads={}, ctx={})",
            self.model.n_vocab(),
            n_threads or os.cpu_count(),
            n_ctx,
        )

    def translate(self, text: str) -> str:
        """Translate a single Japanese string to Chinese."""
        prompt = self.prompt_template.format(text=text)
        result = self.model.create_completion(
            prompt,
            max_tokens=1024,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repeat_penalty=self.repetition_penalty,
            stop=None,
        )
        raw = result["choices"][0]["text"].strip()
        return raw if raw else text


def _resolve_gguf(path: str) -> str:
    """Resolve a GGUF file path.

    - If *path* is a directory, use the first .gguf file inside.
    - If *path* is a file, use it as-is.
    - If *path* doesn't exist, try appending .gguf.
    """
    p = Path(path)
    if p.is_dir():
        ggufs = list(p.glob("*.gguf"))
        if not ggufs:
            raise FileNotFoundError(f"No .gguf files found in {p}")
        result = str(ggufs[0])
        logger.debug("Resolved directory {} → {}", p, result)
        return result
    if p.exists():
        return str(p)
    # try with .gguf suffix
    with_gguf = p.with_suffix(".gguf")
    if with_gguf.exists():
        return str(with_gguf)
    raise FileNotFoundError(f"GGUF model not found: {path}")
