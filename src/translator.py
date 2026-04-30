import json
from pathlib import Path

import torch
from loguru import logger
from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_PROMPT_TEMPLATE = (
    "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 20
DEFAULT_TOP_P = 0.6
DEFAULT_REPETITION_PENALTY = 1.05


class Translator:
    """Japanese-to-Chinese translation via HY-MT1.5-1.8B.

    Prompt format and inference parameters follow the official HY-MT1.5 recommendations.
    """

    def __init__(
        self,
        model_name: str = "Tencent-Hunyuan/HY-MT1.5-1.8B",
        device: str = "cpu",
        model_dir: str = "model",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        max_new_tokens: int = 1024,
        num_beams: int = 1,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    ):
        self.device = device
        self.prompt_template = prompt_template
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        logger.info("Downloading {} from ModelScope …", model_name)
        model_path = snapshot_download(model_name, cache_dir=model_dir)

        logger.info("Loading tokenizer …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model (this will take a while on CPU) …")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float32, trust_remote_code=True
            ).to(device)
            self._is_causal = True
        except Exception:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, torch_dtype=torch.float32, trust_remote_code=True
            ).to(device)
            self._is_causal = False
        self.model.eval()
        logger.debug("Model loaded (causal={})", self._is_causal)

    def translate(self, text: str) -> str:
        """Translate a single Japanese string to Chinese."""
        prompt = self.prompt_template.format(text=text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=self.temperature > 0,
            )

        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if self._is_causal:
            translation = raw[len(prompt):].strip()
        else:
            translation = raw.strip()

        return translation if translation else text

    def translate_segments(
        self, segments: list[dict], output_path: str | None = None
    ) -> list[dict]:
        """Translate all segments, optionally save to JSON, return enriched list."""
        translated: list[dict] = []
        for idx, seg in enumerate(segments, 1):
            result = self.translate(seg["text"])
            logger.info("[{}/{}] 日: {} → 中: {}", idx, len(segments), seg["text"], result)
            translated.append({
                **seg,
                "translation": result,
            })

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(translated, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return translated
