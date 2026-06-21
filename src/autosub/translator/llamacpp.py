import json
import os
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

from loguru import logger

from autosub.core import BaseTranslator

DEFAULT_PROMPT_TEMPLATE = (
    "将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 20
DEFAULT_TOP_P = 0.6
DEFAULT_REPETITION_PENALTY = 1.05
_SERVER_PORT = 18080

# Path to the STQ-enabled llama.cpp binary, relative to project root
_LLAMA_BIN = Path(__file__).resolve().parents[3] / "llama.cpp_stq" / "build" / "bin" / "llama"


class LlamaCppTranslator(BaseTranslator):
    """Japanese-to-Chinese translation via persistent llama.cpp server (STQ).

    Starts a background ``llama-server`` process on init and sends HTTP
    requests for each translation — model stays loaded between calls.
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
        llama_bin: str | Path | None = None,
    ):
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.llama_bin = str(llama_bin or _LLAMA_BIN)
        self._server: subprocess.Popen | None = None

        resolved = _resolve_gguf(model_path)
        self.model_path = resolved

        # Verify binary exists
        bin_path = Path(self.llama_bin)
        if not bin_path.exists():
            build_script = Path(__file__).resolve().parents[3] / "scripts" / "build_llama_stq.sh"
            raise FileNotFoundError(
                f"llama.cpp STQ binary not found at {self.llama_bin}. "
                f"Build it first:\n"
                f"  bash {build_script}"
            )

        # Use llama-server from same directory
        server_bin = bin_path.parent / "llama-server"
        if not server_bin.exists():
            raise FileNotFoundError(f"llama-server not found next to llama binary: {server_bin}")

        logger.info("Starting llama-server (model: {}) …", resolved)
        logger.info(
            "  params: temp={}, top_k={}, top_p={}, repeat_penalty={}",
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        )

        self._server = subprocess.Popen(
            [
                str(server_bin),
                "-m", self.model_path,
                "-ngl", "0",
                "--port", str(_SERVER_PORT),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        self._base_url = f"http://127.0.0.1:{_SERVER_PORT}"
        self._wait_ready(timeout=120)

    def _wait_ready(self, timeout: int = 120) -> None:
        """Poll the server health endpoint until it responds."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Check if server process is still alive
            if self._server is not None and self._server.poll() is not None:
                err = self._server.stderr.read() if self._server.stderr else b""
                raise RuntimeError(
                    f"llama-server exited early (code {self._server.returncode}): "
                    f"{err.decode(errors='replace')[:500]}"
                )
            try:
                req = Request(f"{self._base_url}/health", method="GET")
                with urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        logger.info("  ✓ server ready")
                        return
            except URLError:
                pass
            time.sleep(0.5)
        raise RuntimeError(
            f"llama-server did not start within {timeout}s"
        )

    def translate(self, text: str) -> str:
        """Translate a single string to Chinese via llama-server HTTP API."""
        prompt = self.prompt_template.format(text=text)
        full_prompt = (
            "<｜hy_begin▁of▁sentence｜>"
            "<｜hy_User｜>"
            f"{prompt}"
            "<｜hy_Assistant｜>"
        )

        body = json.dumps({
            "prompt": full_prompt,
            "n_predict": 1024,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repeat_penalty": self.repetition_penalty,
            "stop": ["<｜hy_place▁holder▁no▁2｜>"],
        }).encode()

        req = Request(
            f"{self._base_url}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
            raw = result.get("content", "").strip()
            return raw if raw else text
        except URLError as e:
            logger.error("Translation request failed: {}", e)
            return text

    def __del__(self) -> None:
        if self._server is not None:
            self._server.terminate()
            try:
                self._server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server.kill()


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
