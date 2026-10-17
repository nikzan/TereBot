"""LLM через llama-server: запуск дочерним процессом, стриминг ответа, подсчёт токенов.

Рассуждения выключены дважды: флагом `--reasoning off` при запуске и `enable_thinking: false`
в запросе — на случай, если llama-server запущен вручную без флага.
"""

import asyncio
import glob
import json
import logging
import subprocess
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from terebot.config import REPO_ROOT, LlmSettings, SettingsError

log = logging.getLogger(__name__)

LOG_FILE = REPO_ROOT / "data" / "llama-server.log"
START_TIMEOUT_S = 180


class LlmError(Exception):
    """LLM недоступна, ответила ошибкой или не уложилась в таймаут."""


def resolve_model(pattern: str) -> Path:
    matches = sorted(glob.glob(str(Path(pattern).expanduser())))
    if not matches:
        raise SettingsError(f"llm.model: GGUF не найден по пути {pattern}")
    return Path(matches[-1])


class Llm:
    def __init__(self, settings: LlmSettings):
        self._settings = settings
        self.url = f"http://127.0.0.1:{settings.port}"
        self._process: subprocess.Popen | None = None
        self._client = httpx.AsyncClient(base_url=self.url, timeout=settings.timeout_s)
        # PID llama-server для замера памяти: свой дочерний процесс или уже запущенный.
        self.pid: int | None = None

    async def start(self) -> None:
        """Подключиться к llama-server или запустить его, если порт не отвечает."""
        if await self._healthy():
            self.pid = _pid_on_port(self._settings.port)
            log.info("llama-server уже запущен на %s (pid %s)", self.url, self.pid)
            return
        model = resolve_model(self._settings.model)
        cmd = [
            "llama-server", "-m", str(model), "-c", str(self._settings.ctx_size),
            "--port", str(self._settings.port), "--host", "127.0.0.1", "--reasoning", "off",
        ]  # fmt: skip
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log.info("запуск: %s (лог — %s)", " ".join(cmd), LOG_FILE)
        with open(LOG_FILE, "ab") as out:
            self._process = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        self.pid = self._process.pid
        deadline = time.monotonic() + START_TIMEOUT_S
        while not await self._healthy():
            if self._process.poll() is not None:
                raise LlmError(
                    f"llama-server завершился с кодом {self._process.returncode}, см. {LOG_FILE}"
                )
            if time.monotonic() > deadline:
                raise LlmError(f"llama-server не поднялся за {START_TIMEOUT_S} с, см. {LOG_FILE}")
            await asyncio.sleep(0.5)
        log.info("llama-server готов (pid %s)", self.pid)

    async def close(self) -> None:
        await self._client.aclose()
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(10)
            except subprocess.TimeoutExpired:
                self._process.kill()

    async def _healthy(self) -> bool:
        try:
            r = await self._client.get("/health", timeout=2)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except (httpx.HTTPError, ValueError):
            return False

    async def tokenize(self, text: str) -> int:
        try:
            r = await self._client.post("/tokenize", json={"content": text})
            r.raise_for_status()
            return len(r.json()["tokens"])
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise LlmError(f"подсчёт токенов: {e}") from e

    async def count_tokens(self, messages: list[dict]) -> int:
        """Токены промпта: шаблон чата модели (/apply-template) → /tokenize."""
        try:
            r = await self._client.post("/apply-template", json={"messages": messages})
            r.raise_for_status()
            r = await self._client.post("/tokenize", json={"content": r.json()["prompt"]})
            r.raise_for_status()
            return len(r.json()["tokens"])
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise LlmError(f"подсчёт токенов: {e}") from e

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Текст ответа по кускам. Отмена задачи закрывает HTTP-поток, и генерация обрывается."""
        body = {
            "messages": messages,
            "stream": True,
            "max_tokens": self._settings.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            async with asyncio.timeout(self._settings.timeout_s):
                async with self._client.stream("POST", "/v1/chat/completions", json=body) as r:
                    if r.status_code != 200:
                        await r.aread()
                        raise LlmError(f"HTTP {r.status_code}: {r.text[:200]}")
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[len("data: ") :]
                        if data == "[DONE]":
                            return
                        chunk = json.loads(data)
                        for choice in chunk.get("choices", []):
                            text = choice.get("delta", {}).get("content")
                            if text:
                                yield text
        except TimeoutError:
            raise LlmError(f"нет ответа за {self._settings.timeout_s} с") from None
        except httpx.HTTPError as e:
            raise LlmError(f"{type(e).__name__}: {e}") from e


def _pid_on_port(port: int) -> int | None:
    out = subprocess.run(
        ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True
    ).stdout.split()
    return int(out[0]) if out else None
