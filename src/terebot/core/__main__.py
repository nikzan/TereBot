"""Запуск ядра Клона: `uv run --group core python -m terebot.core`.

Загружает и прогревает модели, при необходимости запускает llama-server и ждёт шлюз
на 127.0.0.1:[core].port. Остановка — Ctrl+C: Журнал Сессии закрывается со сводкой.
"""

import asyncio
import contextlib
import logging
import signal
import sys

from terebot.config import SettingsError, load_persona, load_settings, resolve_settings_path
from terebot.core.server import Core


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "websockets", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        settings = load_settings()
        persona = load_persona()
    except SettingsError as e:
        sys.exit(f"Ошибка настроек: {e}")
    logging.getLogger("terebot.core").info("Настройки: %s", resolve_settings_path())

    async def run() -> None:
        # Явно: библиотеки моделей при загрузке подменяют обработчик SIGINT, и стандартный
        # asyncio.run его тогда не ставит — Ctrl+C не останавливал бы ядро.
        loop = asyncio.get_running_loop()
        main_task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, main_task.cancel)
        core = Core(settings, persona)
        try:
            await core.load()
        except Exception:
            await core.close()
            raise
        await core.serve()

    with contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError):
        asyncio.run(run())


if __name__ == "__main__":
    main()
