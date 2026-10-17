"""Проверка клиента LLM: стриминг ответа, отмена посреди ответа, следующий запрос.

uv run --group core python scripts/probe_llm.py [вопрос]
"""

import asyncio
import logging
import sys
import time

from terebot.config import load_persona, load_settings
from terebot.core.llm import Llm
from terebot.core.prompt import SaidByParticipant, build_messages

LONG = "Теребот, расскажи длинную историю про свой самый странный день, подробно."


def prompt(question: str) -> list[dict]:
    no_budget = 10**9
    return build_messages(
        load_persona(), [], [SaidByParticipant("Вася", question)], no_budget, lambda m: 0
    )[0]


async def main(question: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    llm = Llm(load_settings().llm)
    await llm.start()
    try:
        messages = prompt(question)
        print(f"токенов в промпте: {await llm.count_tokens(messages)}")

        print("\n1) Ответ, куски через «|»:")
        start, first = time.monotonic(), None
        async for chunk in llm.stream(messages):
            first = first or time.monotonic() - start
            print(chunk, end="|", flush=True)
        print(f"\nпервый токен через {first:.2f} с, всего {time.monotonic() - start:.2f} с")

        print("\n2) Отмена через 1 с:")

        async def consume():
            async for chunk in llm.stream(prompt(LONG)):
                print(chunk, end="", flush=True)

        task = asyncio.create_task(consume())
        await asyncio.sleep(1)
        task.cancel()
        try:
            await task
            print("\n(поток закончился сам до отмены)")
        except asyncio.CancelledError:
            print("\n(поток оборван)")

        print("\n3) Следующий запрос после отмены:")
        start = time.monotonic()
        text = "".join([c async for c in llm.stream(messages)])
        print(f"{text}\n({time.monotonic() - start:.2f} с)")
    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Теребот, ты как вообще? Что делал сегодня?"))
