"""Проверка Listener на записи Сессии шлюза: кадры подаются в реальном темпе, печатаются Реплики.

uv run --group core python scripts/probe_listener.py data/recordings/<Сессия> [--limit-s 60]
"""

import argparse
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

from terebot.config import load_settings
from terebot.core.audio import float_to_pcm
from terebot.core.listener import Listener
from terebot.core.models import Asr, SileroVad, SmartTurn

FRAME = 960


async def main(args) -> None:
    wav = args.session / f"{args.user}.wav" if args.user else next(args.session.glob("*.wav"))
    segments = [json.loads(line) for line in wav.with_suffix(".segments.jsonl").open()]
    audio, _ = sf.read(wav, dtype="float32")
    settings = load_settings().speech
    print("загрузка моделей…")
    vad, turn, asr = SileroVad(), SmartTurn(), Asr()
    start = time.monotonic()

    def clock() -> float:
        return time.monotonic() - start

    def on_utterance(u, marks):
        print(
            f"{clock():7.2f}  Реплика {u.t_start:.2f}–{u.t_end:.2f}: «{u.text}» "
            f"(конец хода через {marks.turn_decided - u.t_end:.2f} с, "
            f"текст через {marks.text_ready - u.t_end:.2f} с после конца речи)"
        )

    listener = Listener(
        wav.stem,
        settings,
        vad,
        turn,
        asr,
        ThreadPoolExecutor(2),
        clock,
        on_activity=lambda a: None,
        on_utterance=on_utterance,
    )

    async def ticker():
        while True:
            listener.tick(clock())
            await asyncio.sleep(0.1)

    tick_task = asyncio.create_task(ticker())
    for seg in segments:
        t0 = seg["start_ms"] / 1000
        if t0 > args.limit_s:
            break
        chunk = audio[seg["sample_offset"] : seg["sample_offset"] + round(seg["duration_ms"] * 48)]
        for i in range(0, len(chunk) - FRAME + 1, FRAME):
            t = t0 + (i + FRAME) / 48_000
            await asyncio.sleep(max(0.0, t - clock()))
            listener.frame(float_to_pcm(chunk[i : i + FRAME]), t)
    await asyncio.sleep(3)
    tick_task.cancel()
    await listener.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path)
    parser.add_argument("--user")
    parser.add_argument("--limit-s", type=float, default=60)
    asyncio.run(main(parser.parse_args()))
