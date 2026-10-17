"""Проверка VAD и Smart Turn на записи Сессии шлюза: моменты речи, вероятности конца хода, Реплики.

    uv run --group core python scripts/probe_turn.py data/recordings/<Сессия> [--limit-s 120]

Фрагменты речи берутся из <user>.segments.jsonl со своим временем: в WAV тишины между ними нет,
а ядро тоже видит её как отсутствие кадров.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from terebot.config import load_settings
from terebot.core.audio import downsample_48k_to_16k
from terebot.core.models import SAMPLE_RATE, SileroVad, SmartTurn
from terebot.core.turn import Activity, CheckTurn, EndUtterance, TurnDetector

WINDOW = SileroVad.WINDOW


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path)
    parser.add_argument("--user", help="user ID; по умолчанию первый WAV в каталоге")
    parser.add_argument("--limit-s", type=float, default=120, help="первые N секунд Сессии")
    args = parser.parse_args()

    wav = args.session / f"{args.user}.wav" if args.user else next(args.session.glob("*.wav"))
    segments = [json.loads(line) for line in wav.with_suffix(".segments.jsonl").open()]
    audio48, sr = sf.read(wav, dtype="float32")
    assert sr == 48_000

    settings = load_settings().speech
    vad = SileroVad().stream()
    turn = SmartTurn()
    detector = TurnDetector(settings)
    utterance: list[np.ndarray] = []  # звук 16 кГц текущей Реплики
    speaking = False

    def apply(actions, t):
        nonlocal utterance
        for a in actions:
            if isinstance(a, CheckTurn):
                p = turn(np.concatenate(utterance))
                complete = p > settings.turn_threshold
                print(
                    f"{t:8.2f}  тишина после {a.t_end:.2f}: Smart Turn {p:.2f} → "
                    f"{'договорил' if complete else 'ждём продолжения'}"
                )
                apply(detector.turn_result(a.request_id, complete), t)
            elif isinstance(a, EndUtterance):
                print(
                    f"{t:8.2f}  === Реплика {a.t_start:.2f}–{a.t_end:.2f} "
                    f"({a.t_end - a.t_start:.1f} с), {a.reason}"
                )
                utterance = []
            elif isinstance(a, Activity) and a.duration_ms % 1000 < 100:
                print(f"{t:8.2f}  говорит {a.duration_ms} мс")

    t_prev = 0.0
    for seg in segments:
        start = seg["start_ms"] / 1000
        if start > args.limit_s:
            break
        # Между фрагментами кадров нет: идёт только время.
        while t_prev + 0.1 < start:
            t_prev += 0.1
            apply(detector.tick(t_prev), t_prev)
        n48 = round(seg["duration_ms"] * 48)
        chunk = downsample_48k_to_16k(audio48[seg["sample_offset"] : seg["sample_offset"] + n48])
        for i in range(0, len(chunk) - WINDOW + 1, WINDOW):
            w = chunk[i : i + WINDOW]
            t0 = start + i / SAMPLE_RATE
            t1 = t0 + WINDOW / SAMPLE_RATE
            p = vad(w)
            if (p > settings.vad_threshold) != speaking:
                speaking = not speaking
                print(f"{t0:8.2f}  {'речь' if speaking else 'тишина'} (VAD {p:.2f})")
            if detector.in_utterance or p > settings.vad_threshold:
                utterance.append(w)
            apply(detector.vad(t0, t1, p), t1)
            t_prev = t1
    for _ in range(20):
        t_prev += 0.1
        apply(detector.tick(t_prev), t_prev)


if __name__ == "__main__":
    main()
