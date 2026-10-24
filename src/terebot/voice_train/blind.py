"""Слепой тест Голоса: выбор пар, ключ, обрезка краёв и подсчёт ответов судей.

Пара — живая запись сегмента отложенной выборки и синтез его текста. Какой из файлов A/B
синтез, решает сид. Журнал `used.json` — {имя теста: {model, speaker, seg_ids}}: следующий
тест не берёт сегменты прошлых, а повторная подготовка теста с тем же именем заменяет его запись.
"""

import random
from dataclasses import dataclass

import numpy as np

PASS_SHARE = 0.65
MIN_S, MAX_S = 4.0, 12.0
# Обрезка краёв: рамка 10 мс тише пика на 40 дБ — тишина; по краям речи остаётся 50 мс.
FRAME_MS, SILENCE_DB, MARGIN_MS = 10, -40.0, 50


@dataclass(frozen=True)
class Pair:
    pair: str  # "01"…
    seg_id: str
    text: str
    synthetic: str  # "A" или "B"


@dataclass(frozen=True)
class Score:
    by_judge: dict[str, tuple[int, int]]  # судья → (угадал синтез, ответов)
    correct: int
    total: int

    @property
    def share(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def passed(self) -> bool:
        return self.total > 0 and self.share <= PASS_SHARE


def used_seg_ids(journal: dict[str, dict], except_test: str | None = None) -> set[str]:
    return {s for name, t in journal.items() if name != except_test for s in t["seg_ids"]}


def make_pairs(holdout: list[dict], used: set[str], n: int, seed: int) -> list[Pair]:
    """n сегментов 4–12 с, не бывших в прошлых тестах; выбор и порядок A/B — по сиду."""
    eligible = sorted(
        (r for r in holdout if MIN_S <= r["duration"] <= MAX_S and r["seg_id"] not in used),
        key=lambda r: r["seg_id"],
    )
    if len(eligible) < n:
        raise ValueError(
            f"подходящих сегментов {len(eligible)}, нужно {n}: отложенная выборка исчерпана "
            "прошлыми тестами или в ней мало записей 4–12 с"
        )
    rng = random.Random(seed)
    chosen = rng.sample(eligible, n)
    return [
        Pair(f"{i:02d}", r["seg_id"], r["text"], rng.choice("AB"))
        for i, r in enumerate(chosen, start=1)
    ]


def trim_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    """Ровно 50 мс тишины по краям речи: лишняя срезается (тише пика на 40 дБ), недостающая
    добавляется нулями.

    Одно правило для живой записи и синтеза: у дообученного Голоса ~200/500 мс тишины по краям,
    у живых сегментов ~120 мс, а часть из них шлюз обрезал впритык к речи. Иначе синтез или
    живую запись выдавали бы паузы.
    """
    frame = sr * FRAME_MS // 1000
    n = len(audio) // frame
    if n == 0:
        return audio
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    loud = np.flatnonzero(rms > rms.max() * 10 ** (SILENCE_DB / 20))
    margin = sr * MARGIN_MS // 1000
    start, end = loud[0] * frame - margin, (loud[-1] + 1) * frame + margin
    out = audio[max(0, start) : min(len(audio), end)]
    return np.pad(out, (max(0, -start), max(0, end - len(audio))))


def score(key: list[Pair], answers: dict[str, dict[str, str]]) -> Score:
    """answers: судья → {пара: "A"/"B"}; пустой ответ не считается."""
    truth = {p.pair: p.synthetic for p in key}
    by_judge = {}
    for judge, ans in sorted(answers.items()):
        unknown = set(ans) - set(truth)
        if unknown:
            raise ValueError(f"{judge}: пар {', '.join(sorted(unknown))} нет в ключе")
        given = {k: v.strip().upper() for k, v in ans.items() if v.strip()}
        bad = {k: v for k, v in given.items() if v not in ("A", "B")}
        if bad:
            raise ValueError(f"{judge}: ответ должен быть A или B, а не {bad}")
        by_judge[judge] = (sum(truth[k] == v for k, v in given.items()), len(given))
    return Score(
        by_judge,
        sum(c for c, _ in by_judge.values()),
        sum(t for _, t in by_judge.values()),
    )
