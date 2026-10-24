"""Сводная таблица чекпоинтов прогона и правило выбора чекпоинта.

Строка таблицы — один каталог `eval/<чекпоинт>/results.json` с метриками `terebot.voice_eval`.
Каталог `zero-shot` — Base с тем же референсом, остальные называются по эпохе:
`checkpoint-epoch-N`.
"""

import itertools
import re
import statistics
from dataclasses import dataclass

ZERO_SHOT = "zero-shot"
# Похожесть, которая растёт медленнее, — плато.
PLATEAU_GAIN_PER_EPOCH = 0.01
# Средний CER может быть выше, чем у zero-shot, не больше чем на столько.
CER_MEAN_SLACK = 0.02


@dataclass(frozen=True)
class Row:
    name: str
    epoch: int | None  # None — zero-shot
    similarity_median: float
    cer_median: float
    cer_mean: float
    cer_median_by_category: dict[str, float]


@dataclass(frozen=True)
class Choice:
    epoch: int | None  # None — ни один чекпоинт не прошёл правило
    reasons: list[str]


def epoch_of(name: str) -> int | None:
    if name == ZERO_SHOT:
        return None
    m = re.fullmatch(r"(?:checkpoint-)?epoch-(\d+)", name)
    if not m:
        raise ValueError(f"{name}: ожидается {ZERO_SHOT} или checkpoint-epoch-N")
    return int(m.group(1))


def row(name: str, results: dict) -> Row:
    q = results.get("quality")
    if q is None:
        raise ValueError(f"{name}: нет метрик, сначала python -m terebot.voice_eval --refs")
    return Row(
        name=name,
        epoch=epoch_of(name),
        similarity_median=q["similarity_median"],
        cer_median=q["cer_median"],
        cer_mean=statistics.fmean(p["cer"] for p in results["phrases"]),
        cer_median_by_category=q["cer_median_by_category"],
    )


def table(results_by_name: dict[str, dict]) -> list[Row]:
    """Zero-shot первой строкой, затем чекпоинты по возрастанию эпохи."""
    rows = [row(n, r) for n, r in results_by_name.items()]
    return sorted(rows, key=lambda r: -1 if r.epoch is None else r.epoch)


def choose(rows: list[Row]) -> Choice:
    """Правило из design.md `finetune-voice`.

    Чекпоинт допускается, если медианный CER не выше, чем у zero-shot, а средний — не больше
    чем на 0,02 выше. Среди допущенных выбирается самая ранняя эпоха, после которой медиана
    похожести растёт меньше чем на 0,01 за эпоху (до следующей допущенной). Если плато нет —
    последняя допущенная. Окончательно выбирает владелец.
    """
    zero = next((r for r in rows if r.epoch is None), None)
    if zero is None:
        raise ValueError(f"в таблице нет строки {ZERO_SHOT}")
    reasons = []
    allowed = []
    for r in (r for r in rows if r.epoch is not None):
        if r.cer_median > zero.cer_median:
            reasons.append(
                f"эпоха {r.epoch}: медианный CER {r.cer_median:.3f} выше zero-shot "
                f"{zero.cer_median:.3f}"
            )
        elif r.cer_mean > zero.cer_mean + CER_MEAN_SLACK:
            reasons.append(
                f"эпоха {r.epoch}: средний CER {r.cer_mean:.3f} выше zero-shot "
                f"{zero.cer_mean:.3f} больше чем на {CER_MEAN_SLACK}"
            )
        else:
            allowed.append(r)
    if not allowed:
        return Choice(None, [*reasons, "ни один чекпоинт не прошёл по CER"])
    for cur, nxt in itertools.pairwise(allowed):
        gain = (nxt.similarity_median - cur.similarity_median) / (nxt.epoch - cur.epoch)
        if gain < PLATEAU_GAIN_PER_EPOCH:
            reasons.append(
                f"эпоха {cur.epoch}: дальше похожесть растёт на {gain:+.3f} за эпоху "
                f"(до эпохи {nxt.epoch}) — плато"
            )
            return Choice(cur.epoch, reasons)
    last = allowed[-1]
    reasons.append(f"эпоха {last.epoch}: плато не достигнуто, последняя допущенная")
    return Choice(last.epoch, reasons)
