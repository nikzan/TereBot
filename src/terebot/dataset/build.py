"""Сборка датасета: манифесты train и holdout из принятых сегментов и отчёт.

Отложенная выборка (holdout) набирается целыми блоками — Сессия × 2-минутное окно времени Сессии
(`start_ms`), а не позиции в WAV: там тишина вырезана, и соседство обманчиво. Блоки упорядочены по
хешу «Сессия:окно» и берутся по порядку, пока не наберётся ~10% длительности. Разбиение зависит
только от набора принятых сегментов, поэтому повторная сборка даёт те же файлы.
"""

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from terebot.dataset.jsonl import read_jsonl, write_jsonl, write_text
from terebot.dataset.review import ACCEPTED, REJECTED, Review, latin_words

BLOCK_MS = 120_000
HOLDOUT_SHARE = 0.10
TARGET_MIN = (30, 60)
HIST_BINS = (3, 5, 7, 9, 11, 13, 15)


def block_of(item: dict) -> tuple[str, int]:
    return item["session"], item["start_ms"] // BLOCK_MS


def _block_hash(block: tuple[str, int]) -> str:
    return hashlib.sha256(f"{block[0]}:{block[1]}".encode()).hexdigest()


def split_holdout(items: list[dict], share: float = HOLDOUT_SHARE) -> tuple[list[dict], list[dict]]:
    """(train, holdout). Блок, который раздул бы holdout больше 2×share, пропускается:
    иначе на малом датасете в holdout уйдёт почти всё."""
    blocks = defaultdict(float)
    for it in items:
        blocks[block_of(it)] += it["duration"]
    total = sum(blocks.values())
    chosen, acc = set(), 0.0
    for b in sorted(blocks, key=_block_hash):
        if acc >= share * total:
            break
        if acc + blocks[b] <= 2 * share * total:
            chosen.add(b)
            acc += blocks[b]
    train = [it for it in items if block_of(it) not in chosen]
    holdout = [it for it in items if block_of(it) in chosen]
    return train, holdout


def manifest_items(segments: list[dict], review: Review) -> list[dict]:
    by_id = {s["seg_id"]: s for s in segments}
    return [
        {
            "audio": by_id[e["seg_id"]]["audio"],
            "text": e["text"],
            "duration": by_id[e["seg_id"]]["duration"],
            "session": by_id[e["seg_id"]]["session"],
            "seg_id": e["seg_id"],
            "start_ms": by_id[e["seg_id"]]["start_ms"],
        }
        for e in review.current()
        if e["status"] == ACCEPTED
    ]


def build(dataset_dir: Path) -> str:
    """Пишет manifest/{train,holdout}.jsonl и report.md; возвращает текст отчёта."""
    segments = read_jsonl(dataset_dir / "segments.jsonl")
    review = Review(dataset_dir / "review.jsonl", segments)
    train, holdout = split_holdout(manifest_items(segments, review))
    # start_ms нужен только для блоков; в манифесте — формат из design.md.
    strip = [{k: v for k, v in it.items() if k != "start_ms"} for it in train + holdout]
    write_jsonl(dataset_dir / "manifest" / "train.jsonl", strip[: len(train)])
    write_jsonl(dataset_dir / "manifest" / "holdout.jsonl", strip[len(train) :])
    text = report(segments, review, train, holdout)
    write_text(dataset_dir / "report.md", text)
    return text


def _min(items: list[dict]) -> float:
    return sum(it["duration"] for it in items) / 60


def report(segments: list[dict], review: Review, train: list[dict], holdout: list[dict]) -> str:
    accepted = train + holdout
    total = _min(accepted)
    lo, hi = TARGET_MIN
    verdict = "цель достигнута" if total >= lo else f"до цели не хватает {lo - total:.1f} мин"
    lines = [
        "# Датасет Голоса",
        "",
        f"Принятой речи: **{total:.1f} мин** (сегментов: {len(accepted)}), цель {lo}–{hi} мин — "
        f"{verdict}.",
        f"Обучающая выборка: {_min(train):.1f} мин (сегментов: {len(train)}), "
        f"отложенная: {_min(holdout):.1f} мин (сегментов: {len(holdout)}).",
        "",
        "## По Сессиям",
        "",
        "| Сессия | Принято, мин | В отложенной, мин | Сегментов |",
        "|---|---|---|---|",
    ]
    for session in sorted({s["session"] for s in segments}):
        acc = [it for it in accepted if it["session"] == session]
        hold = [it for it in holdout if it["session"] == session]
        lines.append(f"| {session} | {_min(acc):.1f} | {_min(hold):.1f} | {len(acc)} |")

    lines += [
        "",
        "## Длительности принятых сегментов",
        "",
        "| Длительность | Сегментов | |",
        "|---|---|---|",
    ]
    for a, b in zip(HIST_BINS, HIST_BINS[1:], strict=False):
        n = sum(
            a <= it["duration"] < b or (b == HIST_BINS[-1] and it["duration"] == b)
            for it in accepted
        )
        lines.append(f"| {a}–{b} с | {n} | {'█' * n} |")

    rejected = Counter(("нарезка", s["reject"]) for s in segments if s["reject"])
    rejected.update(("вычитка", e["reason"]) for e in review.current() if e["status"] == REJECTED)
    lines += ["", "## Отбраковка", ""]
    if rejected:
        lines += ["| Шаг | Причина | Сегментов |", "|---|---|---|"]
        lines += [f"| {step} | {reason} | {n} |" for (step, reason), n in sorted(rejected.items())]
    else:
        lines.append("Отбракованных нет.")

    latin = [it for it in accepted if latin_words(it["text"])]
    if latin:
        lines += ["", "## Латиница в принятых текстах", ""]
        lines.append(
            "По руководству всё пишется кириллицей; исправьте в review.jsonl или проверьте."
        )
        lines.append("")
        lines += [f"- `{it['seg_id']}`: {' '.join(latin_words(it['text']))}" for it in latin]

    counts = review.counts()
    queued = sum(counts.values())
    checked = counts[ACCEPTED] + counts[REJECTED]
    share = 100 * checked / queued if queued else 100
    lines += [
        "",
        "## Вычитка",
        "",
        f"Проверено {checked} из {queued} годных сегментов ({share:.0f}%): "
        f"принято {counts[ACCEPTED]}, отбраковано {counts[REJECTED]}, "
        f"не проверено {queued - checked}.",
        "",
    ]
    return "\n".join(lines)
