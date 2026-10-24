"""Сводная таблица чекпоинтов прогона и выбор чекпоинта по правилу из design.md.

    uv run python scripts/voice_train/curve.py pilot-2026-10-18

Читает data/voice/<прогон>/eval/*/results.json с метриками `terebot.voice_eval --refs` и печатает
таблицу в Markdown (для docs/benchmarks/voice-finetune.md): эпоха, медиана похожести к holdout,
медиана и среднее CER, медианный CER по категориям. Ниже — выбранная эпоха и причины. Окончательно
выбирает владелец, в том числе на слух.
"""

import argparse
import json

from terebot.voice_train import VOICE_DIR
from terebot.voice_train.curve import choose, table


def fmt(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", help="имя прогона: каталог data/voice/<прогон>/")
    args = ap.parse_args()

    eval_dir = VOICE_DIR / args.run / "eval"
    found = {
        p.parent.name: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(eval_dir.glob("*/results.json"))
    }
    if not found:
        raise SystemExit(f"в {eval_dir} нет results.json — сначала synth_eval.py и voice_eval")
    try:
        rows = table(found)
        choice = choose(rows)
    except ValueError as e:
        raise SystemExit(str(e)) from None

    quality = next(iter(found.values()))["quality"]
    categories = sorted({c for r in rows for c in r.cer_median_by_category})
    print(
        f"Эталонов — {len(quality['refs'])}, опорный уровень «запись ↔ запись» "
        f"{fmt(quality['baseline_similarity'])}\n"
    )
    head = ["Модель", "Похожесть", "CER мед.", "CER сред.", *categories]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in rows:
        name = "zero-shot" if r.epoch is None else f"эпоха {r.epoch}"
        if r.epoch is not None and r.epoch == choice.epoch:
            name = f"**{name}**"
        cells = [fmt(r.similarity_median), fmt(r.cer_median), fmt(r.cer_mean)]
        cells += [
            fmt(r.cer_median_by_category[c]) if c in r.cer_median_by_category else "—"
            for c in categories
        ]
        print("| " + " | ".join([name, *cells]) + " |")
    print()
    print(f"Выбор по правилу: {'нет' if choice.epoch is None else f'эпоха {choice.epoch}'}")
    for reason in choice.reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    main()
