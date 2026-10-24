"""График кривой прогона по эпохам: похожесть, средний CER и CE на отложенной выборке.

    uv run --no-project --with matplotlib python scripts/voice_train/plot_curve.py \\
        final-2026-10-24 docs/benchmarks/img/final-2026-10-24-curve.png

Читает data/voice/<прогон>/eval/*/results.json (метрики voice_eval --refs) и holdout_ce.json
(check_holdout.py). Пунктир — zero-shot (у CE — Base), точки — «запись ↔ запись».
"""

import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES, REF = "#2a78d6", "#8a8984"


def main() -> None:
    run, out = sys.argv[1], Path(sys.argv[2])
    run_dir = Path(__file__).resolve().parents[2] / "data" / "voice" / run
    rows = {}
    for p in (run_dir / "eval").glob("*/results.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        rows[p.parent.name] = (
            r["quality"]["similarity_median"],
            statistics.fmean(x["cer"] for x in r["phrases"]),
            r["quality"]["baseline_similarity"],
        )
    zero = rows.pop("zero-shot")
    ep = sorted(int(re.search(r"\d+$", k).group()) for k in rows)
    sim = [rows[f"checkpoint-epoch-{e}"][0] for e in ep]
    cer = [rows[f"checkpoint-epoch-{e}"][1] for e in ep]
    ce = json.loads((run_dir / "holdout_ce.json").read_text(encoding="utf-8"))
    hold = [ce[f"epoch-{e}"]["holdout"]["loss"] for e in ep]

    panels = [
        ("Похожесть к holdout, медиана", sim, zero[0], "zero-shot", zero[2]),
        ("CER, среднее по 111 фразам", cer, zero[1], "zero-shot", None),
        ("CE на holdout (loss)", hold, ce["base"]["holdout"]["loss"], "Base", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURF)
    for ax, (title, ys, ref, ref_name, rec) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURF)
        ax.axhline(ref, color=REF, lw=1.2, ls=(0, (4, 3)))
        ax.text(ep[-1], ref, ref_name, color=INK2, fontsize=8.5, ha="right", va="bottom")
        if rec is not None:
            ax.axhline(rec, color=REF, lw=1.2, ls=(0, (1, 2)))
            ax.text(ep[0], rec, "запись ↔ запись", color=INK2, fontsize=8.5, va="bottom")
        ax.plot(ep, ys, color=SERIES, lw=2, marker="o", ms=4.5, mec=SURF, mew=1.5)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xticks(ep)
        ax.set_xlabel("эпоха", color=INK2, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5)
    fig.suptitle(f"Прогон {run} по эпохам", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURF)
    print(out)


if __name__ == "__main__":
    main()
