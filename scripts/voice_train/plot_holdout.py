"""График кривой check_holdout.py: CE на holdout и train по эпохам (PNG для отчёта).

    uv run --no-project --with matplotlib python scripts/voice_train/plot_holdout.py \\
        data/voice/pilot-2026-10-18/holdout_ce.json docs/benchmarks/img/pilot-2026-10-18-holdout-ce.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = json.load(open(sys.argv[1], encoding="utf-8"))
ep = sorted(int(k.split("-")[1]) for k in r if k != "base")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HOLD, TRAIN, BASE = "#2a78d6", "#eb6834", "#8a8984"
panels = [("loss", "Loss = talker + 0,3·sub-talker"), ("talker", "Talker (код 0 и EOS)"), ("sub-talker", "Sub-talker (коды 1–15)")]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURF)
for ax, (key, title) in zip(axes, panels):
    ax.set_facecolor(SURF)
    h = [r[f"epoch-{e}"]["holdout"][key] for e in ep]
    t = [r[f"epoch-{e}"]["train"][key] for e in ep]
    b = r["base"]["holdout"][key]
    ax.axvspan(4.6, 5.4, color="#2a78d6", alpha=0.07, lw=0)
    ax.axhline(b, color=BASE, lw=1.2, ls=(0, (4, 3)))
    ax.text(ep[-1], b, "Base, holdout", color=INK2, fontsize=8.5, ha="right", va="bottom")
    ax.plot(ep, h, color=HOLD, lw=2, marker="o", ms=4.5, mec=SURF, mew=1.5, label="holdout (не видела)")
    ax.plot(ep, t, color=TRAIN, lw=2, marker="o", ms=4.5, mec=SURF, mew=1.5, label="train (обучалась)")
    i = min(range(len(h)), key=h.__getitem__)
    ax.annotate(f"мин. {h[i]:.2f}".replace(".", ","), (ep[i], h[i]), xytext=(6, -14), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xticks(ep); ax.set_xlabel("эпоха", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
axes[0].set_ylabel("CE", color=INK2, fontsize=9)
axes[0].text(5, axes[0].get_ylim()[1], "эпоха 5 —\nлучшая на слух", color=INK2, fontsize=8, ha="center", va="top")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper right", frameon=False, fontsize=9, ncol=2, labelcolor=INK)
fig.suptitle("pilot-2026-10-18: CE исправленной цели по эпохам, holdout 26 клипов vs train 26 клипов", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(sys.argv[2], dpi=160, facecolor=SURF)
