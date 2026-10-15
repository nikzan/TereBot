"""Шаг 4 датасета Голоса: манифесты train и holdout из принятых сегментов и отчёт.

    uv run --group data python scripts/dataset/build.py

Пишет data/dataset/manifest/{train,holdout}.jsonl и data/dataset/report.md и печатает отчёт.
Можно запускать в любой момент вычитки: в манифест попадают только принятые сегменты.
"""

import argparse
from pathlib import Path

from terebot.dataset import DATASET_DIR
from terebot.dataset.build import build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    args = ap.parse_args()
    print(build(args.dataset))


if __name__ == "__main__":
    main()
