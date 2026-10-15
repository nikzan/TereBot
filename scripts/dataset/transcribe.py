"""Шаг 2 датасета Голоса: автоматическая транскрипция сегментов (GigaAM-v3 e2e RNNT).

    uv run --group data python scripts/dataset/transcribe.py

Дописывает auto_text в data/dataset/review.jsonl для сегментов, у которых его ещё нет.
С --force распознаёт всё заново; вычитанные text и status не меняются никогда.
"""

import argparse
from pathlib import Path

from terebot.dataset import DATASET_DIR
from terebot.dataset.transcribe import GigaAM, transcribe_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--force", action="store_true", help="распознать заново уже распознанные")
    args = ap.parse_args()

    n = transcribe_all(args.dataset, GigaAM(), force=args.force)
    print(f"Распознано сегментов: {n}")


if __name__ == "__main__":
    main()
