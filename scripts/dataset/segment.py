"""Шаг 1 датасета Голоса: нарезка записей шлюза с речью Прототипа на сегменты 3–15 с.

    uv run --group data python scripts/dataset/segment.py --prototype <Discord ID Прототипа>

Нарезает Сессии из data/recordings/, которых ещё нет в data/dataset/segments.jsonl.
С --force нарезает всё заново; вычитка в review.jsonl при этом не теряется.
"""

import argparse
import collections
from pathlib import Path

from terebot.dataset import DATASET_DIR, RECORDINGS_DIR
from terebot.dataset.jsonl import read_jsonl
from terebot.dataset.segment import SileroVad, segment_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--prototype", required=True, help="Discord user ID Прототипа")
    ap.add_argument("--recordings", type=Path, default=RECORDINGS_DIR)
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--force", action="store_true", help="нарезать заново уже нарезанные Сессии")
    args = ap.parse_args()

    sessions = segment_all(
        args.recordings, args.dataset, args.prototype, SileroVad(), force=args.force
    )
    rows = [r for r in read_jsonl(args.dataset / "segments.jsonl") if r["session"] in sessions]
    good = [r for r in rows if not r["reject"]]
    print(f"Нарезано Сессий: {len(sessions)}")
    print(f"Годных сегментов: {len(good)}, {sum(r['duration'] for r in good) / 60:.1f} мин")
    for reason, n in collections.Counter(r["reject"] for r in rows if r["reject"]).most_common():
        print(f"Отбраковано «{reason}»: {n}")


if __name__ == "__main__":
    main()
