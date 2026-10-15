"""Датасет Голоса: записи шлюза с речью Прототипа → вычитанные сегменты с транскриптами.

Шаги (каждый — скрипт в scripts/dataset/, запуск в группе uv `data`):
segment → transcribe → review → build. Порядок запуска — scripts/dataset/README.md.

Раскладка `data/dataset/`:
- `segments.jsonl` — все сегменты нарезки, включая отбракованные с причиной;
- `segments/<seg_id>.wav` — аудио годных сегментов;
- `review.jsonl` — автоматический и вычитанный текст, статус; единственный файл с ручным трудом;
- `manifest/{train,holdout}.jsonl`, `report.md` — результат сборки.
"""

from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[3] / "data" / "dataset"
RECORDINGS_DIR = Path(__file__).resolve().parents[3] / "data" / "recordings"
