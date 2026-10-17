"""Дообучение Голоса: чистая логика без моделей (команды — в scripts/voice_train/).

Раскладка прогона `data/voice/<прогон>/`:
- `audio/<seg_id>.wav` — сегменты обучающей выборки в 24 кГц;
- `train.jsonl` — `{audio, text, ref_audio}` для `prepare_data.py` из QwenLM/Qwen3-TTS;
- `run.json` — параметры прогона: данные, референс, гиперпараметры, версии;
- `checkpoint-epoch-N/` — чекпоинты `sft_12hz.py`.
"""

from pathlib import Path

VOICE_DIR = Path(__file__).resolve().parents[3] / "data" / "voice"
SAMPLE_RATE = 24000
SPEAKER = "prototype"
