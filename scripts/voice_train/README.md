# Дообучение Голоса

Дообучает Qwen3-TTS-12Hz-1.7B-Base на вычитанном датасете Прототипа ([scripts/dataset/](../dataset/README.md)) официальным скриптом `finetuning/` из [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) и даёт чекпоинт с говорящим `prototype`. Решения и пороги — в [openspec/changes/finetune-voice/design.md](../../openspec/changes/finetune-voice/design.md) (после архивации — `openspec/specs/voice-training/`), логика без моделей — в `src/terebot/voice_train/`.

Обучение — на CUDA-кластере (одна карта 32 ГБ), все команды — из корня репозитория.

## Подготовка кластера (один раз)

```bash
git clone <репозиторий> lora && cd lora
uv sync --group train
git clone https://github.com/QwenLM/Qwen3-TTS.git data/vendor/Qwen3-TTS
git -C data/vendor/Qwen3-TTS checkout <коммит из docs/REFERENCES.md>
```

Датасет переносится с Mac как в [scripts/dataset/README.md](../dataset/README.md#перенос-на-кластер). Модели и токенизатор скачиваются при первом запуске в `data/models/` и `data/hf/` (`HF_HOME`), поэтому общий `~/.cache` не трогается. Если и кэш uv должен лежать рядом, задайте `UV_CACHE_DIR` и `UV_PYTHON_INSTALL_DIR` до `uv sync`.

Кластер общий: перед запуском найдите свободную карту в `nvidia-smi` и передайте её в `CUDA_VISIBLE_DEVICES`. Без этой переменной `train.sh` не стартует. Внутри процесса выбранная карта видна как `cuda:0`.

Процессы на GPU запускаются интерпретатором venv под обезличенным именем `qwen-lora` (симлинк `.venv/bin/qwen-lora`, имя меняет `PROC_NAME`), а не через `uv run`: так они выглядят в `nvidia-smi` и `htop`. Имя каталога репозитория на кластере тоже видно в командной строке, поэтому клонируйте его под нейтральным именем, например `git clone <репозиторий> lora`.

`sft_12hz.py` зашивает `attn_implementation="flash_attention_2"`. Пакет `flash-attn` не ставим (на PyPI только исходники): transformers в этом случае сам берёт готовое ядро `kernels-community/flash-attn` с HF через пакет `kernels`, в логе это строка `using kernels-community/flash-attn from the kernels library`. Поэтому в группе `train` закреплён torch 2.11 — под него на хабе есть сборка `torch211-cxx11-cu130`.

## Прогон

**1. Данные.** Ресемплинг обучающей выборки в 24 кГц, `train.jsonl` и `run.json`. Пути в `train.jsonl` абсолютные, поэтому шаг выполняется на кластере.

```bash
uv run --group train python scripts/voice_train/prepare.py --run pilot-2026-10-17 --label pilot
```

Референс по умолчанию — самый длинный сегмент обучающей выборки. Другой — `--ref <seg_id>`; сегмент отложенной выборки не принимается.

**2. Обучение.** `prepare_data.py` (коды токенизатора 12 Гц) и `sft_12hz.py`: bf16, AdamW, накопление градиента 4, чекпоинт на каждую эпоху. По умолчанию 10 эпох, батч 2, `lr=2e-5`.

```bash
CUDA_VISIBLE_DEVICES=1 scripts/voice_train/train.sh pilot-2026-10-17
```

Длинный прогон запускайте в `tmux` или через `nohup`: лог пишется в `data/voice/<прогон>/train.log`.

Пробный запуск цепочки — на 10 примерах и 1 эпохе, в отдельном прогоне:

```bash
uv run --group train python scripts/voice_train/prepare.py --run smoke --label pilot
CUDA_VISIBLE_DEVICES=1 scripts/voice_train/train.sh smoke --limit 10 --epochs 1
```

Флаги `train.sh`: `--epochs`, `--batch-size` (при нехватке памяти — 1), `--lr`, `--limit`.

## Что получается

```
data/voice/<прогон>/
  audio/<seg_id>.wav        сегменты train, 24 кГц, моно, 16 бит
  train.jsonl               {audio, text, ref_audio}, один референс на все примеры
  train.codes.jsonl         то же с audio_codes
  run.json                  метка pilot/final, минуты, референс, гиперпараметры, коммит QwenLM, версии, GPU
  train.log                 лог обучения (loss каждые 10 шагов)
  checkpoint-epoch-N/       полная модель custom_voice, говорящий prototype (~4–5 ГБ)
```

Чекпоинт вызывается через `generate_custom_voice(text, speaker="prototype")` без референса. Невыбранные чекпоинты удаляйте после оценки: 10 эпох — это ~45 ГБ.

## Известная проблема

На закреплённом коммите QwenLM `sft_12hz.py` учит модель с ошибками выравнивания меток (QwenLM/Qwen3-TTS#371, PR #278 и #384): чекпоинты не выдают EOS и синтезируют до `max_new_tokens`. Воспроизведено в пробном прогоне 2026-10-17 на всех эпохах. Пока это не исправлено, чекпоинты непригодны для оценки; решение — change `fix-voice-sft-alignment`.
