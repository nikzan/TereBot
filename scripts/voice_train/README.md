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

## Работа на кластере `lab-code62`

Кластер общий, работа не выходит за каталог `/home/nikzan/qwen`. Репозиторий — `/home/nikzan/qwen/lora`, кэши uv — `/home/nikzan/qwen/.cache/`.

- **Только GPU 1.** На GPU 0 чужой процесс: его не трогаем и карту не занимаем. Все GPU-запуски — с `CUDA_VISIBLE_DEVICES=1`.
- **Окружение.** Перед `uv` в каждой новой оболочке:

  ```bash
  export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/home/nikzan/qwen/.cache/uv UV_PYTHON_INSTALL_DIR=/home/nikzan/qwen/.cache/uv-python
  ```

  `train.sh` сам задаёт `HF_HOME=data/hf`. Для ручных запусков (`check_objective.py`, `check_stop.py`) его нужно задать самому: `HF_HOME=$PWD/data/hf`.
- **Имя процесса.** GPU-процессы запускаются через `.venv/bin/qwen-lora`, а не через `uv run`. Симлинк создаёт `train.sh`.
- **Длинные запуски** — через `setsid nohup`, с логом в каталоге прогона. Так процесс переживает выход из ssh, и `ssh lab-code62 '…'` не ждёт его завершения:

  ```bash
  (REPO_COMMIT=<коммит> CUDA_VISIBLE_DEVICES=1 setsid nohup scripts/voice_train/train.sh pilot-2026-10-18 > data/voice/pilot-2026-10-18/train.out 2>&1 < /dev/null &)
  ```

- **Код переносится с Mac rsync-ом.** Git-клон на кластере есть, но устарел, поэтому коммит передаётся переменной `REPO_COMMIT`. `train.sh` берёт его из этой переменной, а без неё — из `git rev-parse HEAD`, но только при чистом дереве; иначе в `run.json` пишется `null`. С Mac:

  ```bash
  rsync -a --exclude .venv --exclude data --exclude node_modules --exclude .git ./ lab-code62:/home/nikzan/qwen/lora/
  echo "REPO_COMMIT=$(git rev-parse HEAD)"
  ```

  На кластере эту строку ставят перед командой: `REPO_COMMIT=<коммит> CUDA_VISIBLE_DEVICES=1 scripts/voice_train/train.sh …`. Незакоммиченные правки в коммит не попадают: перед прогоном, по которому будут выбирать Голос, закоммитьте код.

`sft_12hz.py` зашивает `attn_implementation="flash_attention_2"`. Пакет `flash-attn` не ставим (на PyPI только исходники): transformers в этом случае сам берёт готовое ядро `kernels-community/flash-attn` с HF через пакет `kernels`, в логе это строка `using kernels-community/flash-attn from the kernels library`. Поэтому в группе `train` закреплён torch 2.11 — под него на хабе есть сборка `torch211-cxx11-cu130`.

## Прогон

**1. Данные.** Ресемплинг обучающей выборки в 24 кГц, `train.jsonl` и `run.json`. Пути в `train.jsonl` абсолютные, поэтому шаг выполняется на кластере.

```bash
uv run --group train python scripts/voice_train/prepare.py --run pilot-2026-10-17 --label pilot
```

Референс по умолчанию — самый длинный сегмент обучающей выборки. Другой — `--ref <seg_id>`; сегмент отложенной выборки не принимается.

К каждому сегменту и референсу добавляется тишина: 150 мс в начале и 300 мс в конце (`--pad-start-ms`, `--pad-end-ms`). Шлюз режет речь по микрофону почти впритык, и без отступов Голос обрезает начало и конец фраз. Отступы пишутся в `run.json` (`pad_ms`). Отложенная выборка копируется в каталог прогона как есть (`holdout/`, `holdout.jsonl`): прогон оценивается на ней, даже если датасет потом вырастет.

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
  audio/<seg_id>.wav        сегменты train, 24 кГц, моно, 16 бит, тишина 150 мс / 300 мс по краям
  holdout/<seg_id>.wav      отложенная выборка прогона, исходные WAV датасета
  holdout.jsonl             её тексты, пути относительно каталога прогона
  train.jsonl               {audio, text, ref_audio}, один референс на все примеры
  train.codes.jsonl         то же с audio_codes
  run.json                  метка pilot/final, минуты, референс, гиперпараметры, коммит QwenLM, версии, GPU
  train.log                 лог обучения (loss каждые 10 шагов)
  checkpoint-epoch-N/       полная модель custom_voice, говорящий prototype (~4–5 ГБ)
  eval/<чекпоинт>/          синтез набора оценки и results.json (synth_eval.py, метрики voice_eval)
```

Чекпоинт вызывается через `generate_custom_voice(text, speaker="prototype")` без референса. Невыбранные чекпоинты удаляйте после оценки: 10 эпох — это ~45 ГБ.

## Проверки

**Цель обучения без обучения.** Запускается один раз на прогон, на кодах из `train.codes.jsonl`. Нужна после смены коммита QwenLM или патча.

```bash
CUDA_VISIBLE_DEVICES=1 HF_HOME=$PWD/data/hf .venv/bin/qwen-lora scripts/voice_train/check_objective.py pilot-2026-10-17
```

Скрипт печатает CE talker и sub-talker на Base для исходной и исправленной цели и `ln(2048)` для сравнения. Если исправленная цель не ниже случайной и исходной, он завершается с ошибкой: тогда обучать нельзя.

**Остановка после обучения.** Каждый чекпоинт и Base синтезируют фразы из `config/eval/stop_phrases_ru.txt` с лимитом 400 токенов:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HOME=$PWD/data/hf .venv/bin/qwen-lora scripts/voice_train/check_stop.py pilot-2026-10-18
```

Скрипт печатает таблицу длительностей; `*` — синтез дошёл до лимита. Чекпоинт непригоден, если хотя бы на одной фразе он не остановился сам или длительность вне 0,5–2× от Base. Такой чекпоинт на оценку похожести и CER не идёт. WAV и `summary.json` — в `data/voice/<прогон>/stop_check/`, `--epochs 0 4 9` ограничивает набор эпох.

**Кривая на отложенной выборке.** CE исправленной цели на holdout датасета и на стольких же клипах train, для Base и каждого чекпоинта. Коды holdout готовятся при первом запуске:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HOME=$PWD/data/hf .venv/bin/qwen-lora scripts/voice_train/check_holdout.py pilot-2026-10-18
```

Рост CE на holdout при падении на train — признак переобучения. Чекпоинт по этой кривой не выбирают: для этого есть похожесть и CER.

## Оценка чекпоинтов

**Синтез на кластере.** Тексты отложенной выборки прогона и `config/eval/phrases_ru.txt` для выбранных чекпоинтов и для zero-shot Base с тем же референсом:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HOME=$PWD/data/hf .venv/bin/qwen-lora scripts/voice_train/synth_eval.py <прогон> --epochs 3 5 --zero-shot
```

Один процесс синтезирует ~10 фраз в минуту (RTF ≈ 1,5): он упирается в задержку шагов, а не в вычисления, и `utilization.gpu` = 99% этого не показывает — смотрите фразы в минуту и мощность. Для полного прогона запускайте несколько процессов с общей очередью `--claim`: каждый берёт следующую свободную модель, их можно добавлять и останавливать на ходу. По 5 на карту — ~27 фраз в минуту на карту и ~24 ГБ. Замеры и расчёт на большие датасеты — в design.md, «Скорость синтеза оценки и масштабирование».

```bash
for i in 1 2 3 4 5; do
  (CUDA_VISIBLE_DEVICES=1 HF_HOME=$PWD/data/hf setsid nohup .venv/bin/qwen-lora scripts/voice_train/synth_eval.py <прогон> --zero-shot --claim > data/voice/<прогон>/synth_eval-$i.out 2>&1 < /dev/null &)
done
```

Повторный запуск с `--claim` пропустит все модели: сначала удалите `data/voice/<прогон>/eval/.*.claim`.

Ожидая процесс по `pgrep -f <строка>`, не повторяйте эту строку в команде, которая ждёт: `pgrep` найдёт саму эту команду и не дождётся никогда. Надёжнее запоминать PID.

У прогонов, подготовленных до фиксации отложенной выборки, нужен `--holdout <манифест>`. У `pilot-2026-10-18` это `data/voice/pilot-2026-10-18/dataset-manifest/holdout.jsonl`: копия манифеста, на котором он обучен.

**Метрики на Mac.** После rsync `eval/` и `holdout/` в `data/voice/<прогон>/`:

```bash
uv run --group eval python -m terebot.voice_eval data/voice/<прогон>/eval/* --refs data/voice/<прогон>/holdout
uv run python scripts/voice_train/curve.py <прогон>
```

`curve.py` печатает таблицу в Markdown для отчёта и эпоху по правилу из design.md: медианный CER не выше zero-shot, средний — не больше чем на 0,02 выше, затем самая ранняя эпоха плато похожести (< 0,01 за эпоху).

## Слепой тест

Судьи, которые знают голос Прототипа, но не слышали эту версию Голоса, слушают пары «живая запись / синтез того же текста» и отмечают, где синтез. Тест пройден, если в сумме синтез угадан не чаще 65% ответов. Всё на Mac:

```bash
uv run --exact --group tts-mlx --with pyloudnorm==0.2.0 python scripts/voice_train/blind_test.py prepare --name final-e8 --run final-2026-10-24 --model data/models/final-2026-10-24-e8-mlx-8bit
```

Группа `data` здесь не подходит: её silero-vad тянет torch 2.14.1 без sympy на macOS, и рядом с `tts-mlx` transformers падает на импорте токенизатора. Тесту из неё нужна только pyloudnorm, а `--exact` убирает torch, оставшийся от других групп.

- 20 сегментов 4–12 с из отложенной выборки прогона (`holdout/`), не бывших в прошлых тестах (журнал `data/voice/blind_tests/used.json`).
- Синтез — как в Клоне (`MlxVoice`, стриминг 0,5 с, говорящий `prototype`), затем Opus 96 кбит/с туда и обратно: живые записи прошли тот же кодек Discord.
- Оба файла пары обрабатываются одинаково: ровно 50 мс тишины по краям (лишняя срезается, недостающая добавляется — иначе выдали бы паузы), −23 LUFS, WAV 48 кГц 16 бит без метаданных.
- `data/voice/blind_tests/<имя>/pairs/` — `NN-A.wav`, `NN-B.wav` и `answers.csv`: эту папку отдают судьям. Ключ `key.json` лежит уровнем выше, судьям его не показывать.
- Другая эпоха — тот же `--name` с другим `--model`: тест подготовится заново, а его сегменты в журнале заменятся. Новое имя — новые сегменты.

**Тест по стриму экрана** (один компьютер, судьи слушают в Discord) — ответы вводит ведущий:

```bash
uv run --exact --group tts-mlx --with pyloudnorm==0.2.0 python scripts/voice_train/blind_test.py listen --name final-e8 --judge аня
```

Пара проигрывается сама: A, пауза, B. Ведущий нажимает `A` или `B` (в русской раскладке — те же клавиши), ответ сразу пишется в `data/voice/blind_tests/final-e8/answers-аня.csv`, начинается следующая пара. `1` / `2` — переслушать A / B, пробел — обе, `←` — назад (ответ можно исправить), `q` — выход; повторный запуск продолжит с первой пары без ответа. Нажатия во время звука не засчитываются — отвечать после. Ключ не читается, на экране только номер пары и ответ судьи. Звук в стриме идёт через ещё один кодек Discord — одинаково для A и B. Проверьте до теста, что Discord на macOS передаёт звук стрима.

Или по файлам: каждый судья заполняет копию `answers.csv` (колонка `synthetic`: `A` или `B`) и присылает её как `answers-<имя>.csv`. Подсчёт и запись итога в `docs/benchmarks/voice-finetune.md`:

```bash
uv run --exact --group tts-mlx --with pyloudnorm==0.2.0 python scripts/voice_train/blind_test.py score --name final-e8 data/voice/blind_tests/final-e8/answers-*.csv
```

## Патч к QwenLM

На закреплённом коммите `sft_12hz.py` учит модель другой задаче, чем та, которую решает синтез (QwenLM/Qwen3-TTS#371 и #337, PR #278 и #384). Без исправления чекпоинты не выдают EOS и синтезируют до `max_new_tokens`: так было в пробном прогоне 2026-10-17 на всех эпохах. Разбор — в [design.md change `fix-voice-sft-alignment`](../../openspec/changes/archive/2026-10-18-fix-voice-sft-alignment/design.md).

Исправление — [patches/sft_12hz-alignment.patch](patches/sft_12hz-alignment.patch), правит только `finetuning/sft_12hz.py`:
- текст проходит `text_projection`;
- talker получает всю последовательность с `labels=codec_0_labels`, единственный сдвиг делает loss, EOS остаётся в цели;
- sub-talker получает состояние предыдущего шага (`hidden_states[:, :-1]` по `codec_mask[:, 1:]`);
- loss sub-talker считается `cross_entropy` по логитам без второго сдвига.

Пакет `qwen-tts` в `.venv` не правится.

`train.sh` перед обучением проверяет клон в `data/vendor/Qwen3-TTS`:
- патч уже наложен — пишет «патч выравнивания уже наложен» и идёт дальше;
- клон чистый — накладывает патч;
- иначе — ошибка «код QwenLM … не совпадает с ожидаемым», обучение не начинается.

Чужие правки скрипт не стирает: сброс — `git -C data/vendor/Qwen3-TTS checkout .`. В `run.json` пишутся `train.qwen3_tts_patch` (sha256 файла патча) и `train.repo_commit`.

**Когда удалить патч.** Когда upstream вольёт эквивалентное исправление (следим за #278 и #384), закреплённый коммит в `docs/REFERENCES.md` переводится на новый, а патч удаляется. После этого `check_objective.py` и `check_stop.py` повторяются на пробном запуске.
