# Tasks

## 1. Патч цели обучения

- [x] 1.1 Написать `scripts/voice_train/patches/sft_12hz-alignment.patch` к `finetuning/sft_12hz.py` на `022e286` по design.md, «Что меняет патч»: `text_projection`, talker без ручного сдвига с `labels=codec_0_labels`, состояния sub-talker `hidden_states[:, :-1]` по `codec_mask[:, 1:]`, loss sub-talker через `cross_entropy` по логитам без сдвига; проверка: на кластере `git -C data/vendor/Qwen3-TTS apply --check` проходит на чистом клоне, а после наложения `.venv/bin/qwen-lora -m py_compile finetuning/sft_12hz.py` и `--help` работают
- [x] 1.2 Встроить патч в `scripts/voice_train/train.sh`: проверка «уже наложен / накладываем / не совпадает», отказ при несовпадении, `qwen3_tts_patch` (sha256) и коммит репозитория (`git rev-parse HEAD` или `REPO_COMMIT`) в `run.json`; проверка: `bash -n`; на кластере первый запуск накладывает патч, второй сообщает «уже наложен», а при вручную изменённом `sft_12hz.py` скрипт завершается с ошибкой до загрузки модели
- [x] 1.3 Обновить `scripts/voice_train/README.md`: раздел о работе с кластером (каталог `/home/nikzan/qwen/lora`, `CUDA_VISIBLE_DEVICES=1`, имя процесса `qwen-lora`, переменные uv и HF, `nohup`, rsync кода и `REPO_COMMIT`) и раздел «Известная проблема» → «Патч к QwenLM» с условием удаления; проверка: команды из README выполнены на кластере при задачах 2–3 без отступлений

## 2. Проверка цели без обучения

- [x] 2.1 Реализовать `scripts/voice_train/check_objective.py`: Base, N клипов из `train.codes.jsonl` прогона, батчи через `TTSDataset.collate_fn` из vendor, CE talker и sub-talker для исходной и исправленной цели, `ln(2048)` в таблице; проверка: запуск на кластере (GPU 1) печатает таблицу 2×2 без ошибок
- [x] 2.2 Прогнать `check_objective.py` на `pilot-2026-10-17` (16 клипов) и записать таблицу в `docs/benchmarks/voice-finetune.md` (раздел «Ошибка цели обучения»); проверка: исправленная цель ниже `ln(2048)` и ниже исходной по talker и sub-talker, исходная по talker — не ниже `ln(2048)`; иначе остановиться и разобрать с владельцем до задач группы 3

## 3. Обучение с исправлением

- [x] 3.1 Реализовать `scripts/voice_train/check_stop.py <прогон>` и `config/eval/stop_phrases_ru.txt` (5 коротких фраз): синтез на каждом чекпоинте (`generate_custom_voice`) и на Base (`generate_voice_clone` с референсом прогона), `max_new_tokens=400`, сид 0, таблица «длительность / до лимита», WAV в `data/voice/<прогон>/stop_check/`; проверка: запуск на `pilot-2026-10-17` помечает все чекпоинты непригодными, а Base — останавливающейся на всех фразах
- [x] 3.2 Пробный запуск `smoke-fix` (`prepare.py`, `train.sh --limit 10 --epochs 1`) на GPU 1 и `check_stop.py smoke-fix`; проверка: в `run.json` есть `qwen3_tts_patch`, в конфиге чекпоинта `custom_voice` и `prototype`, чекпоинт останавливается на всех фразах с длительностью 0,5–2× от Base; если нет — остановиться и передать вывод владельцу
- [x] 3.3 Пробный прогон `pilot-2026-10-18`: тот же манифест, 10 эпох, затем `check_stop.py`; rsync `run.json`, `train.log`, `stop_check/` на Mac; проверка: 10 чекпоинтов, таблица `check_stop` с результатом по каждой эпохе записана в `docs/benchmarks/voice-finetune.md` вместе с loss по эпохам
- [x] 3.4 [владелец] Послушать `stop_check/` Base и эпох 0, 4, 9 нового пилота; проверка: в отчёте отмечено, что речь не ускоряется от эпохи к эпохе (issue #179) и голос остаётся речью, а не шумом

## 4. Документы и уборка

- [x] 4.1 Обновить `docs/REFERENCES.md` (патч, его sha256, источники #278/#337/#371/#384, условие удаления), design.md `finetune-voice` (решение «Код дообучения QwenLM» — со ссылкой на патч), её задачу 4.2 (чекпоинт `smoke-fix` вместо 3.2) и формулировку этапа 5 в `docs/PLAN.md` («официальным скриптом Qwen3-TTS с патчем выравнивания»); проверка: `openspec validate finetune-voice` и `openspec validate fix-voice-sft-alignment` проходят, ссылки в документах ведут на существующие файлы
- [ ] 4.2 [владелец] Подтвердить удаление на кластере `data/voice/pilot-2026-10-17/checkpoint-epoch-*` и `data/voice/smoke/checkpoint-epoch-0` (~47 ГБ), после чего агент удаляет их; проверка: `du -sh data/voice/*` на кластере не показывает этих каталогов, а `run.json`, `train.log` и `listen/` старого пилота есть на Mac

## 5. Готовность к продолжению этапа 5

- [x] 5.1 Проверить, что этап 5 (`docs/PLAN.md`, «Дообучение Голоса») можно вести дальше по `finetune-voice`: чекпоинты `pilot-2026-10-18` проходят требование «Дообученный Голос заканчивает фразу», прогон воспроизводим по `run.json` (коммит QwenLM, отпечаток патча, коммит репозитория), а задачи 4.x `finetune-voice` ссылаются на новые чекпоинты; проверка: в `docs/benchmarks/voice-finetune.md` есть разделы «Ошибка цели обучения» и «Пробный прогон с исправлением» с таблицами из 2.2 и 3.3, и в нём сказано, что задача 7.1 `finetune-voice` выполняется на `pilot-2026-10-18`
