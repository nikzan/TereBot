# TereBot

Локальный голосовой Клон конкретного человека для приватного Discord-сервера на русском языке. Термины — [CONTEXT.md](CONTEXT.md), план — [docs/PLAN.md](docs/PLAN.md).

## Быстрый старт

Нужны Node ≥ 22.18 и [uv](https://docs.astral.sh/uv/). Discord-приложение, токен и тестовый сервер настраиваются по [docs/discord-setup.md](docs/discord-setup.md).

```bash
cp .env.example .env          # впишите DISCORD_TOKEN
```

Реальные ID пишите в локальную копию настроек (она в `.gitignore` и заменяет `config/settings.toml`):

```bash
cp config/settings.toml config/settings.local.toml   # впишите guild_id и whitelist
```

Шлюз:

```bash
cd gateway
npm install
npm start
```

## Запуск Клона

Клон — это три процесса на одном Mac: шлюз (Discord), ядро (речь, Политика Диалога, Голос) и `llama-server`, который ядро запускает само. Шлюз и ядро можно запускать в любом порядке: шлюз сам переподключается к ядру каждые 3 с.

Один раз:

```bash
brew install llama.cpp
uv run --with huggingface_hub hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf
```

В `config/settings.local.toml` выключите эхо — с ним шлюз к ядру не подключается:

```toml
[gateway]
echo = false
```

Остальные секции ядра (`[core]`, `[speech]`, `[llm]`, `[voice]`, `[journal]`) необязательны, значения по умолчанию и их смысл — в `config/settings.toml`. Голос по умолчанию — zero-shot по референсу `data/bench/reference/ref.wav` с текстом в `ref.txt` рядом (этап 3). Дообученный Голос подключается в `[voice]` полями `model` и `speaker`, без правок кода.

Паспорт, few-shot примеры и лексикон произношения — `config/passport.md`, `config/fewshot.jsonl`, `config/lexicon.tsv`. Это шаблоны; личные версии кладите рядом как `passport.local.md`, `fewshot.local.jsonl`, `lexicon.local.tsv` (они в `.gitignore`), и ядро прочитает их вместо шаблонов.

Ядро (из корня репозитория, в отдельном терминале):

```bash
uv run --group core python -m terebot.core
```

При первом запуске оно скачивает модели (Smart Turn, Silero VAD, GigaAM, Qwen3-TTS). Клон готов, когда в логе появится «Ядро готово». Если на порту `[llm].port` никто не отвечает, ядро само запускает `llama-server` (лог — `data/llama-server.log`) и останавливает его при выходе. Остановка ядра — Ctrl+C.

Шлюз (в другом терминале):

```bash
cd gateway
npm start
```

В логе шлюза должно появиться `[core] подключено к ядру`. Затем в Discord — `/join` из голосового канала: Клон здоровается, а отвечает тому, кто скажет «Теребот, …».

Журнал Сессии — `data/sessions/<Сессия>/journal.jsonl`, по строке JSON на событие: Реплики, решения Политики с причинами, ответы Клона, задержка по этапам и память раз в секунду, в конце — сводка. Имя каталога то же, что у записей шлюза в `data/recordings/`. Тексты и метрики включаются отдельно в `[journal]`.

Проверить ядро без Discord — тестовым шлюзом: он ведёт Сессию и подаёт WAV (моно 48 кГц) как речь участника из whitelist.

```bash
cd gateway
node scripts/fake-gateway.ts --wav ../data/probe/question.wav
```

## Тесты

Тесты и линтер Python-ядра (из корня репозитория):

```bash
uv sync
uv run ruff check
uv run pytest
```

Тестам конвейера датасета Голоса нужна группа `data`, тестам ядра — `core`: `uv run --group core --group data pytest`. Сам конвейер — [scripts/dataset/README.md](scripts/dataset/README.md).

Тесты шлюза (из `gateway/`; сверяют настройки с Python-ядром, поэтому нужен uv):

```bash
npm test
```
