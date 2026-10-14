# TereBot

Локальный голосовой Клон конкретного человека для приватного Discord-сервера на русском языке. Термины — [CONTEXT.md](CONTEXT.md), план — [docs/PLAN.md](docs/PLAN.md).

## Быстрый старт

Нужны Node ≥ 22.18 и [uv](https://docs.astral.sh/uv/). Discord-приложение, токен и тестовый сервер настраиваются по [docs/discord-setup.md](docs/discord-setup.md).

```bash
cp .env.example .env          # впишите DISCORD_TOKEN
```

В `config/settings.toml` замените `guild_id` и `whitelist` на свои ID.

Шлюз:

```bash
cd gateway
npm install
npm start
```

Тесты и линтер Python-ядра (из корня репозитория):

```bash
uv sync
uv run ruff check
uv run pytest
```

Тесты шлюза (из `gateway/`; сверяют настройки с Python-ядром, поэтому нужен uv):

```bash
npm test
```
