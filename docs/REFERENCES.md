# Первоисточники

Ссылки, по которым сверяемся перед выбором версий и API. Колонка «Проверено» — что подтверждено напрямую (реестр пакетов, HF API, README/PR) на 2026-10-14. Общий обзор — [research/2026-10-deep-research.md](research/2026-10-deep-research.md); он опирается во многом на вторичные источники, поэтому факты из него сверять здесь.

## Discord-шлюз (Node)

| Что | Ссылка | Проверено 2026-10-14 |
|---|---|---|
| `@discordjs/voice` | https://www.npmjs.com/package/@discordjs/voice · https://github.com/discordjs/discord.js/tree/main/packages/voice | latest `0.19.2` (опубликован 2026-03-17); `1.0.0-dev.*` — только dev-тег |
| Фикс приёма под DAVE | https://github.com/discordjs/discord.js/pull/11449 (закрывает #11419) | Влит 2026-03-13, т.е. до релиза 0.19.2; фактическую работу подтверждает только Эхо-тест |
| Баг #11653 | https://github.com/discordjs/discord.js/issues/11653 | **Открыт, PR нет** (перепроверено 2026-10-15). Срабатывает, когда бот заходит в канал, где уже есть люди (наш основной сценарий). Обход — патч [`gateway/patches/@discordjs+voice+0.19.2.patch`](../gateway/patches/@discordjs+voice+0.19.2.patch): `lastTransitionId !== undefined` в `DAVESession.decrypt`, применяется `patch-package` на `postinstall`. При обновлении `@discordjs/voice` пересмотреть; если upstream починит — удалить |
| Баг #11441 | https://github.com/discordjs/discord.js/issues/11441 | Закрыт как not planned 2026-03-05 (перепроверено 2026-10-15): до 34% потерь пакетов при смене ключей DAVE — патча нет, принимаем и измеряем в Эхо-тесте |
| `@snazzah/davey` (DAVE) | https://www.npmjs.com/package/@snazzah/davey | latest `0.1.12` |
| `discord.js` | https://discord.js.org | latest `14.27.0` |
| `@discordjs/opus`, `prism-media` | npm | `0.10.0`, `1.3.5`; `@discordjs/opus` на arm64 собирается, `generateDependencyReport()` находит Opus и DAVE (2026-10-15) |
| `patch-package` | https://www.npmjs.com/package/patch-package | `8.0.1` |
| Протокол голоса Discord | https://docs.discord.com/developers/topics/voice-connections | — |
| DAVE обязателен | https://discord.com/blog/bringing-dave-to-all-discord-platforms | — |

## Распознавание речи

| Что | Ссылка | Проверено 2026-10-14 |
|---|---|---|
| GigaAM-v3 (веса) | https://huggingface.co/ai-sage/GigaAM-v3 · https://github.com/salute-developers/GigaAM | MIT; на PyPI `gigaam 0.1.0` |
| GigaAM-v3 ONNX | https://huggingface.co/istupakov/gigaam-v3-onnx | MIT |
| `onnx-asr` | https://github.com/istupakov/onnx-asr · https://istupakov.github.io/onnx-asr/usage/ | `0.12.0`; имена моделей: `gigaam-v3-ctc`, `gigaam-v3-rnnt`, `gigaam-v3-e2e-ctc`, `gigaam-v3-e2e-rnnt` (e2e — с пунктуацией). Бенчмарков на Apple Silicon нет |
| Smart Turn v3 | https://huggingface.co/pipecat-ai/smart-turn-v3 | BSD-2 |
| Pipecat | https://docs.pipecat.ai · https://github.com/pipecat-ai/pipecat | `pipecat-ai 1.12.0`, Python ≥ 3.11 |

## LLM

| Что | Ссылка | Проверено 2026-10-14 |
|---|---|---|
| Qwen3.5-4B | https://huggingface.co/Qwen/Qwen3.5-4B | Apache 2.0 |
| GGUF-кванты | https://huggingface.co/unsloth/Qwen3.5-4B-GGUF | Есть (официального `Qwen/Qwen3.5-4B-GGUF` нет) |
| llama.cpp | https://github.com/ggml-org/llama.cpp | `llama-cpp-python 0.3.36` на PyPI |

## Голос (TTS)

| Что | Ссылка | Проверено 2026-10-14 |
|---|---|---|
| Qwen3-TTS (код, fine-tuning) | https://github.com/QwenLM/Qwen3-TTS · https://arxiv.org/abs/2601.15621 | Русский в списке 10 языков; есть стриминг, `create_voice_clone_prompt` / `generate_voice_clone`, каталог `finetuning`. **MPS/Apple Silicon в README не упоминается** — примеры на CUDA |
| Qwen3-TTS веса | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base · https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base | Apache 2.0; `qwen-tts 0.1.1` на PyPI |
| Qwen3-TTS на MLX | https://github.com/Blaizzy/mlx-audio · https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 | `mlx-audio 0.5.9`, Qwen3-TTS со стримингом; есть конвертации `mlx-community/Qwen3-TTS-12Hz-{1.7B,0.6B}-Base-{bf16,8bit,4bit}` |
| Сторонние замеры скорости на Mac | https://tinycomputers.io/posts/the-real-cost-of-running-qwen-tts-locally-three-machines-compared.html · https://soniqo.audio/benchmarks | PyTorch MPS на M3 Max: RTF ≈ 2,5–3,5 (медленнее реального времени); MLX на M5 Pro: RTF ≈ 0,8. Вторичные источники, только как ориентир |
| Пин transformers | https://github.com/QwenLM/Qwen3-TTS/issues/237 | Не проверялось (README версию не указывает) |
| Chatterbox Multilingual | https://github.com/resemble-ai/chatterbox · https://huggingface.co/ResembleAI/chatterbox | MIT; `chatterbox-tts 0.1.7` |
| VoxCPM2 | https://github.com/OpenBMB/VoxCPM · https://huggingface.co/openbmb/VoxCPM2 | Apache 2.0 |

## Русская нормализация

| Что | Ссылка | Проверено 2026-10-14 |
|---|---|---|
| RUAccent | https://github.com/Den4ikAI/ruaccent | `1.5.8.3` |
