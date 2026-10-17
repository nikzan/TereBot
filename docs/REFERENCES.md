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
| `onnx-asr` | https://github.com/istupakov/onnx-asr · https://istupakov.github.io/onnx-asr/usage/ | `0.12.0`; имена моделей: `gigaam-v3-ctc`, `gigaam-v3-rnnt`, `gigaam-v3-e2e-ctc`, `gigaam-v3-e2e-rnnt` (e2e — с пунктуацией). Бенчмарков на Apple Silicon нет. 2026-10-15: по умолчанию выбирается CoreML, и `gigaam-v3-e2e-ctc` на нём падает при распознавании (`CoreMLExecutionProvider … error code: -1`); с `providers=["CPUExecutionProvider"]` работает. Для этапа 6 провайдер выбирать явно |
| Smart Turn v3 | https://huggingface.co/pipecat-ai/smart-turn-v3 | BSD-2 |
| Smart Turn v3.2 (ядро) | https://huggingface.co/pipecat-ai/smart-turn-v3/blob/f766f81d3cfdf7737ac64aad813d91bbfd56bf93/smart-turn-v3.2-cpu.onnx · https://github.com/pipecat-ai/smart-turn (`inference.py`) | Проверено 2026-10-17 по HF API: ревизия `f766f81d3cfdf7737ac64aad813d91bbfd56bf93`, файл `smart-turn-v3.2-cpu.onnx`, лицензия BSD-2. Вход — последние ≤ 8 с моно 16 кГц через `WhisperFeatureExtractor(chunk_length=8)` (`transformers`), тензор `input_features`; выход — вероятность завершённости хода, порог 0,5. Грузится `hf_hub_download` с закреплённой ревизией |
| Silero VAD ONNX (ядро) | https://github.com/snakers4/silero-vad/blob/v6.2.3/src/silero_vad/data/silero_vad.onnx | Проверено 2026-10-17: тег `v6.2.3` (та же версия, что `silero-vad==6.2.3` в группе `data`), файл 2 327 524 байт. В ядре — через `onnxruntime` без torch: окна по 512 сэмплов 16 кГц, контекст 64 сэмпла, состояние `state` (2×1×128) своё у каждого участника |
| Pipecat | https://docs.pipecat.ai · https://github.com/pipecat-ai/pipecat | `pipecat-ai 1.12.0`, Python ≥ 3.11 |

## LLM

| Что | Ссылка | Проверено 2026-10-14 |
|---|---|---|
| Qwen3.5-4B | https://huggingface.co/Qwen/Qwen3.5-4B | Apache 2.0 |
| GGUF-кванты | https://huggingface.co/unsloth/Qwen3.5-4B-GGUF | Есть (официального `Qwen/Qwen3.5-4B-GGUF` нет) |
| llama.cpp | https://github.com/ggml-org/llama.cpp | `llama-cpp-python 0.3.36` на PyPI |
| `llama-server` (ядро) | https://github.com/ggml-org/llama.cpp/tree/master/tools/server | Проверено 2026-10-17 по `llama-server --help` (llama.cpp 0.6.0, Homebrew, build 11429): `--reasoning [on\|off\|auto]` выключает рассуждения в чат-шаблоне; `-c` — окно контекста; эндпоинты `/health`, `/v1/chat/completions` (`stream: true`, SSE; `chat_template_kwargs` в запросе), `/tokenize`, `/apply-template`. Клиент ядра — `httpx` 0.28.1 |

## Голос (TTS)

| Что | Ссылка | Проверено 2026-10-14 |
|---|---|---|
| Qwen3-TTS (код, fine-tuning) | https://github.com/QwenLM/Qwen3-TTS · https://arxiv.org/abs/2601.15621 | Русский в списке 10 языков; есть стриминг, `create_voice_clone_prompt` / `generate_voice_clone`, каталог `finetuning`. **MPS/Apple Silicon в README не упоминается** — примеры на CUDA |
| Qwen3-TTS веса | https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base · https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base | Apache 2.0; `qwen-tts 0.1.1` на PyPI |
| Qwen3-TTS на MLX | https://github.com/Blaizzy/mlx-audio · https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 | `mlx-audio 0.5.9`, Qwen3-TTS со стримингом; есть конвертации `mlx-community/Qwen3-TTS-12Hz-{1.7B,0.6B}-Base-{bf16,8bit,4bit}` |
| Сторонние замеры скорости на Mac | https://tinycomputers.io/posts/the-real-cost-of-running-qwen-tts-locally-three-machines-compared.html · https://soniqo.audio/benchmarks | PyTorch MPS на M3 Max: RTF ≈ 2,5–3,5 (медленнее реального времени); MLX на M5 Pro: RTF ≈ 0,8. Вторичные источники, только как ориентир |
| Пин transformers | https://github.com/QwenLM/Qwen3-TTS/issues/237 · https://github.com/QwenLM/Qwen3-TTS/pull/360 | Проверено 2026-10-15 по метаданным PyPI: `qwen-tts 0.1.1` требует `transformers==4.57.3`, `accelerate==1.12.0`. Issue #237 (импорт `check_model_inputs` ломается на transformers 5.x) закрыт ботом за неактивностью без фикса; в комментариях работает только 4.57.3. PR #360 (переход на `transformers>=5.15.1,<6`) открыт, не влит. Держим 4.57.3 в изолированной группе uv |
| Стриминг в `qwen-tts` | https://pypi.org/project/qwen-tts/0.1.1/ (`qwen_tts/inference/qwen3_tts_model.py`) | Проверено 2026-10-15 по исходникам 0.1.1: `generate_voice_clone` возвращает `(list[np.ndarray], sr)` целиком; `non_streaming_mode=False` в докстринге описан как имитация потокового ввода текста, «не настоящая потоковая генерация». В Python API стриминга вывода нет → на PyTorch время до первого звука = полный синтез. Промпт голоса — `create_voice_clone_prompt(ref_audio, ref_text)` → `voice_clone_prompt=` |
| Qwen3-TTS Base в `mlx-audio` | https://github.com/Blaizzy/mlx-audio (`mlx_audio/tts/models/qwen3_tts`) · https://huggingface.co/mlx-community | Проверено 2026-10-15 по исходникам `mlx-audio 0.5.9`: Base клонирует голос через `generate(text, ref_audio=, ref_text=, lang_code=, stream=True, streaming_interval=)` (режим ICL, `repetition_penalty` принудительно ≥ 1,5); кодирование референса кэшируется по `(ref_text, ref_audio)`. Стриминг для ICL есть. Требует `mlx>=0.31.1`, `transformers>=5.14.0`. Конвертер — `mlx_audio.convert` (`--quantize`, `--q-bits`, `--dtype`). На HF есть `mlx-community/Qwen3-TTS-12Hz-{1.7B,0.6B}-Base-{bf16,8bit,6bit,5bit,4bit}` |
| Chatterbox Multilingual | https://github.com/resemble-ai/chatterbox · https://huggingface.co/ResembleAI/chatterbox | MIT; `chatterbox-tts 0.1.7`. Проверено 2026-10-15 по исходникам: `ChatterboxMultilingualTTS.from_pretrained(device="mps")` поддержан явно (веса грузятся на CPU и переносятся), русский (`ru`) в `SUPPORTED_LANGUAGES`, стриминга нет. Закрепляет `torch==2.6.0`, `transformers==5.2.0` — несовместим с `speechbrain>=1.1` (`torch>=2.7.1`) и с остальными группами TTS |
| Chatterbox на MLX | https://huggingface.co/mlx-community/chatterbox-multilingual-v3 | Проверено 2026-10-15: `mlx-audio 0.5.9` поддерживает мультиязычные чекпоинты Chatterbox (`lang_code=`), стриминг игнорируется; готовая конвертация — `mlx-community/chatterbox-multilingual-v3` |
| VoxCPM2 | https://github.com/OpenBMB/VoxCPM · https://huggingface.co/openbmb/VoxCPM2 | Apache 2.0 |

## Русская нормализация

| Что | Ссылка | Проверено 2026-10-14 |
|---|---|---|
| RUAccent | https://github.com/Den4ikAI/ruaccent | `1.5.8.3` |
| `num2words` | https://pypi.org/project/num2words/ · https://github.com/savoirfairelinux/num2words | Проверено 2026-10-17: `0.5.14`, `num2words(n, lang="ru")`. Числа не склоняет: только именительный падеж («в 2026 году» → «в две тысячи двадцать шесть году»), поэтому в ядре — страховка после правила в промпте |

## Политика Диалога

| Что | Ссылка | Проверено 2026-10-15 |
|---|---|---|
| `rapidfuzz` | https://pypi.org/project/RapidFuzz/ · https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html | `3.14.6`; `fuzz.ratio` — нормированное расстояние Индела, 0–100 |
| `pyyaml` (dev, сценарии) | https://pypi.org/project/PyYAML/ | `6.0.3` |
