# Голосовой клон друга в Discord на русском, полностью локально: состояние на октябрь 2026 и рекомендуемый пайплайн

Лучший путь на октябрь 2026 — каскад «GigaAM-v3 (STT) → Qwen3.5/3.6 или T-pro 2.0 (LLM) → Qwen3-TTS-1.7B-Base с дообучением на голосе друга (TTS)». Оркестрация — Pipecat с Silero VAD и Smart Turn v3. Discord-аудио лучше вынести в маленький Node.js-сайдкар на @discordjs/voice ≥ 0.19.2: это единственная библиотека, где приём голоса под обязательным шифрованием DAVE работает в официальном релизе.\[1\]\[2\] End-to-end речевые модели пока не умеют одновременно хорошо говорить по-русски и клонировать конкретный голос локально, поэтому заменять ими каскад рано.

## TL;DR

- **Стек:** GigaAM-v3 e2e_rnnt распознаёт русский. На FLEURS-ru у него WER 5,36% в Q8; по карточке ai-sage/GigaAM-v3 на Hugging Face средний WER v3_RNNT — 8,4% против 25,1% у Whisper (колл-центр — 9,5% против 23,9%). За «мозг» отвечает Qwen3.5-9B (8–12 ГБ VRAM) или Qwen3.5/3.6-27B / T-pro 2.0 (24–32 ГБ). Голос синтезирует Qwen3-TTS-12Hz-1.7B-Base: лицензия Apache 2.0, русский в числе 10 языков, первый пакет через 97 мс,\[3\] есть официальный скрипт fine-tuning. Альтернативы — VoxCPM2 (Apache 2.0, 30 языков, LoRA)\[4\] и Chatterbox Multilingual V3 (MIT, встроенный водяной знак). Fish Audio S2 Pro звучит отлично, но коммерческое использование по его лицензии требует отдельного договора. Voxtral TTS русский не поддерживает.\[5\]
- **Discord:** с 2 марта 2026 E2EE (DAVE) обязательно для всех звонков. Отправка звука работает в discord.py 2.7+ (нужен `davey`) и py-cord 2.8. Приём в релизных Python-библиотеках пока не работает: py-cord PR #3159 не влит, discord-ext-voice-recv работает только в форках. Поэтому Discord-I/O надёжнее делать на Node (@discordjs/voice ≥ 0.19.2), а весь ИИ — в Python-проекте под uv.
- **Задержка и железо:** на RTX 3090/4090/5090 реально получить 0,7–1,2 с до первого звука. На 8–12 ГБ — около 1,2–2 с, с компромиссами: 0,6B-TTS, LLM на 4–9B в Q4 и иногда частичная выгрузка на CPU. На Mac (M-серии, от 32 ГБ) — около 1,5–3 с: TTS на MPS — узкое место, проверяйте на практике. Сначала соберите 30–60 минут чистой речи друга. Они нужны и для дообучения TTS, и для оценки похожести.

## Key Findings

1. **TTS для русского с клонированием.** В 2026 появились три по-настоящему сильные открытые модели с разрешающей лицензией: Qwen3-TTS (январь 2026), VoxCPM2 (апрель 2026) и Chatterbox Multilingual V3.\[4\]\[6\] Fish Audio S2 Pro (март 2026) по качеству, вероятно, сильнее, но русский у него во втором эшелоне (Tier 2), а лицензия исследовательская.\[7\]\[8\] Voxtral TTS от Mistral (Voxtral-4B-TTS-2603), по анонсу самой Mistral, поддерживает «9 popular languages», и русского среди них нет; веса выложены «under CC BY NC 4.0 license», поэтому для нашей задачи он не подходит.
2. **Zero-shot против fine-tuning.** Zero-shot по 3–10 секундам даёт узнаваемый тембр, но манеру речи — темп, «словечки», характерные паузы — передаёт плохо. У Qwen3-TTS и VoxCPM2 есть официальные пути дообучения (SFT/LoRA). Дообучение на 30–60 минутах речи друга — главный рычаг похожести, он важнее выбора модели.
3. **Voice conversion вторым этапом в 2026 уже не обязателен.** По README Plachtaa/seed-vc, у Seed-VC «algorithm delay of ~300ms and device side delay of ~100ms» (для tiny-конфигурации при block 0,18 с — 430 мс, замер на RTX 3060 Laptop), и в собственной оценке авторов он обходит RVCv2 по похожести. Но upstream-репозиторий архивирован 21.11.2025.\[9\] RVC/Applio полезны как «полировка тембра», если после дообучения TTS похожести всё ещё не хватает. Задержки они добавляют.
4. **STT.** GigaAM-v3 (Сбер, MIT, 220–240M параметров, 700 тыс. часов русской речи) — фактический SOTA для русского. В сравнении с Gemini 2.5 Pro в роли LLM-судьи (карточка ai-sage/GigaAM-v3) он выигрывает у Whisper-large-v3 со счётом 70:30. T-one (T-Банк, 71M, потоковый, чанки по 300 мс) — лучший вариант для настоящего стриминга, но он заточен под телефонию.\[10\]
5. **Turn detection.** Smart Turn v3 (по репозиторию pipecat-ai/smart-turn — Whisper Tiny + линейный классификатор, ~8M параметров, int8 на 8 МБ) поддерживает 23 языка; по блогу Daily.co инференс занимает 12,6 мс на AWS c7a.2xlarge, а на русском тест-сете из 1 470 сэмплов точность 93,67%. В Pipecat он включён по умолчанию вместе с Silero VAD (`stop_secs=0.2`).\[11\]
6. **LLM.** В линейке Qwen3.5 (февраль–март 2026, Apache 2.0) есть 9B, 27B и 35B-A3B.\[12\] Qwen3.6 (апрель 2026) добавил 27B и 35B-A3B.\[13\] T-pro 2.0 построен на Qwen3-32B: у него плотный кириллический токенизатор и EAGLE-спекулятивное декодирование под русский.\[14\] Личность друга лучше всего передаёт связка «system prompt + few-shot + RAG по фактам + (опционально) LoRA на переписках».
7. **Speech-to-speech.** Qwen3-Omni-30B-A3B понимает и произносит русский,\[15\] но локально не умеет клонировать произвольный голос. Клонирование в Qwen3.5-Omni документировано только через облачный DashScope.\[16\] Moshi/Unmute от Kyutai ориентированы на английский и французский. Вывод: каскад остаётся единственным путём к «голосу друга».
8. **Discord DAVE — главный инженерный риск 2026 года.** Python-приём под DAVE работает только на неслитых ветках, и у них есть известные баги. Например, в py-cord звук пропадает через 5–9 минут из-за UDPKeepAlive.\[17\]\[18\]

## Details

### 1. TTS и клонирование голоса

| Модель | Релиз | Размер | Русский | Клонирование | Стриминг / скорость | VRAM (оценка) | Лицензия | Вердикт |
|---|---|---|---|---|---|---|---|---|
| **Qwen3-TTS-12Hz-1.7B-Base / 0.6B-Base** (Alibaba) | 22.01.2026 | 0,6B / 1,7B | Да, один из 10 языков\[19\] | Zero-shot от 3 с, есть fine-tuning\[20\] | Первый пакет 97 мс (12Hz-токенизатор), поддержка vLLM-Omni | ~4–8 ГБ | Apache 2.0 | **Основной выбор** |
| **VoxCPM2** (OpenBMB) | 04.2026 | 2B | Да, один из 30 языков, с автоопределением\[4\]\[21\] | Controllable / «Ultimate» (аудио + транскрипт), LoRA/SFT | RTF ~0,3 на RTX 4090;\[22\] ~0,13 с Nano-vLLM / vLLM-Omni; выход 48 кГц | ~8–12 ГБ | Apache 2.0 | Сильная альтернатива |
| **Chatterbox Multilingual V3** (Resemble AI) | 2026 | 0,5B | Да, один из 23+ языков\[23\] | Zero-shot ~5–10 с\[24\]\[25\] | Заявлено < 200 мс (вендор) | ~4–8 ГБ | MIT | Лёгкий вариант для 8 ГБ; водяной знак Perth\[26\] |
| **Fish Audio S2 Pro** | 03.2026 | ~4–5B (бэкбон Qwen3-4B)\[27\] | Tier 2 | Zero-shot ~10 с, fine-tuning код открыт | RTF 0,195 и TTFA ~100 мс — на H200\[28\] | 8–16 ГБ (вторичные данные) | Fish Audio Research License: коммерция только по договору\[8\] | Лучшая выразительность, лицензия ограничивает |
| **ESpeech-TTS-1_RL-V2** (F5-TTS, русский форк) | 2025 | ~0,3B | Только русский + RUAccent\[29\] | Zero-shot по референсу | Нативного стриминга нет | ~4–6 ГБ | На HF указан apache-2.0 — проверьте происхождение данных\[30\] | Хорош для ударений, медленнее для диалога |
| F5-TTS-Russian (hotstone228) | 12.2024 | ~0,3B | Да | Zero-shot | Без нативного стриминга | ~4–6 ГБ | CC BY-NC-SA 4.0\[31\] | Только некоммерческие эксперименты |
| XTTS-v2 (Coqui) | 2023 | ~0,5B | Да | Zero-shot 6 с, fine-tuning\[32\] | Стриминг есть | ~4 ГБ | CPML (некоммерческая);\[33\] компания закрыта | Устарел |
| Voxtral TTS (Mistral) | 23–26.03.2026 | 4B | **Нет** (9 языков: en, fr, de, es, nl, pt, it, hi, ar)\[34\] | 3 с | 70–90 мс TTFA | ≥ 16 ГБ | CC BY-NC 4.0 | Не подходит |
| Kokoro-82M, Orpheus, Dia, Sesame CSM, VibeVoice, Higgs Audio | 2025–2026 | — | Русского нет или он слабый | Kokoro не клонирует;\[35\] остальные англо- и китаецентричны | — | — | Разные | Для русского клона не рекомендуются |

CosyVoice3 (Fun-CosyVoice3-0.5B-2512) поддерживается в vLLM-Omni и клонирует по ref_audio + ref_text,\[36\] но качество его русского я не проверял. IndexTTS2 делает упор на китайский и английский. В обзорах с июня по октябрь 2026 упоминаются OmniVoice (600+ языков), Higgs TTS 3 и Breeze TTS 2 (веса некоммерческие).\[6\]\[37\] Первичных данных о русском у них я не нашёл — **считайте это непроверенным**.

**Почему именно Qwen3-TTS — основной кандидат:**
- Apache 2.0 распространяется и на модели, и на оба токенизатора.\[3\]
- Русский заявлен официально.\[19\]\[38\]
- У 12Hz-токенизатора задержка первого пакета — 97 мс.\[3\]
- В репозитории есть каталог `finetuning`.\[19\]
- Модель поддерживается в vLLM-Omni через OpenAI-совместимый `/v1/audio/speech`, в том числе Base-модель для клонирования.\[36\]
- Есть режим `create_voice_clone_prompt`: промпт голоса считается один раз и переиспользуется.\[19\]

Практическая деталь: по issue #237 в QwenLM/Qwen3-TTS, «qwen-tts 0.1.1 requires transformers==4.57.3 (hardcoded in setup.py)», так что закрепите эту версию.

**Почему VoxCPM2 — второй кандидат.** Режим «Ultimate cloning» продолжает референсное аудио вместе с его транскриптом и так переносит ритм и эмоцию.\[39\] Плюс официальная LoRA-WebUI, Apache 2.0 и выход 48 кГц без ресемплинга под Discord.\[4\]\[40\] Минус — 2B параметров и RTF ~0,3 на 4090 без ускорителя;\[22\] с Nano-vLLM RTF ~0,13, что уже подходит для реального времени.\[4\]

**Русская специфика (ударения, ё, числа, англицизмы)** решается нормализацией текста перед TTS:
- **Числа, даты, сокращения:** разворачивайте в слова (`num2words`, lang="ru", плюс правила склонения) и просите LLM писать числа словами.
- **Ударения и ё:** RUAccent для омографов; проверьте, понимает ли модель знак ударения `+` (Qwen3-TTS и VoxCPM2 его не документируют). Восстанавливайте «ё» словарём.
- **Англицизмы и ники:** пользовательский словарь транслитерации — так, как друг их произносит.

**Voice conversion.** Seed-VC (zero-shot, 1–30 с референса; форк seed-vc-realtime, tiny-конфигурация ~430 мс) — для быстрой проверки.\[9\] RVC v2 / Applio (10–30+ минут на обучение, низкая задержка на чанк) — для полировки тембра после TTS. Исследовательские X-VC, Zero-VC (алгоритмическая задержка 20 мс)\[41\] и SynthVC стоит отслеживать. Моя позиция: **сначала дообучите TTS, и только если похожесть (SIM) недостаточна — добавляйте RVC**. Каждый этап VC добавляет 50–400 мс и риск артефактов на русских шипящих.

### 2. STT, VAD, turn detection, barge-in

| Модель | Релиз | Размер | Русский WER (источник) | Стриминг | Лицензия | Вердикт |
|---|---|---|---|---|---|---|
| **GigaAM-v3 e2e_rnnt** (Сбер) | конец 2025 | 220–240M | FLEURS-ru 5,36% (Q8, transcribe.cpp); win 70:30 против Whisper-large-v3 (судья — Gemini 2.5 Pro); 11,4% на Golos crowd (gigastt) | Посегментный (после VAD), очень быстрый | MIT | **Основной** |
| GigaAM-v3 e2e_ctc | конец 2025 | ~220M | FLEURS-ru 5,50%\[42\] | То же, декод быстрее | MIT | Для CPU / Mac |
| **T-one** (T-Банк) | 2025 | 71M | OpenSTT calls re-labeled: 7,94 (у Whisper large-v3 — 20,82) | Настоящий стриминг, чанки по 300 мс | Apache 2.0 | Если нужны частичные гипотезы |
| Whisper large-v3 / turbo, faster-whisper, WhisperX | 2023–2024 | 809M–1,55B | В среднем 25,1% против 8,4% у GigaAM-v3 RNNT (карточка ai-sage/GigaAM-v3) | Псевдо-стриминг | MIT | Fallback для смешанной речи |
| Parakeet TDT 0.6B v3 (NVIDIA) | 2025 | 0,6B | 25 европейских языков, включая русский\[43\] | Да | CC-BY-4.0 | Хорош на Mac/CPU в ONNX INT8 |
| Vosk ru 0.54 | — | 20–65M | 13–21 на звонках | Да | Apache 2.0 | Только для очень слабого железа |
| Qwen-Audio-3.0-ASR | 09.2026 (arXiv 2609.07549) | — | Отчёт приводит zh/en; русский не проверен | Первый токен < 200 мс | ? | Следить |

GigaAM-v3 доступен как нативный пакет `gigaam`, ONNX-экспорт `istupakov/gigaam-v3-onnx`, sherpa-onnx (экспорт от 16.12.2025) и готовые серверы с Whisper-совместимым API (`gigaam-asr-russian`, `gigastt`).\[44\]\[45\]\[46\]\[47\] Для FP8 авторы форка сообщают ΔWER практически ноль.\[48\]

**VAD и turn detection.** Silero VAD — стандарт (в Pipecat по умолчанию `stop_secs=0.2`). Smart Turn v3.x (BSD-2, Whisper Tiny + линейный классификатор, 8 МБ ONNX int8) анализирует последние 8 секунд реплики по аудио и видит интонацию; на русском точность 93,67%, ложных срабатываний 3,33%.\[49\]\[50\]\[51\] TEN VAD и LiveKit turn detector — альтернативы.

**Barge-in и эхо.** В Discord бот получает отдельные Opus-потоки по SSRC и не слышит сам себя, поэтому классическое AEC не нужно. Но у собеседников без наушников бот может «услышать себя» через их микрофон: помогают шумодав Krisp в клиенте и правило — во время речи бота повышать порог VAD и требовать ≥ 300–500 мс устойчивой речи для прерывания. При barge-in нужно остановить воспроизведение, отменить генерацию LLM и TTS и обрезать историю до фактически произнесённой части. Pipecat делает это через interruption frames.

### 3. «Мозг»: LLM для русского и личности друга

| Модель | Релиз | Размер | Русский | VRAM Q4 (оценка) | Лицензия | Для чего |
|---|---|---|---|---|---|---|
| **Qwen3.5-9B** | 02–03.2026 | 9B dense | Хороший (201 язык) | ~5,5–6,5 ГБ | Apache 2.0 | Уровень 8–12 ГБ |
| **Qwen3.5-27B / Qwen3.6-27B** | 24.02.2026 / 04.2026 | 27B dense | Хороший | ~17 ГБ | Apache 2.0 | Уровень 24–32 ГБ |
| **Qwen3.5/3.6-35B-A3B** | 02 / 04.2026 | 35B MoE, 3B активных | Хороший | ~22 ГБ | Apache 2.0 | Быстрее dense; хорош на Mac |
| **T-pro-it-2.0** (T-Банк) | 12.2025 | 32B (Qwen3-32B) | Лучший среди открытых по русским бенчмаркам авторов; плотный кириллический токенизатор | ~19–20 ГБ | Apache 2.0 | Максимум «русскости»; EAGLE ускоряет |
| Gemma 4 (Google) | 02.04.2026 | E2B, E4B, 26B MoE, 31B | Хороший | 26B MoE ~16 ГБ | Gemma ToU | Альтернатива |
| T-lite, Vikhr, Saiga, GigaChat-открытые | 2024–2025 | 7–20B | Хорошие | — | Разные | Предыдущее поколение |

**Режим рассуждений.** Для голосового бота обязательно отключите thinking: в Qwen3.5 и T-pro 2.0 есть non-thinking режим. Иначе время до первого токена вырастет на секунды.

**Как передать личность друга** — по убыванию отдачи:
1. **System prompt-«паспорт»:** биография, отношения с участниками, темы, табу, словечки, длина реплик («1–2 коротких предложения, разговорно, без списков и эмодзи»).
2. **Few-shot:** 20–50 реальных пар «реплика → ответ друга» из переписок, с согласия.
3. **RAG по памяти и фактам:** `sqlite-vec` или `chromadb` с фактами и историями друга плюс память о прошлых разговорах и о том, кто есть кто, по Discord user ID.
4. **LoRA/QLoRA** на 2–20 тыс. сообщений (`unsloth`/`trl`). Даёт стиль, но может портить фактическую корректность. Rank 16–64, проверка на отложенных диалогах. Начинать без LoRA, добавлять вторым этапом.

**Инференс-движки.** llama.cpp (`llama-server`) — самый простой путь: GGUF Q4_K_M/Q5_K_M, OpenAI-совместимый API, CUDA и Metal. vLLM / SGLang — AWQ/FP8, лучший TTFT при одновременном TTS и EAGLE-спекуляция (для T-pro 2.0 есть готовые EAGLE-веса).\[52\] ExLlamaV3 (EXL3) — лучшая плотность качества на бит на NVIDIA. MLX-LM — для Mac. Ollama удобна, но даёт меньше контроля.

### 4. End-to-end speech-to-speech: стоит ли заменять каскад?

| Модель | Русский | Клонирование заданного голоса | Вывод |
|---|---|---|---|
| Qwen3-Omni-30B-A3B | Вход: 19 языков, выход: 10, включая ru; похожесть голоса на русском 0,759 в их тесте | Нет (фиксированные голоса); talker ~+10 ГБ | Отличный «слух+мозг», но голос не друга |
| Qwen3.5-Omni | Да | Через облачный DashScope; локальный паритет не гарантирован | Не для оффлайн-клона |
| Moshi / Kyutai Unmute | Ориентированы на en/fr | Нет / ограниченно | Нет |
| Ultravox, GLM-4-Voice, Step-Audio, Llama-Omni, Sesame CSM | Слабый или неизвестный русский | В основном нет | Нет |

**Вывод.** Каскад даёт контроль над каждым звеном и остаётся единственной локальной схемой, где голос похож на друга. Гибрид «Qwen3-Omni как уши+мозг с отключённым talker → Qwen3-TTS» возможен, но это 30B MoE ради сомнительного выигрыша — оставьте для экспериментов.

### 5. Интеграция с Discord (самое важное изменение 2026 года)

**DAVE обязателен.** Discord объявил, что с 1 марта 2026 клиенты и приложения без поддержки DAVE не могут участвовать в звонках.\[53\] С начала марта 2026 все голосовые и видеозвонки (DM, групповые, голосовые каналы, Go Live) шифруются end-to-end.\[54\] Stage-каналы, по данным вторичного источника, вне периметра E2EE.\[55\] Шифрование идёт на уровне кадра (AES128-GCM поверх Opus), транспорт — Opus стерео 48 кГц.\[56\]

| Библиотека | Отправка под DAVE | Приём под DAVE | Статус (октябрь 2026) |
|---|---|---|---|
| **@discordjs/voice ≥ 0.19.2** (Node) | Да (0.19.0, 17.08.2025)\[2\] | **Да**: issue #11419 исправлен PR #11449 в 0.19.2 от 13.03.2026\[1\]\[2\] | Открытые баги: #11653 (не восстанавливается после MLS welcome при transition 0, если пользователь уже в канале) и #11441 (часть пакетов теряется при ротации ключей)\[57\]\[58\] |
| discord.py 2.7.0 / 2.7.1 + `davey` | Да | Upstream discord-ext-voice-recv 0.5.2 не расшифровывает DAVE — получается «шум»\[59\] | Нужны форки: zacker150 (самый проработанный), idoldange или rdphillips7 (PR #54 не влит)\[60\]\[61\]\[62\] |
| py-cord 2.8.0 (18.05.2026) / 2.8.1 (25.07.2026) | Да (#3143)\[63\]\[64\] | **Нет в релизе**: PR #3159 открыт, ветка `fix/voice-rec-2`\[17\]\[18\] | Баг #3388: приём замолкает через ~5–9 мин; обход — `UDPKeepAlive.delay = 5`\[18\] |
| songbird 0.6.0 (Rust, 05.04.2026) | Да | Да, через feature "receive"\[65\] | Issue #310: приём «глохнет» после MLS-перехода или перезахода участника\[66\] |

Противоречие: в issue #11419 один пользователь утверждал, что «py-cord 2.6.1 корректно обрабатывает DAVE».\[57\] Это не согласуется с changelog py-cord, документацией и сторонними отчётами; я считаю это утверждение ошибочным или устаревшим.\[17\]\[67\]\[68\]

**Рекомендованная архитектура:**
- **Discord-шлюз — Node.js-сайдкар** (~150–300 строк на discord.js + @discordjs/voice ≥ 0.19.2). Он подписывается на `receiver.subscribe(userId)` для каждого говорящего, декодирует Opus → PCM 48 кГц, шлёт кадры по 20 мс в Python через локальный WebSocket с метаданными `{user_id, ts}`, а обратно принимает PCM от TTS и проигрывает через `AudioPlayer` с `StreamType.Raw`.
- Обязательно добавьте **watchdog**: если от говорящего пользователя N секунд нет декодируемых пакетов, переподпишитесь, а при необходимости переподключитесь. Это обходит #11653-подобные зависания.
- **Python-ядро (uv)** — Pipecat-пайплайн с собственным транспортом поверх WebSocket. Готового Discord-транспорта в Pipecat нет (есть Daily, LiveKit, SmallWebRTC, FastAPI WebSocket, WebSocket Server, Local).\[69\]
- **Чистый Python** возможен на форке zacker150, но вы зависите от неслитого кода.\[62\]

**Виртуальный аудиокабель** (VB-Cable / BlackHole / PipeWire) + обычный клиент Discord: DAVE делает сам официальный клиент, но вы получаете один смешанный поток без разделения по пользователям, нужен отдельный пользовательский аккаунт (self-bots правила Discord запрещают; ручной клиент с роутингом звука — серая зона), и у аккаунта нет пометки BOT. Только как временный обходной путь для прототипа.

**Готовые проекты.** Готового open-source решения «русский голосовой клон в Discord с DAVE-приёмом» я не нашёл. Pipecat (BSD-2), LiveKit Agents, RealtimeSTT/RealtimeTTS, Open-LLM-VTuber дают части пайплайна.\[70\] Discord-бриджи в агентах openclaw и hermes-agent в 2026 ломались как раз на DAVE-приёме — их issues полезно читать как список граблей.\[71\]\[72\]

### 6. Подготовка данных для клона

**Сколько нужно аудио:**
- **Zero-shot:** 3–15 с чистой речи (Qwen3-TTS — от 3 с, Chatterbox — ~5–10 с, VoxCPM2 — 5–50 с).\[20\]\[25\]\[43\]
- **Дообучение TTS (SFT/LoRA):** минимум ~10–15 мин, комфортно 30–60 мин, идеально 1–3 ч разговорной речи. Это моя оценка по практике сообщества; официальных порогов я не нашёл.
- **RVC:** 10–30+ мин.

**Как собирать:**
1. **Специальная запись:** USB-микрофон, тихая комната, 48 кГц / 24 бит, WAV. Пусть друг **разговаривает** (истории, споры, шутки), а не читает — иначе получится «дикторская» интонация. Добавьте фрагмент с числами и англицизмами.
2. **Записи из Discord** (с согласия всех): раздельные пользовательские потоки уже содержат только его голос, но Opus и Krisp ухудшают тембр.
3. **Чистка:** отделение голоса (`audio-separator`), лёгкий шумодав (DeepFilterNet). Сильный шумодав «съедает» тембр.
4. **Диаризация:** pyannote.audio, если в записи несколько людей; фильтрация ECAPA-эмбеддингом по эталону.
5. **Нарезка:** Silero VAD, сегменты по 3–15 с без обрывов слов.
6. **Транскрипция:** GigaAM-v3 e2e_rnnt с ручной вычиткой. Ошибки в транскриптах — главная причина «каши» после fine-tuning. Сохраняйте ё, ударения в омографах и реальные «эээ», если хотите их в голосе.
7. **Формат:** JSONL `{audio, text, duration}` под скрипт fine-tuning (у Qwen3-TTS — `finetuning/`, у VoxCPM2 — `lora_ft_webui.py`).

**Как оценивать** — на отложенных 30–50 фразах: speaker similarity (косинус ECAPA-TDNN / WavLM-SV между синтезом и реальными записями; ориентир — SIM «реальная ↔ реальная»), WER/CER разборчивости через GigaAM-v3, MOS/UTMOS и главный тест — **слепой ABX среди друзей**.

**Типичные ошибки:** переобучение на 10 минутах (голос «заикается»), музыка или эхо в датасете, несоответствие транскрипта аудио, референс с не той эмоцией, 8 кГц или сильно сжатые исходники.

### 7. Оптимизация задержки

**Целевой бюджет** до первого звука от момента, когда человек замолчал (оценки для RTX 4090, требуют замеров):

| Этап | Бюджет | Как достичь |
|---|---|---|
| Discord → сайдкар → Python | 20–60 мс | Кадры по 20 мс, локальный WebSocket |
| VAD-пауза + Smart Turn | 200–300 мс | `stop_secs=0.2` + Smart Turn ~12–60 мс на CPU |
| ASR финала | 50–150 мс | GigaAM на GPU по сегменту; либо T-one стримингом |
| LLM TTFT + первая фраза (~10–15 токенов) | 150–350 мс | Non-thinking, prefix caching, Q4/AWQ, EAGLE |
| TTS первый пакет | 100–300 мс | Qwen3-TTS 12Hz (97 мс в отчёте) или VoxCPM2 со стримингом |
| Воспроизведение + сеть + jitter-буфер | 100–200 мс | Отправка с первого чанка |
| **Итого** | **~0,7–1,3 с** | |

**Техники:**
1. **Стриминг везде:** LLM → разбиение на фразы по `.?!…` (и по запятой после ~8 слов) → TTS по фразам → Opus по чанкам.
2. **Перекрытие:** пока звучит фраза N, синтезируйте N+1.
3. **Прогрев:** пробный прогон всех моделей при старте и заранее посчитанный `voice_clone_prompt`.
4. **Одна GPU:** LLM через `llama-server` или vLLM с лимитом `--gpu-memory-utilization`, TTS отдельным процессом; ASR и Smart Turn можно держать на CPU.
5. **Квантизация:** LLM — Q4_K_M/AWQ/EXL3, ASR — INT8, TTS — bf16 (TTS чувствителен к квантизации).
6. **Короткие ответы:** «1–2 предложения» и `max_tokens` ~80.
7. **«Филлеры»:** заранее синтезированные голосом друга «ну…», «хм, слушай», если LLM задерживается больше чем на 700 мс.

### 8. Безопасность и защита от злоупотреблений

- **Согласие:** письменное, с описанием, где используется голос, кто имеет доступ и как отозвать. При отзыве удаляйте датасет, чекпойнты и LoRA.
- **Маркировка:** бот-аккаунт Discord (тег APP/BOT), имя вида «Вася [ИИ]», голосовое «это ИИ-копия Васи» при входе в канал. В ЕС с 2 августа 2026 действуют обязательства прозрачности AI Act (ст. 50) для дипфейков. В России правовая охрана голоса обсуждалась в законопроектах; проверьте актуальное состояние, если сервер публичный.
- **Правила Discord:** запрещено выдавать себя за другого человека в целях обмана, запрещены self-bots; записывать голос других участников можно только с их ведома.
- **Водяные знаки:** Chatterbox ставит водяной знак Perth на каждый клип;\[26\] для Qwen3-TTS или VoxCPM2 добавьте AudioSeal (Meta) или WavMark пост-процессом.
- **Ограничение доступа:** whitelist серверов и каналов, команды только для владельца, логирование всех синтезированных фраз, запрет синтеза произвольного текста по команде — это защищает от «скажи голосом Васи X».
- **Хранение:** датасет и чекпойнты — только локально, с шифрованием диска; не публикуйте LoRA голоса.

## Рекомендуемые конфигурации

| | **A. Максимальное качество** (RTX 4090/5090, 24–32 ГБ) | **B. Баланс** (RTX 3090/4090 24 ГБ или 16 ГБ) | **C. Минимальное железо** (8–12 ГБ) | **D. Mac Apple Silicon** (M-серия, 32–64 ГБ) |
|---|---|---|---|---|
| Discord I/O | Node + @discordjs/voice ≥ 0.19.2 | То же | То же | То же |
| VAD/turn | Silero + Smart Turn v3 (CPU) | То же | То же | То же |
| STT | GigaAM-v3 e2e_rnnt (GPU) | GigaAM-v3 e2e_rnnt | GigaAM-v3 e2e_ctc ONNX INT8 на CPU | GigaAM-v3 ONNX (CPU) или Parakeet v3 INT8 |
| LLM | T-pro-it-2.0 AWQ/Q4 (~20 ГБ) или Qwen3.6-27B Q4 через vLLM/EXL3 (на 24 ГБ вместе с TTS тесно) | Qwen3.5-9B Q8 или 14–20B-класс в Q4 (35B-A3B Q4 на 24 ГБ с TTS тесно) | Qwen3.5-9B Q4_K_M (~5,5–6,5 ГБ) или 4B | Qwen3.5/3.6-35B-A3B через MLX (4-bit) |
| TTS | Qwen3-TTS-1.7B-Base fine-tuned (или VoxCPM2 + LoRA через Nano-vLLM) | Qwen3-TTS-1.7B-Base fine-tuned | Qwen3-TTS-0.6B-Base или Chatterbox Multilingual V3 | Qwen3-TTS-0.6B/1.7B на MPS — **проверить скорость** |
| VC-полировка | RVC (опционально) | Нет | Нет | Нет |
| VRAM | ~22–30 ГБ | ~14–20 ГБ | ~8–11 ГБ (LLM частично на CPU при 8 ГБ) | 24–40 ГБ unified |
| Ожидаемая задержка до первого звука | ~0,7–1,1 с | ~0,8–1,3 с | ~1,2–2 с | ~1,5–3 с |

Задержки в таблице — инженерные оценки по опубликованным TTFA/RTF, а не мои замеры. Самое неопределённое место — TTS на Mac: официальных цифр для MPS у Qwen3-TTS и VoxCPM2 мне не встречалось. Альтернатива — держать Mac для LLM и STT, а TTS гонять на отдельной машине с NVIDIA в локальной сети.

## «Идеальный» пайплайн

```
[Discord голосовой канал]  (Opus 48 кГц, DAVE E2EE)
        │  раздельные потоки по user_id (SSRC)
        ▼
[Node-сайдкар: discord.js + @discordjs/voice ≥0.19.2 + @snazzah/davey]
   decode Opus→PCM 48k → mono → (16k) → WS кадры 20 мс {user_id, pcm}
   watchdog переподписки при «глухоте» после MLS-ротации
        │ localhost WebSocket
        ▼
[Python / Pipecat pipeline] (uv-проект)
   1. Per-user Silero VAD (stop_secs=0.2) → Smart Turn v3 (конец реплики)
   2. STT: GigaAM-v3 e2e_rnnt → текст с пунктуацией + speaker = Discord-ник
   3. Контекст: «паспорт друга» + few-shot + RAG (факты/память) + журнал канала
   4. LLM (llama-server / vLLM, non-thinking, stream) → разбиение на фразы
   5. Нормализация RU: числа→слова, ё, ударения (RUAccent), словарь англицизмов
   6. TTS: Qwen3-TTS-1.7B-Base (fine-tuned), кэш voice_clone_prompt,
      стриминг чанков → водяной знак (AudioSeal) → (опц. RVC)
   7. Barge-in: VAD собеседника во время речи бота → cancel LLM/TTS, стоп плеера
        │ PCM 24/48k
        ▼
[Node-сайдкар] → Opus encode → AudioPlayer → Discord
```

## Пошаговый план внедрения (uv)

**Шаг 0. Проверка Discord-шлюза (день 1, до всякого ИИ).** Поднимите Node-сайдкар и сделайте «эхо с задержкой 2 с» для каждого пользователя. Убедитесь, что приём не глохнет в течение часа и при входе/выходе участников.

```bash
mkdir discord-gateway && cd discord-gateway
npm init -y
npm i discord.js @discordjs/voice@^0.19.2 @snazzah/davey prism-media @discordjs/opus ws
node -e "console.log(require('@discordjs/voice').generateDependencyReport())"
```

**Шаг 1. Python-проект.**

```bash
uv init --app friend-clone --python 3.12
cd friend-clone
uv add torch torchaudio --index pytorch=https://download.pytorch.org/whl/cu128
uv add "pipecat-ai[silero,websocket]" numpy soundfile soxr websockets
uv add onnx-asr onnxruntime-gpu          # GigaAM-v3 ONNX
uv add qwen-tts "transformers==4.57.3"   # закрепить <5.0
uv add ruaccent num2words openai sqlite-vec sentence-transformers audioseal
uv add --group data pyannote.audio audio-separator deepfilternet speechbrain
uv add --group train peft trl datasets accelerate unsloth
uv add --dev pytest ruff
```

Названия и совместимость пакетов (`onnx-asr`, `audio-separator`, `unsloth` с вашей версией torch, `audioseal`) проверьте перед фиксацией в `uv.lock`. Для Mac уберите CUDA-индекс и `onnxruntime-gpu`, добавьте `mlx-lm`.

**Структура проекта:** `config/` (persona.md, fewshot.jsonl, lexicon_ru.tsv, settings.toml); `data/` (в .gitignore, на зашифрованном диске: raw, clean, segments, manifests); `scripts/` (01_separate, 02_diarize, 03_segment, 04_transcribe, 05_finetune_tts, 06_eval_voice, 07_lora_llm); `src/friend_clone/` (transport_ws, stt_gigaam, tts_qwen, ru_normalize, memory, safety, bot); `discord-gateway/` (Node-сайдкар).

**Шаг 2. Данные.** Запишите 30–60 минут разговорной речи, прогоните скрипты 01–04 и вычитайте транскрипты — 2–4 часа ручной работы, лучшее вложение времени в проекте.

**Шаг 3. Zero-shot baseline.** Qwen3-TTS-1.7B-Base, VoxCPM2 и Chatterbox Multilingual V3 на одном 10-секундном референсе и 30 тестовых фразах с числами, англицизмами и омографами. SIM/WER плюс слепое прослушивание — выберите модель.

**Шаг 4. Дообучение TTS** официальным скриптом выбранной модели. Отложите 10% данных и остановитесь, когда SIM перестанет расти, а WER начнёт расти.

**Шаг 5. LLM.**

```bash
llama-server -m Qwen3.5-9B-Q4_K_M.gguf -c 8192 -ngl 99 --port 8080 \
  --cache-reuse 256 --jinja
```

Напишите `persona.md` и few-shot, отключите thinking, наполните RAG. LoRA — только после того, как промпт-версия перестанет улучшаться.

**Шаг 6. Сборка пайплайна** в `bot.py`: `WebSocketServerTransport` с собственным сериализатором, `SileroVADAnalyzer` + `LocalSmartTurnAnalyzerV3`, ваш `GigaAMSTTService`, `OpenAILLMService` на `base_url=http://localhost:8080/v1`, ваш `QwenTTSService`. Включите interruptions и метрики TTFB по этапам.

**Шаг 7. Многопользовательский режим.** Поток каждого пользователя — в свой VAD/turn-детектор, в контекст LLM пишется «[Петя]: …». Задайте политику, когда бот отвечает (обращение по имени, вопрос, пауза), иначе он будет перебивать всех.

**Шаг 8. Безопасность и маркировка** — whitelist, водяной знак, приветствие «я ИИ-копия». **Шаг 9. Нагрузочный прогон** — 2 часа разговора с 3–4 людьми: ищем «глухоту» после ротации ключей DAVE, утечки VRAM и рост задержки.

## Recommendations

1. **Делайте каскад, а не S2S.** Это единственный локальный путь к голосу конкретного человека на русском в 2026.
2. **Основная связка:** GigaAM-v3 → Qwen3.5-9B (8–12 ГБ) / Qwen3.6-27B или T-pro 2.0 (24–32 ГБ) → Qwen3-TTS-1.7B-Base fine-tuned. Резервные TTS — VoxCPM2 (если «Ultimate cloning» лучше передаёт манеру) и Chatterbox V3 (мало VRAM или нужен MIT с водяным знаком).
3. **Discord-I/O — на Node.** Пересмотрите это решение, когда py-cord выпустит релиз с PR #3159 или discord.py-экосистема получит официальный DAVE-receive.
4. **Не тратьте время на Voxtral TTS** (нет русского, NC), F5-TTS-Russian от hotstone228 (NC) и XTTS-v2 (CPML, проект мёртв) для финального решения.
5. **Fish Audio S2 Pro** — только для личного некоммерческого использования; для большего нужен договор с Fish Audio.\[8\]
6. **Главный рычаг качества** — 30–60 минут чистой разговорной речи с вычитанными транскриптами, а не выбор модели.

## Caveats

- **Быстро меняющаяся область.** Статусы Discord-библиотек (PR #3159, #11653, songbird #310) могут измениться в любую неделю. Слиты ли py-cord #3159 и discord-ext-voice-recv #54, я подтвердил только косвенно.
- **Цифры задержки и VRAM** в конфигурациях — инженерные оценки. Официальные данные есть только для отдельных этапов: Qwen3-TTS — 97 мс первого пакета;\[3\] VoxCPM2 — RTF ~0,3 / 0,13 на 4090;\[4\]\[22\] Fish S2 — RTF 0,195 на H200;\[28\] Smart Turn — 12,6 мс на CPU (AWS c7a.2xlarge, по блогу Daily.co).
- **Качество русского у TTS-моделей** в основном заявлено вендорами. Независимых русскоязычных сравнений клонирования за 2026 год я не нашёл; рейтинги из обзоров (TTS Arena Elo и т.п.) противоречат друг другу и почти не включают русский.
- **Данные о WER GigaAM** получены на разных наборах (FLEURS 5,36%, Golos 11,4%, колл-центр) и между собой не сравнимы; сравнение с Whisper «~18%» в README gigastt сделано не на тех же данных.\[45\]
- **Лицензия ESpeech** (apache-2.0 на HF) требует проверки: исходный F5-TTS обучен на данных с некоммерческими ограничениями.
- **Новые модели второй половины 2026** (OmniVoice, Higgs TTS 3, Breeze TTS 2, Qwen-Audio-3.0-ASR) известны мне только по вторичным обзорам или абстрактам.

**Что проверить на практике до финального выбора стека:**
1. Часовой тест DAVE-приёма с входами и выходами участников.
2. Слепой ABX трёх TTS после zero-shot и после дообучения.
3. Ударения и омографы: «замок», «мука», «все/всё».
4. Реальный TTFA TTS при одновременно загруженной LLM на одной GPU.
5. Поведение barge-in, когда собеседник говорит без наушников.
6. Задержку на Mac (MPS) или решение вынести TTS на отдельную машину.

## Sources

1. [fix(voice): strip padding from packets and add guards by Snazzah · Pull Request #11449 · discordjs/discord.js](https://github.com/discordjs/discord.js/pull/11449)
2. [build(deps): bump @discordjs/voice from 0.18.0 to 0.19.2 by dependabot\[bot\] · Pull Request #13 · skygenesisenterprise/astron-collection](https://github.com/skygenesisenterprise/astron-collection/pull/13)
3. [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621)
4. [GitHub - OpenBMB/VoxCPM: VoxCPM2: Tokenizer-Free TTS for Multilingual Speech Generation, Creative Voice Design, and True-to-Life Cloning](https://github.com/OpenBMB/VoxCPM)
5. [Mistral Voxtral TTS: Open-Weight Model Beats ElevenLabs at 90ms Latency](https://abhs.in/blog/mistral-voxtral-tts-open-weight-beats-elevenlabs-90ms-latency-2026)
6. [Best Open-Source Text to Speech in 2026: 10 Free Models Ranked](https://texttolab.com/blog/open-source-text-to-speech)
7. [fishaudio/s2-pro · Hugging Face](https://huggingface.co/fishaudio/s2-pro)
8. [s2-pro / README.md](https://huggingface.co/ibrahim019/s2-pro/blob/main/README.md)
9. [GitHub - jiaheguo521/seed-vc-realtime: zero-shot voice conversion & singing voice conversion, with real-time support · GitHub](https://github.com/jiaheguo521/seed-vc-realtime)
10. [GitHub - voicekit-team/T-one: T-one is a high-performance streaming ASR pipeline for Russian, specialized for the telephony domain. · GitHub](https://github.com/voicekit-team/T-one)
11. [Speech Input & Turn Detection - Pipecat](https://docs.pipecat.ai/pipecat/learn/speech-input)
12. [Qwen 3.5 Review 2026: The Full Model Family — From 0.8B Edge to 122B Enterprise - ComputerTech](https://computertech.co/qwen-3-5-review/)
13. [Best Qwen Models Ranked: Which to Run Locally (Mid-2026)](https://insiderllm.com/guides/qwen-models-guide/)
14. [T-pro 2.0: An Efficient Russian Hybrid-Reasoning Model and Playground](https://arxiv.org/html/2512.10430)
15. [Qwen/Qwen3-Omni-30B-A3B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
16. [Qwen3.5-Omni Can Clone Your Voice, Whisper, and Shout](https://www.itechguides.com/qwen3-5-omni-can-clone-your-voice-whisper-and-shout-but-theres-a-catch/)
17. [fix: voice recording with DAVE E2EE, CPU transcription, token logging by RandomUsernameForMe · Pull Request #2 · RandomUsernameForMe/discord-noter](https://github.com/RandomUsernameForMe/discord-noter/pull/2)
18. [Voice receive stops after a few minutes: UDPKeepAlive sleeps 5000 seconds](https://github.com/Pycord-Development/pycord/issues/3388)
19. [GitHub - QwenLM/Qwen3-TTS: Qwen3-TTS is an open-source series of TTS models developed by the Qwen team at Alibaba Cloud, supporting stable, expressive, and streaming speech generation, free-form voice design, and vivid voice cloning. · GitHub](https://github.com/QwenLM/Qwen3-TTS)
20. [Qwen3-TTS: The Complete 2026 Guide to Open-Source Voice Cloning and AI Speech Generation](https://medium.com/@zh.milo/qwen3-tts-the-complete-2026-guide-to-open-source-voice-cloning-and-ai-speech-generation-1a2efca05cd6)
21. [Voxcpm](https://voxcpm.app/blog/how-to-use-voxcpm2)
22. [VoxCPM2 TTS & AI Voice Cloning](https://voxcpm.net/)
23. [GitHub - resemble-ai/chatterbox: SoTA open-source TTS · GitHub](https://github.com/resemble-ai/chatterbox)
24. [Chatterbox Voice Cloning](https://docs.clore.ai/guides/audio-and-voice/chatterbox-tts)
25. [Chatterbox TTS Setup: Free ElevenLabs Killer (MIT, 2026)](https://localaimaster.com/blog/chatterbox-tts-setup-guide)
26. [Evermx](https://evermx.com/open-source/chatterbox-open-source-tts-zero-shot-voice-cloning-multilingual)
27. [Changelog - Fish Audio - Overview](https://docs.fish.audio/developer-guide/getting-started/changelog)
28. [Fish Audio Open-Sources S2: Fine-Grained Control Meets Production Streaming - Fish Audio Blog](https://fish.audio/blog/fish-audio-open-sources-s2/)
29. [ESpeech/ESpeech-TTS-1\_RL-V2 · Hugging Face](https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2)
30. [ESpeech/ESpeech-TTS-1\_RL-V2 · Discussions](https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2/discussions)
31. [README.md · hotstone228/F5-TTS-Russian at 079e90f18f74fcfcefa677248e4f253f59b10b14](https://huggingface.co/hotstone228/F5-TTS-Russian/blob/079e90f18f74fcfcefa677248e4f253f59b10b14/README.md)
32. [Best Open-Source TTS 2026: Chatterbox 65.3% Beats ElevenLabs](https://findskill.ai/blog/best-open-source-tts-2026/)
33. [Best TTS for Voice Cloning (2026): Quality, Data, Ethics](https://www.codesota.com/speech/best-for-voice-cloning)
34. [Speaking of Voxtral](https://mistral.ai/news/voxtral-tts/)
35. [Qwen3 TTS — Free AI Voice Generator & Voice Cloning (Colab)](https://vynex.ai/tools/qwen3-tts)
36. [Text to Speech - vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/)
37. [Best Open Source Self-Hosted Text-to-Speech Models in 2026](https://pinggy.io/blog/best_open_source_self_hosted_text_to_speech_models/)
38. [Qwen3-TTS Local Setup: Voice Cloning and Voice Design on Your Own GPU](https://localaimaster.com/blog/qwen3-tts-local-setup)
39. [VoxCPM2 - AI Model Library](https://vast.ai/model/voxcpm2)
40. [VoxCPM: An Open-Source Model for Multilingual Speech Synthesis and Voice Cloning](https://silenceper.com/en/article/2026-06-19-voxcpm-open-source-tts-voice-cloning/)
41. [Zero-VC: Zero-Lookahead Streaming Voice Conversion via Speaker Anonymization](https://arxiv.org/pdf/2606.20218)
42. [transcribe.cpp/docs/models/gigaam.md at main · handy-computer/transcribe.cpp](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/gigaam.md)
43. [GitHub - timoncool/VoxCPM2\_portable: Portable Windows build of VoxCPM2 — multilingual TTS (30 languages incl. Russian), Voice Design, Voice Cloning, and end-to-end LoRA fine-tuning (video/audio → dataset → training). One-click install.](https://github.com/timoncool/VoxCPM2_portable)
44. [GitHub - badigady/gigaam-v3: GigaAM v3 Russian ASR server · GitHub](https://github.com/badigady/gigaam-v3)
45. [GitHub - ekhodzitsky/gigastt: Local STT server powered by GigaAM v3. · GitHub](https://github.com/ekhodzitsky/gigastt)
46. [GitHub - x0152/gigaam-asr-russian: OpenAI Whisper-compatible API server for Russian speech-to-text, powered by GigaAM · GitHub](https://github.com/x0152/gigaam-asr-russian)
47. [FR: Add GigaAM v3 Russian speech recognition model · Issue #127 · BryceWG/BiBi-Keyboard](https://github.com/BryceWG/BiBi-Keyboard/issues/127)
48. [vpermilp/GigaAM-v3 · Hugging Face](https://huggingface.co/vpermilp/GigaAM-v3)
49. [pipecat-ai/smart-turn-v3 · Hugging Face](https://huggingface.co/pipecat-ai/smart-turn-v3)
50. [Announcing Smart Turn v3, with CPU inference in just 12ms](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/)
51. [Smart Turn Overview - Pipecat](https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview)
52. [\[2512.10430\] T-pro 2.0: An Efficient Russian Hybrid-Reasoning Model and Playground](https://arxiv.org/abs/2512.10430)
53. [Bringing DAVE to All Discord Platforms](https://discord.com/blog/bringing-dave-to-all-discord-platforms)
54. [Every Voice and Video Call on Discord Is Now End-to-End Encrypted](https://discord.com/blog/every-voice-and-video-call-on-discord-is-now-end-to-end-encrypted)
55. [Discord Finalizes E2EE for All Voice and Video Calls; Legacy Clients Without DAVE Cut Off](https://claypier.com/en/discord-dave-e2ee-enforcement-march-2026/)
56. [Voice - Documentation - Discord](https://docs.discord.com/developers/topics/voice-connections)
57. [@discordjs/voice 0.19.x DAVE encryption causes reconnect loops and zero audio capture](https://github.com/discordjs/discord.js/issues/11419)
58. [voice: DAVE decrypt failures never recover after an MLS welcome at transition 0 (lastTransitionId 0 is falsy) · Issue #11653 · discordjs/discord.js](https://github.com/discordjs/discord.js/issues/11653)
59. [fix(voice\_recorder): DAVE-decrypt received voice frames (root cause of scratching) by Pu11en · Pull Request #14 · Pu11en/ebi-agent-chat-relay](https://github.com/Pu11en/ebi-agent-chat-relay/pull/14)
60. [GitHub - idoldange/discord-ext-voice-recv: Voice receive extension package for discord.py · GitHub](https://github.com/idoldange/discord-ext-voice-recv)
61. [Unpin \`discord-ext-voice-recv\` fork after upstream PR #54 is merged · Issue #27 · ruxailab/meeting-summarizer](https://github.com/ruxailab/meeting-summarizer/issues/27)
62. [GitHub - zacker150/discord-ext-voice-recv: Voice receive extension package for discord.py · GitHub](https://github.com/zacker150/discord-ext-voice-recv)
63. [pycord latest version](https://python.libhunt.com/pycord-latest-version)
64. [py-cord · PyPI](https://pypi.org/project/py-cord/)
65. [Releases · serenity-rs/songbird](https://github.com/serenity-rs/songbird/releases)
66. [Voz: áudio de outros participantes fica indecifrável após re-chave DAVE (songbird #310) · Issue #1 · FelipeMayerDev/Vesktop](https://github.com/FelipeMayerDev/Vesktop/issues/1)
67. [Changelog - Pycord v2.8 Documentation](https://docs.pycord.dev/en/master/changelog.html)
68. [Voice Related - Pycord v2.8 Documentation](https://docs.pycord.dev/en/master/api/voice.html)
69. [GitHub - pipecat-ai/pipecat: Open Source framework for voice agents, multimodal apps, and realtime AI. Maintained by Daily and the community. · GitHub](https://github.com/pipecat-ai/pipecat)
70. [Pipecat Documentation - Pipecat](https://docs.pipecat.ai/overview/introduction)
71. [Discord voice: audio receive broken due to DAVE E2EE decryption failure (@discordjs/voice 0.19.0) · Issue #24825 · openclaw/openclaw](https://github.com/openclaw/openclaw/issues/24825)
72. [Discord voice receiver goes silently deaf after a DAVE re-key or voice reconnect — credentials are snapshotted once at start() · Issue #77968 · NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent/issues/77968)
