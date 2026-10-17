"""Загрузка и проверка несекретных настроек.

Читается config/settings.local.toml, если он есть (в .gitignore, там реальные ID),
иначе config/settings.toml из git. Файлы не сливаются: локальный заменяет общий целиком.
Тот же выбор файла и ту же схему проверяет шлюз (gateway/src/config.ts).
"""

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

# Discord snowflake: целое число до 2^64, в настройках хранится строкой.
_SNOWFLAKE = re.compile(r"^\d{17,20}$")


class SettingsError(Exception):
    """Настройки отсутствуют или не соответствуют схеме."""


@dataclass(frozen=True)
class DiscordSettings:
    guild_id: str
    whitelist: tuple[str, ...]


# Варианты имени Прототипа, которые не должны будить Клона, вместе с падежными формами.
DEFAULT_WAKE_BLOCKLIST = (
    "терем", "терема", "терему", "теремом", "тереме",
    "терешкин", "терешкина", "терешкину", "терешкиным", "терешкине",
)  # fmt: skip


@dataclass(frozen=True)
class DialogueSettings:
    """Пороги Политики Диалога и варианты Слова-вызова."""

    wake_words: tuple[str, ...] = ("теребот",)
    # rapidfuzz.fuzz.ratio, 0–100.
    wake_threshold: float = 80
    wake_blocklist: tuple[str, ...] = DEFAULT_WAKE_BLOCKLIST
    window_s: float = 20
    barge_in_ms: int = 1000


@dataclass(frozen=True)
class CoreSettings:
    """Связь шлюза и ядра: ядро — WebSocket-сервер на 127.0.0.1:port."""

    port: int = 8765


@dataclass(frozen=True)
class SpeechSettings:
    """Восприятие речи: VAD, конец Реплики, распознавание."""

    # Вероятность Silero VAD, выше которой окно считается речью.
    vad_threshold: float = 0.5
    # Вероятность Smart Turn, выше которой участник договорил.
    turn_threshold: float = 0.5
    # Столько тишины после речи — и Smart Turn решает, закончена ли Реплика.
    silence_ms: int = 200
    # Паузы короче этого не обнуляют длительность непрерывной речи.
    continuity_ms: int = 300
    # Дольше этого после конца речи Реплика не ждёт продолжения.
    max_pause_ms: int = 1500
    # Реплика длиннее распознаётся частями, поделёнными по паузам.
    asr_max_s: float = 20


DEFAULT_GGUF = (
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/*/Qwen3.5-4B-Q4_K_M.gguf"
)


@dataclass(frozen=True)
class LlmSettings:
    """LLM через llama-server; ядро запускает его само, если порт не отвечает."""

    # Путь к GGUF, `~` и `*` раскрываются.
    model: str = DEFAULT_GGUF
    port: int = 8080
    # Окно llama-server (-c).
    ctx_size: int = 4096
    # Бюджет промпта; остаток окна — на ответ.
    context_tokens: int = 3000
    max_tokens: int = 256
    timeout_s: float = 30


@dataclass(frozen=True)
class VoiceSettings:
    """Голос: zero-shot (ref_audio + ref_text) или дообученная модель (speaker)."""

    model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    # Относительно корня репозитория. Если ref_text не задан, он читается из ref_audio с .txt.
    ref_audio: Path | None = REPO_ROOT / "data" / "bench" / "reference" / "ref.wav"
    ref_text: str | None = None
    # Говорящий дообученной модели; если задан, референс не используется.
    speaker: str | None = None
    streaming_interval: float = 0.5
    cache_limit_gb: float = 1.0
    min_phrase_chars: int = 20
    max_phrase_chars: int = 200
    greetings: tuple[str, ...] = ("Всем привет!", "Здарова.", "О, привет всем.")


@dataclass(frozen=True)
class JournalSettings:
    # Тексты Реплик и ответов, решения Политики.
    transcripts: bool = True
    # Задержка по этапам, память, сводка.
    metrics: bool = True
    dir: Path = REPO_ROOT / "data" / "sessions"


@dataclass(frozen=True)
class Settings:
    discord: DiscordSettings
    dialogue: DialogueSettings = DialogueSettings()
    core: CoreSettings = CoreSettings()
    speech: SpeechSettings = SpeechSettings()
    llm: LlmSettings = LlmSettings()
    voice: VoiceSettings = VoiceSettings()
    journal: JournalSettings = JournalSettings()


def resolve_settings_path(config_dir: Path = CONFIG_DIR) -> Path:
    local = config_dir / "settings.local.toml"
    return local if local.exists() else config_dir / "settings.toml"


def load_settings(path: Path | None = None) -> Settings:
    if path is None:
        path = resolve_settings_path()
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        raise SettingsError(f"{path}: файл настроек не найден") from None
    except tomllib.TOMLDecodeError as e:
        raise SettingsError(f"{path}: некорректный TOML: {e}") from None

    def fail(field: str, problem: str) -> SettingsError:
        return SettingsError(f"{path}: поле {field} {problem}")

    discord = raw.get("discord")
    if not isinstance(discord, dict):
        raise fail("[discord]", "отсутствует")

    guild_id = discord.get("guild_id")
    if guild_id is None:
        raise fail("discord.guild_id", "отсутствует")
    if not isinstance(guild_id, str) or not _SNOWFLAKE.match(guild_id):
        raise fail("discord.guild_id", "должно быть строкой с Discord ID из 17–20 цифр")

    whitelist = discord.get("whitelist")
    if whitelist is None:
        raise fail("discord.whitelist", "отсутствует")
    if not isinstance(whitelist, list):
        raise fail("discord.whitelist", "должно быть списком строк с Discord user ID")
    if not whitelist:
        raise fail("discord.whitelist", "пусто: whitelist пуст, Клону некого слушать")
    for i, user_id in enumerate(whitelist):
        if not isinstance(user_id, str) or not _SNOWFLAKE.match(user_id):
            raise fail(f"discord.whitelist[{i}]", "должно быть строкой с Discord ID из 17–20 цифр")

    def table(name: str) -> dict:
        value = raw.get(name, {})
        if not isinstance(value, dict):
            raise fail(f"[{name}]", "должно быть таблицей")
        return value

    def number(section: str, field: str, fallback: float, integer: bool = False) -> float:
        value = table(section).get(field)
        if value is None:
            return fallback
        valid_type = int if integer else (int, float)
        if isinstance(value, bool) or not isinstance(value, valid_type) or value <= 0:
            kind = "целым числом" if integer else "числом"
            raise fail(f"{section}.{field}", f"должно быть {kind} больше 0")
        return value

    def integer(section: str, field: str, fallback: int) -> int:
        return int(number(section, field, fallback, integer=True))

    def probability(section: str, field: str, fallback: float) -> float:
        value = number(section, field, fallback)
        if value >= 1:
            raise fail(f"{section}.{field}", "должно быть числом от 0 до 1")
        return value

    def flag(section: str, field: str, fallback: bool) -> bool:
        value = table(section).get(field)
        if value is None:
            return fallback
        if not isinstance(value, bool):
            raise fail(f"{section}.{field}", "должно быть true или false")
        return value

    def text(section: str, field: str, fallback: str | None) -> str | None:
        value = table(section).get(field)
        if value is None:
            return fallback
        if not isinstance(value, str) or not value.strip():
            raise fail(f"{section}.{field}", "должно быть непустой строкой")
        return value

    def path(section: str, field: str, fallback: Path | None) -> Path | None:
        value = text(section, field, None)
        return fallback if value is None else REPO_ROOT / Path(value).expanduser()

    def words(
        section: str, field: str, fallback: tuple[str, ...], allow_empty: bool
    ) -> tuple[str, ...]:
        value = table(section).get(field)
        if value is None:
            return fallback
        if not isinstance(value, list) or not all(isinstance(w, str) and w.strip() for w in value):
            raise fail(f"{section}.{field}", "должно быть списком непустых строк")
        if not value and not allow_empty:
            raise fail(f"{section}.{field}", "пусто")
        return tuple(value)

    dialogue = DialogueSettings()
    wake_threshold = number("dialogue", "wake_threshold", dialogue.wake_threshold)
    if wake_threshold > 100:
        raise fail("dialogue.wake_threshold", "должно быть числом от 0 до 100")
    dialogue = DialogueSettings(
        wake_words=words("dialogue", "wake_words", dialogue.wake_words, allow_empty=False),
        wake_threshold=wake_threshold,
        wake_blocklist=words("dialogue", "wake_blocklist", dialogue.wake_blocklist, True),
        window_s=number("dialogue", "window_s", dialogue.window_s),
        barge_in_ms=integer("dialogue", "barge_in_ms", dialogue.barge_in_ms),
    )

    port = integer("core", "port", CoreSettings.port)
    if port > 65535:
        raise fail("core.port", "должно быть номером порта до 65535")

    speech = SpeechSettings()
    speech = SpeechSettings(
        vad_threshold=probability("speech", "vad_threshold", speech.vad_threshold),
        turn_threshold=probability("speech", "turn_threshold", speech.turn_threshold),
        silence_ms=integer("speech", "silence_ms", speech.silence_ms),
        continuity_ms=integer("speech", "continuity_ms", speech.continuity_ms),
        max_pause_ms=integer("speech", "max_pause_ms", speech.max_pause_ms),
        asr_max_s=number("speech", "asr_max_s", speech.asr_max_s),
    )
    if speech.max_pause_ms < speech.silence_ms:
        raise fail("speech.max_pause_ms", "должно быть не меньше speech.silence_ms")

    llm = LlmSettings()
    llm = LlmSettings(
        model=text("llm", "model", llm.model),
        port=integer("llm", "port", llm.port),
        ctx_size=integer("llm", "ctx_size", llm.ctx_size),
        context_tokens=integer("llm", "context_tokens", llm.context_tokens),
        max_tokens=integer("llm", "max_tokens", llm.max_tokens),
        timeout_s=number("llm", "timeout_s", llm.timeout_s),
    )
    if llm.context_tokens + llm.max_tokens > llm.ctx_size:
        raise fail("llm.context_tokens", "вместе с llm.max_tokens больше llm.ctx_size")

    voice = VoiceSettings()
    speaker = text("voice", "speaker", None)
    voice = VoiceSettings(
        model=text("voice", "model", voice.model),
        ref_audio=None if speaker else path("voice", "ref_audio", voice.ref_audio),
        ref_text=None if speaker else text("voice", "ref_text", None),
        speaker=speaker,
        streaming_interval=number("voice", "streaming_interval", voice.streaming_interval),
        cache_limit_gb=number("voice", "cache_limit_gb", voice.cache_limit_gb),
        min_phrase_chars=integer("voice", "min_phrase_chars", voice.min_phrase_chars),
        max_phrase_chars=integer("voice", "max_phrase_chars", voice.max_phrase_chars),
        greetings=words("voice", "greetings", voice.greetings, allow_empty=False),
    )
    if voice.max_phrase_chars < voice.min_phrase_chars:
        raise fail("voice.max_phrase_chars", "должно быть не меньше voice.min_phrase_chars")

    journal = JournalSettings()
    journal = JournalSettings(
        transcripts=flag("journal", "transcripts", journal.transcripts),
        metrics=flag("journal", "metrics", journal.metrics),
        dir=path("journal", "dir", journal.dir),
    )

    return Settings(
        discord=DiscordSettings(guild_id=guild_id, whitelist=tuple(whitelist)),
        dialogue=dialogue,
        core=CoreSettings(port=port),
        speech=speech,
        llm=llm,
        voice=voice,
        journal=journal,
    )


# --- Файлы личности Клона ---


@dataclass(frozen=True)
class Persona:
    """Паспорт, few-shot примеры и лексикон произношения."""

    passport: str
    # Пары (Реплика, ответ Прототипа).
    fewshot: tuple[tuple[str, str], ...]
    # Пары (слово, запись для Голоса).
    lexicon: tuple[tuple[str, str], ...]


def resolve_local(path: Path) -> Path:
    """passport.md → passport.local.md, если личная версия есть (она в .gitignore)."""
    local = path.with_name(f"{path.stem}.local{path.suffix}")
    return local if local.exists() else path


def load_persona(config_dir: Path = CONFIG_DIR) -> Persona:
    passport_path = resolve_local(config_dir / "passport.md")
    fewshot_path = resolve_local(config_dir / "fewshot.jsonl")
    lexicon_path = resolve_local(config_dir / "lexicon.tsv")
    try:
        passport = passport_path.read_text(encoding="utf-8").strip()
        fewshot_lines = fewshot_path.read_text(encoding="utf-8").splitlines()
        lexicon_lines = lexicon_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as e:
        raise SettingsError(f"{e.filename}: файл личности не найден") from None
    if not passport:
        raise SettingsError(f"{passport_path}: Паспорт пуст")

    fewshot = []
    for n, line in enumerate(fewshot_lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            raise SettingsError(f"{fewshot_path}:{n}: некорректный JSON: {e}") from None
        user, answer = item.get("user"), item.get("assistant")
        if not isinstance(user, str) or not isinstance(answer, str) or not user or not answer:
            raise SettingsError(f"{fewshot_path}:{n}: нужны непустые строки user и assistant")
        fewshot.append((user, answer))

    lexicon = []
    for n, line in enumerate(lexicon_lines, 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2 or not all(p.strip() for p in parts):
            raise SettingsError(f"{lexicon_path}:{n}: нужна строка «слово<TAB>запись для Голоса»")
        lexicon.append((parts[0].strip(), parts[1].strip()))

    return Persona(passport=passport, fewshot=tuple(fewshot), lexicon=tuple(lexicon))
