"""Загрузка и проверка несекретных настроек.

Читается config/settings.local.toml, если он есть (в .gitignore, там реальные ID),
иначе config/settings.toml из git. Файлы не сливаются: локальный заменяет общий целиком.
Тот же выбор файла и ту же схему проверяет шлюз (gateway/src/config.ts).
"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

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
class Settings:
    discord: DiscordSettings
    dialogue: DialogueSettings = DialogueSettings()


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

    dialogue = raw.get("dialogue", {})
    if not isinstance(dialogue, dict):
        raise fail("[dialogue]", "должно быть таблицей")
    defaults = DialogueSettings()

    def words(field: str, fallback: tuple[str, ...], allow_empty: bool) -> tuple[str, ...]:
        value = dialogue.get(field)
        if value is None:
            return fallback
        if not isinstance(value, list) or not all(isinstance(w, str) and w.strip() for w in value):
            raise fail(f"dialogue.{field}", "должно быть списком непустых строк")
        if not value and not allow_empty:
            raise fail(f"dialogue.{field}", "пусто: Клона нечем позвать")
        return tuple(value)

    def number(field: str, fallback: float, integer: bool = False) -> float:
        value = dialogue.get(field)
        if value is None:
            return fallback
        valid_type = int if integer else (int, float)
        if isinstance(value, bool) or not isinstance(value, valid_type) or value <= 0:
            kind = "целым числом" if integer else "числом"
            raise fail(f"dialogue.{field}", f"должно быть {kind} больше 0")
        return value

    wake_threshold = number("wake_threshold", defaults.wake_threshold)
    if wake_threshold > 100:
        raise fail("dialogue.wake_threshold", "должно быть числом от 0 до 100")

    return Settings(
        discord=DiscordSettings(guild_id=guild_id, whitelist=tuple(whitelist)),
        dialogue=DialogueSettings(
            wake_words=words("wake_words", defaults.wake_words, allow_empty=False),
            wake_threshold=wake_threshold,
            wake_blocklist=words("wake_blocklist", defaults.wake_blocklist, allow_empty=True),
            window_s=number("window_s", defaults.window_s),
            barge_in_ms=int(number("barge_in_ms", defaults.barge_in_ms, integer=True)),
        ),
    )
