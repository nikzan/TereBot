"""Загрузка и проверка несекретных настроек из config/settings.toml.

Тот же файл и ту же схему проверяет шлюз (gateway/src/config.ts).
"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.toml"

# Discord snowflake: целое число до 2^64, в настройках хранится строкой.
_SNOWFLAKE = re.compile(r"^\d{17,20}$")


class SettingsError(Exception):
    """Настройки отсутствуют или не соответствуют схеме."""


@dataclass(frozen=True)
class DiscordSettings:
    guild_id: str
    whitelist: tuple[str, ...]


@dataclass(frozen=True)
class Settings:
    discord: DiscordSettings


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> Settings:
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

    return Settings(discord=DiscordSettings(guild_id=guild_id, whitelist=tuple(whitelist)))
