from pathlib import Path

import pytest

from terebot.config import DEFAULT_SETTINGS_PATH, SettingsError, load_settings

FIXTURES = Path(__file__).parent / "fixtures"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "settings.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_file():
    settings = load_settings(FIXTURES / "settings_valid.toml")
    assert settings.discord.guild_id == "123456789012345678"
    assert settings.discord.whitelist == (
        "111111111111111111",
        "222222222222222222",
        "333333333333333333",
    )


def test_repo_settings_load():
    settings = load_settings(DEFAULT_SETTINGS_PATH)
    assert settings.discord.whitelist


def test_empty_whitelist(tmp_path):
    path = write(tmp_path, '[discord]\nguild_id = "123456789012345678"\nwhitelist = []\n')
    with pytest.raises(SettingsError, match="whitelist пуст") as e:
        load_settings(path)
    assert str(path) in str(e.value)


def test_missing_field(tmp_path):
    path = write(tmp_path, '[discord]\nwhitelist = ["111111111111111111"]\n')
    with pytest.raises(SettingsError, match="discord.guild_id отсутствует") as e:
        load_settings(path)
    assert str(path) in str(e.value)


def test_missing_section(tmp_path):
    with pytest.raises(SettingsError, match=r"\[discord\] отсутствует"):
        load_settings(write(tmp_path, ""))


@pytest.mark.parametrize(
    "body, field",
    [
        ('guild_id = 123456789012345678\nwhitelist = ["111111111111111111"]', "discord.guild_id"),
        ('guild_id = "123456789012345678"\nwhitelist = "111111111111111111"', "discord.whitelist"),
        ('guild_id = "123456789012345678"\nwhitelist = [111111111111111111]', "whitelist[0]"),
    ],
)
def test_wrong_type(tmp_path, body, field):
    with pytest.raises(SettingsError, match=field.replace("[", r"\[")):
        load_settings(write(tmp_path, f"[discord]\n{body}\n"))


def test_missing_file(tmp_path):
    with pytest.raises(SettingsError, match="не найден"):
        load_settings(tmp_path / "nope.toml")
