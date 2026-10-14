from pathlib import Path

import pytest

from terebot.config import CONFIG_DIR, SettingsError, load_settings, resolve_settings_path

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
    settings = load_settings(CONFIG_DIR / "settings.toml")
    assert settings.discord.whitelist


def test_default_loads_resolved_file():
    assert load_settings() == load_settings(resolve_settings_path())


VALID = '[discord]\nguild_id = "123456789012345678"\nwhitelist = ["111111111111111111"]\n'


def test_resolve_without_local_uses_shared(tmp_path):
    (tmp_path / "settings.toml").write_text(VALID, encoding="utf-8")
    assert resolve_settings_path(tmp_path) == tmp_path / "settings.toml"


def test_resolve_prefers_local(tmp_path):
    (tmp_path / "settings.toml").write_text(VALID, encoding="utf-8")
    (tmp_path / "settings.local.toml").write_text(
        VALID.replace("111111111111111111", "444444444444444444"), encoding="utf-8"
    )
    path = resolve_settings_path(tmp_path)
    assert path == tmp_path / "settings.local.toml"
    assert load_settings(path).discord.whitelist == ("444444444444444444",)


def test_local_replaces_shared_without_merging(tmp_path):
    (tmp_path / "settings.toml").write_text(VALID, encoding="utf-8")
    (tmp_path / "settings.local.toml").write_text(
        '[discord]\nwhitelist = ["111111111111111111"]\n', encoding="utf-8"
    )
    with pytest.raises(SettingsError, match="settings.local.toml: поле discord.guild_id"):
        load_settings(resolve_settings_path(tmp_path))


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
