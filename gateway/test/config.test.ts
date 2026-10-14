import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { DEFAULT_SETTINGS_PATH, SettingsError, loadSettings } from "../src/config.ts";

const REPO_ROOT = fileURLToPath(new URL("../../", import.meta.url));
const FIXTURE = join(REPO_ROOT, "tests", "fixtures", "settings_valid.toml");

// Загружает тот же файл Python-ядром (terebot.config), чтобы сравнить результаты.
function loadWithPython(path: string): { guild_id: string; whitelist: string[] } {
  const code =
    "import json, sys; from terebot.config import load_settings; " +
    "d = load_settings(sys.argv[1]).discord; " +
    "print(json.dumps({'guild_id': d.guild_id, 'whitelist': list(d.whitelist)}))";
  const out = execFileSync("uv", ["run", "--quiet", "python", "-c", code, path], {
    cwd: REPO_ROOT,
    encoding: "utf8",
  });
  return JSON.parse(out);
}

function writeTemp(text: string): string {
  const path = join(mkdtempSync(join(tmpdir(), "terebot-")), "settings.toml");
  writeFileSync(path, text);
  return path;
}

test("общий образец: whitelist из трёх ID", () => {
  const settings = loadSettings(FIXTURE);
  assert.equal(settings.discord.guildId, "123456789012345678");
  assert.deepEqual(settings.discord.whitelist, [
    "111111111111111111",
    "222222222222222222",
    "333333333333333333",
  ]);
});

for (const [name, path] of [
  ["общий образец", FIXTURE],
  ["config/settings.toml", DEFAULT_SETTINGS_PATH],
] as const) {
  test(`${name}: шлюз и ядро видят одинаковые настройки`, () => {
    const node = loadSettings(path);
    const py = loadWithPython(path);
    assert.equal(node.discord.guildId, py.guild_id);
    assert.deepEqual(node.discord.whitelist, py.whitelist);
  });
}

test("пустой whitelist", () => {
  const path = writeTemp('[discord]\nguild_id = "123456789012345678"\nwhitelist = []\n');
  assert.throws(() => loadSettings(path), (e: Error) => {
    assert.ok(e instanceof SettingsError);
    assert.match(e.message, /whitelist пуст/);
    assert.ok(e.message.includes(path));
    return true;
  });
});

test("отсутствующее поле", () => {
  const path = writeTemp('[discord]\nwhitelist = ["111111111111111111"]\n');
  assert.throws(() => loadSettings(path), /discord\.guild_id отсутствует/);
});

test("неверный тип: guild_id числом", () => {
  const path = writeTemp('[discord]\nguild_id = 123\nwhitelist = ["111111111111111111"]\n');
  assert.throws(() => loadSettings(path), /discord\.guild_id должно быть строкой/);
});

test("неверный тип: ID в whitelist числом", () => {
  const path = writeTemp('[discord]\nguild_id = "123456789012345678"\nwhitelist = [1]\n');
  assert.throws(() => loadSettings(path), /discord\.whitelist\[0\]/);
});
