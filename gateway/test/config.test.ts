import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import {
  CONFIG_DIR,
  GATEWAY_DEFAULTS,
  SettingsError,
  loadSettings,
  resolveSettingsPath,
} from "../src/config.ts";

const REPO_ROOT = fileURLToPath(new URL("../../", import.meta.url));
const FIXTURE = join(REPO_ROOT, "tests", "fixtures", "settings_valid.toml");

// Загружает тот же файл Python-ядром (terebot.config), чтобы сравнить результаты.
// Без path ядро само выбирает файл, как при обычном старте.
function loadWithPython(path?: string): { guild_id: string; whitelist: string[] } {
  const code =
    "import json, sys; from pathlib import Path; from terebot.config import load_settings; " +
    "d = load_settings(Path(sys.argv[1]) if len(sys.argv) > 1 else None).discord; " +
    "print(json.dumps({'guild_id': d.guild_id, 'whitelist': list(d.whitelist)}))";
  const args = ["run", "--quiet", "python", "-c", code, ...(path ? [path] : [])];
  const out = execFileSync("uv", args, {
    cwd: REPO_ROOT,
    encoding: "utf8",
  });
  return JSON.parse(out);
}

function writeTemp(text: string, name = "settings.toml", dir = mkdtempSync(join(tmpdir(), "terebot-"))): string {
  const path = join(dir, name);
  writeFileSync(path, text);
  return path;
}

const VALID = '[discord]\nguild_id = "123456789012345678"\nwhitelist = ["111111111111111111"]\n';

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
  ["config/settings.toml", join(CONFIG_DIR, "settings.toml")],
  ["файл по умолчанию", undefined],
] as const) {
  test(`${name}: шлюз и ядро видят одинаковые настройки`, () => {
    const node = path ? loadSettings(path) : loadSettings();
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

test("без settings.local.toml читается settings.toml", () => {
  const shared = writeTemp(VALID);
  const dir = join(shared, "..");
  assert.equal(resolveSettingsPath(dir), shared);
});

test("settings.local.toml заменяет settings.toml целиком", () => {
  const shared = writeTemp(VALID);
  const dir = join(shared, "..");
  const local = writeTemp('[discord]\nwhitelist = ["111111111111111111"]\n', "settings.local.toml", dir);
  assert.equal(resolveSettingsPath(dir), local);
  assert.throws(() => loadSettings(resolveSettingsPath(dir)), /settings\.local\.toml: поле discord\.guild_id/);
});

test("без [gateway] и [recording] берутся значения по умолчанию", () => {
  const settings = loadSettings(writeTemp(VALID));
  assert.deepEqual(settings.gateway, GATEWAY_DEFAULTS);
  assert.equal(settings.recording.enabled, true);
  assert.equal(settings.recording.dir, join(REPO_ROOT, "data", "recordings"));
});

test("config/settings.toml: поля шлюза совпадают со значениями по умолчанию", () => {
  const settings = loadSettings(join(CONFIG_DIR, "settings.toml"));
  assert.deepEqual(settings.gateway, GATEWAY_DEFAULTS);
});

test("поля [gateway] и [recording] читаются", () => {
  const path = writeTemp(
    VALID +
      "[gateway]\necho = false\necho_delay_ms = 500\nautoleave_s = 5\n" +
      '[recording]\nenabled = false\ndir = "/tmp/rec"\n',
  );
  const settings = loadSettings(path);
  assert.equal(settings.gateway.echo, false);
  assert.equal(settings.gateway.echoDelayMs, 500);
  assert.equal(settings.gateway.autoleaveS, 5);
  assert.equal(settings.gateway.watchdogSilenceMs, GATEWAY_DEFAULTS.watchdogSilenceMs);
  assert.deepEqual(settings.recording, { enabled: false, dir: "/tmp/rec" });
});

test("неверный тип в [gateway]", () => {
  const path = writeTemp(VALID + '[gateway]\necho_delay_ms = "2000"\n');
  assert.throws(() => loadSettings(path), /gateway\.echo_delay_ms должно быть целым/);
});

test("неверный тип в [recording]", () => {
  const path = writeTemp(VALID + '[recording]\nenabled = "yes"\n');
  assert.throws(() => loadSettings(path), /recording\.enabled должно быть true или false/);
});
