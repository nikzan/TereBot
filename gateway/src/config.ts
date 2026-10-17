// Загрузка и проверка несекретных настроек.
// Читается config/settings.local.toml, если он есть (в .gitignore, там реальные ID),
// иначе config/settings.toml из git. Файлы не сливаются: локальный заменяет общий целиком.
// Тот же выбор файла и ту же схему проверяет ядро (src/terebot/config.py).

import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "smol-toml";

export const CONFIG_DIR = fileURLToPath(new URL("../../config/", import.meta.url));

export function resolveSettingsPath(configDir: string = CONFIG_DIR): string {
  const local = join(configDir, "settings.local.toml");
  return existsSync(local) ? local : join(configDir, "settings.toml");
}

// Discord snowflake: целое число до 2^64, в настройках хранится строкой.
const SNOWFLAKE = /^\d{17,20}$/;

export class SettingsError extends Error {
  override name = "SettingsError";
}

export interface DiscordSettings {
  guildId: string;
  whitelist: readonly string[];
}

export interface GatewaySettings {
  echo: boolean;
  echoDelayMs: number;
  autoleaveS: number;
  watchdogSilenceMs: number;
  watchdogMaxResubscribes: number;
}

export interface RecordingSettings {
  enabled: boolean;
  // Абсолютный путь; в файле настроек — относительно корня репозитория.
  dir: string;
}

export interface CoreSettings {
  // Ядро — WebSocket-сервер на 127.0.0.1:port.
  port: number;
}

export interface Settings {
  discord: DiscordSettings;
  gateway: GatewaySettings;
  recording: RecordingSettings;
  core: CoreSettings;
}

// Секции [gateway] и [recording] необязательны: недостающие поля берутся отсюда.
// Так settings.local.toml, где есть только [discord], остаётся рабочим.
export const GATEWAY_DEFAULTS: GatewaySettings = {
  echo: true,
  echoDelayMs: 2000,
  autoleaveS: 30,
  watchdogSilenceMs: 3000,
  watchdogMaxResubscribes: 3,
};

export const RECORDING_DEFAULTS = { enabled: true, dir: "data/recordings" } as const;

export const CORE_DEFAULTS: CoreSettings = { port: 8765 };

export const REPO_ROOT = fileURLToPath(new URL("../../", import.meta.url));

function isTable(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function loadSettings(path: string = resolveSettingsPath()): Settings {
  let raw: Record<string, unknown>;
  try {
    raw = parse(readFileSync(path, "utf8"));
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") {
      throw new SettingsError(`${path}: файл настроек не найден`);
    }
    throw new SettingsError(`${path}: некорректный TOML: ${(e as Error).message}`);
  }

  const fail = (field: string, problem: string) =>
    new SettingsError(`${path}: поле ${field} ${problem}`);

  const discord = raw.discord;
  if (!isTable(discord)) throw fail("[discord]", "отсутствует");

  const guildId = discord.guild_id;
  if (guildId === undefined) throw fail("discord.guild_id", "отсутствует");
  if (typeof guildId !== "string" || !SNOWFLAKE.test(guildId)) {
    throw fail("discord.guild_id", "должно быть строкой с Discord ID из 17–20 цифр");
  }

  const whitelist = discord.whitelist;
  if (whitelist === undefined) throw fail("discord.whitelist", "отсутствует");
  if (!Array.isArray(whitelist)) {
    throw fail("discord.whitelist", "должно быть списком строк с Discord user ID");
  }
  if (whitelist.length === 0) {
    throw fail("discord.whitelist", "пусто: whitelist пуст, Клону некого слушать");
  }
  whitelist.forEach((userId, i) => {
    if (typeof userId !== "string" || !SNOWFLAKE.test(userId)) {
      throw fail(`discord.whitelist[${i}]`, "должно быть строкой с Discord ID из 17–20 цифр");
    }
  });

  const optionalTable = (name: string): Record<string, unknown> => {
    const table = raw[name];
    if (table === undefined) return {};
    if (!isTable(table)) throw fail(`[${name}]`, "должно быть таблицей");
    return table;
  };

  const bool = (table: Record<string, unknown>, field: string, fallback: boolean): boolean => {
    const value = table[field.split(".")[1]!];
    if (value === undefined) return fallback;
    if (typeof value !== "boolean") throw fail(field, "должно быть true или false");
    return value;
  };

  const int = (table: Record<string, unknown>, field: string, fallback: number, min: number): number => {
    const value = table[field.split(".")[1]!];
    if (value === undefined) return fallback;
    // smol-toml возвращает целые TOML как number (или bigint для очень больших).
    if (typeof value !== "number" || !Number.isInteger(value) || value < min) {
      throw fail(field, `должно быть целым числом не меньше ${min}`);
    }
    return value;
  };

  const gw = optionalTable("gateway");
  const gateway: GatewaySettings = {
    echo: bool(gw, "gateway.echo", GATEWAY_DEFAULTS.echo),
    echoDelayMs: int(gw, "gateway.echo_delay_ms", GATEWAY_DEFAULTS.echoDelayMs, 0),
    autoleaveS: int(gw, "gateway.autoleave_s", GATEWAY_DEFAULTS.autoleaveS, 1),
    watchdogSilenceMs: int(gw, "gateway.watchdog_silence_ms", GATEWAY_DEFAULTS.watchdogSilenceMs, 100),
    watchdogMaxResubscribes: int(
      gw,
      "gateway.watchdog_max_resubscribes",
      GATEWAY_DEFAULTS.watchdogMaxResubscribes,
      1,
    ),
  };

  const rec = optionalTable("recording");
  const dir = rec.dir ?? RECORDING_DEFAULTS.dir;
  if (typeof dir !== "string" || dir === "") {
    throw fail("recording.dir", "должно быть непустой строкой с путём");
  }
  const recording: RecordingSettings = {
    enabled: bool(rec, "recording.enabled", RECORDING_DEFAULTS.enabled),
    dir: resolve(REPO_ROOT, dir),
  };

  // Остальные секции ядра ([speech], [llm], …) шлюз не читает.
  const coreTable = optionalTable("core");
  const port = int(coreTable, "core.port", CORE_DEFAULTS.port, 1);
  if (port > 65535) throw fail("core.port", "должно быть номером порта до 65535");

  return { discord: { guildId, whitelist: whitelist as string[] }, gateway, recording, core: { port } };
}
