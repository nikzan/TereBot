// Загрузка и проверка несекретных настроек.
// Читается config/settings.local.toml, если он есть (в .gitignore, там реальные ID),
// иначе config/settings.toml из git. Файлы не сливаются: локальный заменяет общий целиком.
// Тот же выбор файла и ту же схему проверяет ядро (src/terebot/config.py).

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
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

export interface Settings {
  discord: DiscordSettings;
}

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

  return { discord: { guildId, whitelist: whitelist as string[] } };
}
