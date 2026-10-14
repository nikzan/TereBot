// Discord-шлюз Клона. Этап 1: войти в Discord и убедиться, что бот на тестовом сервере.

import { Client, Events, GatewayIntentBits } from "discord.js";

import { type Settings, SettingsError, loadSettings, resolveSettingsPath } from "./config.ts";

const SETUP_DOC = "docs/discord-setup.md";

function die(message: string): never {
  console.error(message);
  process.exit(1);
}

const token = process.env.DISCORD_TOKEN?.trim();
if (!token) {
  die(
    "Не задан DISCORD_TOKEN. Скопируйте .env.example в .env в корне репозитория " +
      `и впишите токен бота (см. ${SETUP_DOC}).`,
  );
}

// Токен не должен попасть в лог даже внутри сообщения об ошибке.
const redact = (text: string) => text.replaceAll(token, "<DISCORD_TOKEN>");

const settingsPath = resolveSettingsPath();

function loadSettingsOrDie(): Settings {
  try {
    return loadSettings(settingsPath);
  } catch (e) {
    if (e instanceof SettingsError) die(`Ошибка настроек: ${e.message}`);
    throw e;
  }
}

const settings = loadSettingsOrDie();
console.log(`Настройки: ${settingsPath}`);

const client = new Client({
  intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates],
});

client.once(Events.ClientReady, (ready) => {
  console.log(`Бот онлайн: ${ready.user.tag}`);
  const guild = ready.guilds.cache.get(settings.discord.guildId);
  if (guild) {
    console.log(`Сервер: ${guild.name} (${guild.id})`);
  } else {
    console.warn(
      `Бот не приглашён на сервер ${settings.discord.guildId} из ${settingsPath}. ` +
        `Проверьте guild_id и пригласите бота по инструкции ${SETUP_DOC}, раздел 2.`,
    );
  }
});

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, async () => {
    await client.destroy();
    process.exit(0);
  });
}

try {
  await client.login(token);
} catch (e) {
  die(`Не удалось войти в Discord: ${redact(e instanceof Error ? e.message : String(e))}`);
}
