// Discord-шлюз Клона. Этап 2: /join и /leave, раздельный приём голоса, эхо и запись WAV.

import { join } from "node:path";

import { Client, Events, GatewayIntentBits, type VoiceBasedChannel } from "discord.js";

import { AutoLeave, handleCommand, registerCommands } from "./commands.ts";
import { type Settings, SettingsError, loadSettings, resolveSettingsPath } from "./config.ts";
import { ChannelConnection } from "./connection.ts";
import { Echo } from "./echo.ts";
import { logger } from "./log.ts";
import { Player } from "./player.ts";
import { Receiver } from "./receiver.ts";
import { Recorder, sessionDirName } from "./recorder.ts";

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

const log = logger("gateway");
const { gateway: gw, recording } = settings;
log(
  `эхо: ${gw.echo ? `вкл, задержка ${gw.echoDelayMs} мс` : "выкл"}; ` +
    `запись: ${recording.enabled ? recording.dir : "выкл"}`,
);

const connection = new ChannelConnection();
const receiver = new Receiver({
  silenceMs: gw.watchdogSilenceMs,
  maxResubscribes: gw.watchdogMaxResubscribes,
  onReconnect: (reason) => void connection.reconnect(reason),
});
const player = new Player();
const echo = gw.echo ? new Echo(receiver, player, gw.echoDelayMs) : null;

// Сессия — от /join до /leave или автовыхода. Переподключение её не прерывает:
// запись продолжается в тот же каталог.
let recorder: Recorder | null = null;

receiver.on("segmentStart", (e) => recorder?.segmentStart(e));
receiver.on("frame", (e) => recorder?.frame(e));
receiver.on("segmentEnd", (e) => recorder?.segmentEnd(e));

connection.on("ready", (vc) => {
  receiver.attach(vc);
  player.attach(vc);
  echo?.start();
});
// Любое закрытие соединения, включая переподключение: потоки и файлы закрываются.
connection.on("down", () => {
  echo?.stop();
  receiver.detach();
  recorder?.closeAll();
});
connection.on("lost", (reason: string) => endSession(reason));

const autoLeave = new AutoLeave(gw.autoleaveS * 1000, () => {
  if (connection.leave(`в канале нет людей ${gw.autoleaveS} с`)) endSession("автовыход");
});

function humansIn(channel: VoiceBasedChannel): number {
  return channel.members.filter((m) => !m.user.bot).size;
}

function startSession(): void {
  if (recorder || !recording.enabled) return;
  const now = new Date();
  recorder = new Recorder(join(recording.dir, sessionDirName(now)), settings.discord.whitelist, now.getTime());
  log(`Сессия: запись в ${recorder.dir}`);
}

function endSession(reason: string): void {
  autoLeave.cancel();
  recorder?.closeAll();
  if (recorder) log(`Сессия закончена (${reason}), записи в ${recorder.dir}`);
  recorder = null;
  receiver.logStats();
}

const actions = {
  async join(channelId: string): Promise<string> {
    const channel = await client.channels.fetch(channelId);
    if (!channel?.isVoiceBased()) throw new Error(`${channelId} — не голосовой канал`);
    startSession();
    try {
      await connection.join(channel);
    } catch (e) {
      endSession("подключение не удалось");
      throw e;
    }
    autoLeave.update(humansIn(channel));
    return channel.name;
  },
  async leave(): Promise<boolean> {
    if (!connection.leave("/leave")) return false;
    endSession("/leave");
    return true;
  },
};

client.on(Events.InteractionCreate, async (interaction) => {
  if (!interaction.isChatInputCommand() || interaction.guildId !== settings.discord.guildId) return;
  try {
    await handleCommand(interaction, settings.discord.whitelist, actions);
  } catch (e) {
    log(`ошибка команды /${interaction.commandName}: ${redact(e instanceof Error ? e.message : String(e))}`);
  }
});

client.on(Events.VoiceStateUpdate, (oldState, newState) => {
  const channel = connection.channel;
  if (!channel || (oldState.channelId !== channel.id && newState.channelId !== channel.id)) return;
  autoLeave.update(humansIn(channel));
});

client.once(Events.ClientReady, async (ready) => {
  console.log(`Бот онлайн: ${ready.user.tag}`);
  const guild = ready.guilds.cache.get(settings.discord.guildId);
  if (guild) {
    console.log(`Сервер: ${guild.name} (${guild.id})`);
    try {
      await registerCommands(guild);
      log("команды /join и /leave зарегистрированы на сервере");
    } catch (e) {
      log(`не удалось зарегистрировать команды: ${redact(e instanceof Error ? e.message : String(e))}`);
    }
  } else {
    console.warn(
      `Бот не приглашён на сервер ${settings.discord.guildId} из ${settingsPath}. ` +
        `Проверьте guild_id и пригласите бота по инструкции ${SETUP_DOC}, раздел 2.`,
    );
  }
});

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, async () => {
    // Сначала файлы: заголовки WAV должны быть исправлены до выхода.
    connection.leave(signal);
    endSession(signal);
    receiver.close();
    await client.destroy();
    process.exit(0);
  });
}

try {
  await client.login(token);
} catch (e) {
  die(`Не удалось войти в Discord: ${redact(e instanceof Error ? e.message : String(e))}`);
}
