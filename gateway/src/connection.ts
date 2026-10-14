// Подключение к голосовому каналу, переподключение и запись в лог состояний и событий DAVE.
//
// События:
//   "ready" (VoiceConnection) — соединение готово; приёмник и плеер подключаются к нему заново,
//     потому что после переподключения это может быть другой объект;
//   "down" (reason) — соединение закрывается (/leave, автовыход, переподключение, остановка);
//     подписчики должны закрыть потоки и файлы;
//   "lost" (reason) — Клон потерял канал без команды (выкинули из канала, сеть не вернулась).

import { EventEmitter } from "node:events";

import {
  VoiceConnection,
  VoiceConnectionStatus,
  entersState,
  joinVoiceChannel,
} from "@discordjs/voice";
import type { VoiceBasedChannel } from "discord.js";

import { type Log, logger } from "./log.ts";

const READY_TIMEOUT_MS = 30_000;
// Сколько ждать, что @discordjs/voice сам восстановит соединение после Disconnected.
const RECOVER_TIMEOUT_MS = 5_000;

const log: Log = logger("voice");
const daveLog: Log = logger("dave");

export class ChannelConnection extends EventEmitter {
  #connection: VoiceConnection | null = null;
  #channel: VoiceBasedChannel | null = null;

  get channel(): VoiceBasedChannel | null {
    return this.#channel;
  }

  async join(channel: VoiceBasedChannel): Promise<void> {
    if (this.#connection) this.#teardown(`переход в «${channel.name}»`);
    this.#channel = channel;
    log(`подключение к «${channel.name}» (${channel.id})`);
    const connection = joinVoiceChannel({
      channelId: channel.id,
      guildId: channel.guild.id,
      adapterCreator: channel.guild.voiceAdapterCreator,
      // Оглохший бот не получает голос участников.
      selfDeaf: false,
      selfMute: false,
      debug: true,
    });
    this.#connection = connection;
    this.#watch(connection);
    try {
      await entersState(connection, VoiceConnectionStatus.Ready, READY_TIMEOUT_MS);
    } catch (e) {
      this.#teardown("не дождались Ready");
      this.#channel = null;
      throw new Error(`соединение с «${channel.name}» не стало Ready за ${READY_TIMEOUT_MS / 1000} с`, {
        cause: e,
      });
    }
  }

  // Переподключение к тому же каналу с новым соединением (решение watchdog).
  async reconnect(reason: string): Promise<void> {
    const channel = this.#channel;
    if (!channel) return;
    log(`переподключение: ${reason}`);
    try {
      await this.join(channel);
    } catch (e) {
      log(`переподключение не удалось: ${(e as Error).message}`);
      this.emit("lost", "переподключение не удалось");
    }
  }

  // false — Клон и так не был в канале.
  leave(reason: string): boolean {
    if (!this.#connection) return false;
    this.#teardown(reason);
    this.#channel = null;
    return true;
  }

  #teardown(reason: string): void {
    const connection = this.#connection;
    if (!connection) return;
    this.#connection = null;
    this.emit("down", reason);
    log(`отключение: ${reason}`);
    if (connection.state.status !== VoiceConnectionStatus.Destroyed) connection.destroy();
  }

  #watch(connection: VoiceConnection): void {
    connection.on("stateChange", (oldState, newState) => {
      if (oldState.status === newState.status) return;
      log(`состояние ${oldState.status} → ${newState.status}`);
      if (connection !== this.#connection) return;
      if (newState.status === VoiceConnectionStatus.Ready) {
        this.emit("ready", connection);
      } else if (newState.status === VoiceConnectionStatus.Disconnected) {
        void this.#recover(connection);
      }
    });
    connection.on("debug", (message: string) => {
      // [NW] [DAVE] … — переходы, MLS, ошибки расшифровки. Остальной debug (пинги WS) слишком шумный.
      if (message.includes("[DAVE]")) daveLog(message.replace(/^.*\[DAVE\] /, ""));
    });
    connection.on("transitioned", (transitionId: number) => daveLog(`transitioned id=${transitionId}`));
    connection.on("error", (e: Error) => log(`ошибка соединения: ${e.message}`));
  }

  // Стандартный приём из документации @discordjs/voice: если за 5 с соединение не перешло
  // в Signalling/Connecting само (например, Клона переместили), значит канал потерян.
  async #recover(connection: VoiceConnection): Promise<void> {
    try {
      await Promise.race([
        entersState(connection, VoiceConnectionStatus.Signalling, RECOVER_TIMEOUT_MS),
        entersState(connection, VoiceConnectionStatus.Connecting, RECOVER_TIMEOUT_MS),
      ]);
    } catch {
      if (connection !== this.#connection) return;
      this.#teardown("соединение потеряно");
      this.#channel = null;
      this.emit("lost", "соединение потеряно");
    }
  }
}
