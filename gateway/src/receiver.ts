// Раздельный приём голоса по участникам. Главный шов шлюза: эхо, запись и (на этапе 6)
// мост в Python-ядро подписываются на одни и те же события.
//
// События:
//   "segmentStart" {userId, t} — пошёл фрагмент речи (первый декодированный кадр подписки);
//   "frame" {userId, pcm, t}  — кадр 20 мс: моно s16, 48 кГц, 960 сэмплов; t — время получения, мс;
//   "segmentEnd" {userId, t}   — фрагмент закончился (1 с тишины, сбой или отключение).
//
// Подписка открывается по событию speaking start и закрывается после 1 с тишины.
// Watchdog следит, чтобы у говорящего участника шли кадры, и переподписывает или переподключает.
// Раз в минуту в лог пишутся счётчики пакетов по каждому участнику.

import { EventEmitter } from "node:events";

import OpusPkg from "@discordjs/opus";
import {
  type AudioReceiveStream,
  EndBehaviorType,
  RTP_OPUS_PAYLOAD_TYPE,
  type VoiceConnection,
  VoiceConnectionStatus,
} from "@discordjs/voice";

import { logger } from "./log.ts";
import { SAMPLE_RATE, stereoToMono } from "./mixer.ts";
import { type WatchdogAction, Watchdog } from "./watchdog.ts";

const { OpusEncoder } = OpusPkg;

export interface Frame {
  userId: string;
  pcm: Int16Array;
  t: number;
}

export interface SegmentEdge {
  userId: string;
  t: number;
}

export interface ReceiverOptions {
  silenceMs: number;
  maxResubscribes: number;
  // Решение watchdog переподключиться к каналу.
  onReconnect: (reason: string) => void;
}

const END_AFTER_SILENCE_MS = 1000;
const WATCHDOG_TICK_MS = 250;
const STATS_INTERVAL_MS = 60_000;

const log = logger("recv");
const statsLog = logger("stats");

// Счётчики за всё время работы шлюза (не сбрасываются при переподключении).
interface Counters {
  rtp: number; // RTP-пакеты Opus от участника, пришедшие по UDP, пока на него есть подписка
  seqLost: number; // пропуски в номерах RTP — пакеты, потерянные по дороге
  pushed: number; // пакеты, которые @discordjs/voice расшифровал и отдал в поток подписки
  decoded: number; // кадры, которые декодировал Opus
  decodeErrors: number;
  streamErrors: number;
  resubscribes: number;
}

interface UserState {
  stream: AudioReceiveStream | null;
  decoder: InstanceType<typeof OpusEncoder>;
  segmentOpen: boolean;
  pendingResubscribe: boolean;
  lastSeq: number | null;
  counters: Counters;
}

export class Receiver extends EventEmitter {
  readonly #options: ReceiverOptions;
  readonly #watchdog: Watchdog;
  #connection: VoiceConnection | null = null;
  #users = new Map<string, UserState>();
  #udpAttached = new WeakSet<object>();
  #watchdogTimer: ReturnType<typeof setInterval> | null = null;
  #statsTimer: ReturnType<typeof setInterval> | null = null;
  #detachConnection: (() => void) | null = null;

  constructor(options: ReceiverOptions) {
    super();
    this.#options = options;
    this.#watchdog = new Watchdog({
      silenceMs: options.silenceMs,
      maxResubscribes: options.maxResubscribes,
    });
  }

  // Вызывается на каждое "ready" соединения. Тот же объект после внутреннего переподключения
  // @discordjs/voice не переподписывается, но UDP-сокет у него новый — счётчик вешаем заново.
  attach(connection: VoiceConnection): void {
    if (this.#connection !== connection) {
      this.detach();
      this.#connection = connection;
      const speaking = connection.receiver.speaking;
      const onStart = (userId: string) => {
        this.#watchdog.speakingStart(userId, Date.now());
        this.#subscribe(userId);
      };
      const onEnd = (userId: string) => this.#watchdog.speakingEnd(userId, Date.now());
      speaking.on("start", onStart);
      speaking.on("end", onEnd);
      this.#detachConnection = () => {
        speaking.off("start", onStart);
        speaking.off("end", onEnd);
      };
      this.#watchdogTimer = setInterval(() => this.#apply(this.#watchdog.tick(Date.now())), WATCHDOG_TICK_MS);
      this.#statsTimer ??= setInterval(() => this.logStats(), STATS_INTERVAL_MS);
      log("приёмник подключён");
    }
    this.#attachUdpCounter(connection);
  }

  // Соединение закрывается: закрыть все подписки и фрагменты. Счётчики остаются.
  detach(): void {
    if (!this.#connection) return;
    this.#detachConnection?.();
    this.#detachConnection = null;
    if (this.#watchdogTimer) clearInterval(this.#watchdogTimer);
    this.#watchdogTimer = null;
    this.#watchdog.reset();
    const t = Date.now();
    for (const [userId, user] of this.#users) {
      user.pendingResubscribe = false;
      this.#endSegment(userId, user, t);
      user.stream?.destroy();
      user.stream = null;
      user.lastSeq = null;
    }
    this.#connection = null;
    log("приёмник отключён");
  }

  // Полная остановка: финальные счётчики в лог.
  close(): void {
    this.detach();
    if (this.#statsTimer) clearInterval(this.#statsTimer);
    this.#statsTimer = null;
    this.logStats();
  }

  logStats(): void {
    for (const [userId, user] of this.#users) {
      const c = user.counters;
      const sent = c.rtp + c.seqLost;
      const lostPct = sent === 0 ? 0 : (100 * (sent - c.decoded)) / sent;
      statsLog(
        `user=${userId} rtp=${c.rtp} seq_lost=${c.seqLost} dave_dropped=${c.rtp - c.pushed} ` +
          `decoded=${c.decoded} decode_err=${c.decodeErrors} stream_err=${c.streamErrors} ` +
          `resub=${c.resubscribes} lost=${lostPct.toFixed(1)}%`,
      );
    }
  }

  #user(userId: string): UserState {
    let user = this.#users.get(userId);
    if (!user) {
      user = {
        stream: null,
        decoder: new OpusEncoder(SAMPLE_RATE, 2),
        segmentOpen: false,
        pendingResubscribe: false,
        lastSeq: null,
        counters: {
          rtp: 0,
          seqLost: 0,
          pushed: 0,
          decoded: 0,
          decodeErrors: 0,
          streamErrors: 0,
          resubscribes: 0,
        },
      };
      this.#users.set(userId, user);
    }
    return user;
  }

  #subscribe(userId: string): void {
    const connection = this.#connection;
    if (!connection) return;
    const user = this.#user(userId);
    if (user.stream) return;
    const stream = connection.receiver.subscribe(userId, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: END_AFTER_SILENCE_MS },
    });
    user.stream = stream;

    stream.on("data", (packet: Buffer) => {
      user.counters.pushed++;
      let pcm: Int16Array;
      try {
        pcm = stereoToMono(user.decoder.decode(packet));
      } catch (e) {
        user.counters.decodeErrors++;
        log(`user=${userId} ошибка декодирования Opus: ${(e as Error).message}`);
        return;
      }
      const t = Date.now();
      user.counters.decoded++;
      this.#watchdog.frame(userId, t);
      if (!user.segmentOpen) {
        user.segmentOpen = true;
        this.emit("segmentStart", { userId, t } satisfies SegmentEdge);
      }
      this.emit("frame", { userId, pcm, t } satisfies Frame);
    });

    stream.on("error", (e: Error) => {
      user.counters.streamErrors++;
      log(`user=${userId} поток подписки упал: ${e.message}`);
      this.#apply(this.#watchdog.streamError(userId, Date.now()));
    });

    stream.once("close", () => {
      if (user.stream !== stream) return;
      user.stream = null;
      this.#endSegment(userId, user, Date.now());
      // Новую подписку открываем только после close старой: @discordjs/voice по close
      // удаляет подписку из своей таблицы по user ID и иначе удалил бы уже новую.
      if (user.pendingResubscribe) {
        user.pendingResubscribe = false;
        this.#subscribe(userId);
      }
    });
  }

  #endSegment(userId: string, user: UserState, t: number): void {
    if (!user.segmentOpen) return;
    user.segmentOpen = false;
    this.emit("segmentEnd", { userId, t } satisfies SegmentEdge);
  }

  #apply(actions: WatchdogAction[]): void {
    for (const action of actions) {
      if (action.kind === "reconnect") {
        log(`user=${action.userId} watchdog: переподключение (${action.reason})`);
        this.#options.onReconnect(action.reason);
        return;
      }
      const user = this.#user(action.userId);
      user.counters.resubscribes++;
      log(`user=${action.userId} watchdog: переподписка (${action.reason})`);
      if (user.stream && !user.stream.destroyed) {
        user.pendingResubscribe = true;
        user.stream.destroy();
      } else if (!user.stream) {
        this.#subscribe(action.userId);
      } else {
        // Поток уже уничтожен (упал с ошибкой), close ещё впереди — подпишемся по нему.
        user.pendingResubscribe = true;
      }
    }
  }

  // Свой слушатель на UDP-сокете соединения: считает RTP-пакеты и пропуски номеров по участникам.
  // Регистрируется после слушателя VoiceReceiver, поэтому подписка по speaking start уже существует.
  #attachUdpCounter(connection: VoiceConnection): void {
    const state = connection.state;
    if (state.status !== VoiceConnectionStatus.Ready) return;
    const networkingState = state.networking.state;
    if (!("udp" in networkingState)) return;
    const udp = networkingState.udp;
    if (this.#udpAttached.has(udp)) return;
    this.#udpAttached.add(udp);
    udp.on("message", (msg: Buffer) => {
      if (msg.length <= 12 || (msg[1]! & 0x7f) !== RTP_OPUS_PAYLOAD_TYPE) return;
      const userId = connection.receiver.ssrcMap.get(msg.readUInt32BE(8))?.userId;
      if (!userId || !connection.receiver.subscriptions.has(userId)) return;
      const user = this.#user(userId);
      const seq = msg.readUInt16BE(2);
      if (user.lastSeq !== null) {
        const gap = (seq - user.lastSeq - 1) & 0xffff;
        // Большой «пропуск» — это переупорядочивание или новый поток, а не потери.
        if (gap > 0 && gap < 1000) user.counters.seqLost += gap;
      }
      user.lastSeq = seq;
      user.counters.rtp++;
    });
  }
}
