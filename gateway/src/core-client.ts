// Связь с Python-ядром (ADR 0002): шлюз — WebSocket-клиент, ядро — сервер на 127.0.0.1:[core].port.
// Шлюз шлёт ядру Сессию, участников, границы речи и звук участников из whitelist; ядро шлёт
// звук ответа по фразам. Звук ответа идёт через очередь фраз (speech-out.ts) по таймеру 20 мс,
// а её события — начало ответа, конец фразы и ответа — уходят обратно в ядро.
// Без ядра шлюз работает как раньше и пробует подключиться каждые 3 с.

import {
  type Message,
  ProtocolError,
  decodeAudio,
  decodeMessage,
  encodeAudio,
  encodeMessage,
} from "./core-protocol.ts";
import { logger } from "./log.ts";
import { FRAME_MS, monoToStereo } from "./mixer.ts";
import type { Player } from "./player.ts";
import type { Frame, Receiver, SegmentEdge } from "./receiver.ts";
import { SpeechOut } from "./speech-out.ts";

const RECONNECT_MS = 3000;
const TICK_MS = 10;
// Если таймер отстал сильнее (процесс подвис), догонять не пытаемся — перескакиваем.
const MAX_LAG_MS = 200;
const RETRY_LOG_MS = 60_000;

const log = logger("core");

export interface SessionInfo {
  name: string;
  t: number;
}

export interface Participant {
  user_id: string;
  name: string;
}

export interface CoreClientOptions {
  port: number;
  whitelist: readonly string[];
  receiver: Receiver;
  player: Player;
  // Текущее состояние для hello после (пере)подключения.
  state: () => { session: SessionInfo | null; participants: Participant[] };
}

export class CoreClient {
  readonly #options: CoreClientOptions;
  readonly #whitelist: ReadonlySet<string>;
  #speech = new SpeechOut();
  #ws: WebSocket | null = null;
  #open = false;
  #stopped = true;
  #retryTimer: ReturnType<typeof setTimeout> | null = null;
  #tickTimer: ReturnType<typeof setInterval> | null = null;
  #next = 0;
  #failures = 0;
  #lastRetryLog = 0;

  constructor(options: CoreClientOptions) {
    this.#options = options;
    this.#whitelist = new Set(options.whitelist);
    const { receiver } = options;
    receiver.on("segmentStart", (e: SegmentEdge) => this.#edge("segment_start", e));
    receiver.on("segmentEnd", (e: SegmentEdge) => this.#edge("segment_end", e));
    receiver.on("frame", (e: Frame) => {
      if (!this.#open || !this.#whitelist.has(e.userId)) return;
      this.#ws!.send(encodeAudio({ kind: "participant", userId: e.userId, t: e.t, pcm: e.pcm }));
    });
  }

  start(): void {
    this.#stopped = false;
    this.#connect();
  }

  stop(): void {
    this.#stopped = true;
    if (this.#retryTimer) clearTimeout(this.#retryTimer);
    this.#retryTimer = null;
    this.#ws?.close();
  }

  sessionStart(session: SessionInfo): void {
    this.#send({ type: "session_start", session: { name: session.name, t: session.t }, t: session.t });
  }

  sessionEnd(t: number): void {
    // Канала больше нет: недозвучавший ответ выбрасывается.
    this.#stopSpeech(t);
    this.#send({ type: "session_end", t });
  }

  participantJoined(p: Participant, t: number): void {
    if (this.#whitelist.has(p.user_id)) {
      this.#send({ type: "participant_joined", user_id: p.user_id, name: p.name, t });
    }
  }

  participantLeft(userId: string, t: number): void {
    if (this.#whitelist.has(userId)) this.#send({ type: "participant_left", user_id: userId, t });
  }

  #edge(type: "segment_start" | "segment_end", e: SegmentEdge): void {
    if (this.#whitelist.has(e.userId)) this.#send({ type, user_id: e.userId, t: e.t });
  }

  #send(message: Message): void {
    if (this.#open) this.#ws!.send(encodeMessage(message));
  }

  #connect(): void {
    const url = `ws://127.0.0.1:${this.#options.port}`;
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    this.#ws = ws;
    ws.addEventListener("open", () => {
      this.#open = true;
      // Номера ответов у каждого процесса ядра свои: после его перезапуска они начинаются
      // заново, и прежний список выброшенных ответов глушил бы новые.
      this.#speech = new SpeechOut();
      log(`подключено к ядру ${url}` + (this.#failures ? ` (после ${this.#failures} попыток)` : ""));
      this.#failures = 0;
      const { session, participants } = this.#options.state();
      this.#send({
        type: "hello",
        session: session && { name: session.name, t: session.t },
        participants: participants
          .filter((p) => this.#whitelist.has(p.user_id))
          .map((p) => ({ user_id: p.user_id, name: p.name })),
        t: Date.now(),
      });
      this.#startTicking();
    });
    ws.addEventListener("message", (e) => this.#receive(e.data));
    // При отказе в соединении встроенный WebSocket Node шлёт только error, без close,
    // поэтому обрыв обрабатывается по первому из двух событий.
    const down = () => {
      if (this.#ws !== ws) return;
      const wasOpen = this.#open;
      this.#open = false;
      this.#ws = null;
      this.#stopTicking();
      this.#stopSpeech(Date.now());
      if (wasOpen) log("связь с ядром потеряна");
      else this.#logFailure(url);
      ws.close();
      if (!this.#stopped) this.#retryTimer = setTimeout(() => this.#connect(), RECONNECT_MS);
    };
    ws.addEventListener("close", down);
    ws.addEventListener("error", down);
  }

  #logFailure(url: string): void {
    this.#failures++;
    const now = Date.now();
    if (this.#failures === 1 || now - this.#lastRetryLog >= RETRY_LOG_MS) {
      this.#lastRetryLog = now;
      log(`ядро на ${url} недоступно (попыток: ${this.#failures}), повтор каждые ${RECONNECT_MS / 1000} с`);
    }
  }

  #receive(data: unknown): void {
    try {
      if (data instanceof ArrayBuffer) {
        const frame = decodeAudio(Buffer.from(data));
        if (frame.kind === "response") this.#speech.audio(frame.responseId, frame.phrase, frame.pcm);
        return;
      }
      const m = decodeMessage(String(data));
      const now = Date.now();
      switch (m.type) {
        case "phrase_begin":
          this.#speech.phraseBegin(m.response_id as number, m.phrase as number);
          break;
        case "phrase_end_marker":
          this.#speech.phraseEndMarker(m.response_id as number, m.phrase as number);
          break;
        case "response_end":
          this.#speech.responseEnd(m.response_id as number);
          break;
        case "stop_after_phrase":
          this.#emit(this.#speech.stopAfterPhrase(m.response_id as number, now));
          break;
        case "stop_now":
          this.#stopSpeech(now);
          break;
        default:
          log(`неожиданное сообщение от ядра: ${m.type}`);
      }
    } catch (e) {
      if (!(e instanceof ProtocolError)) throw e;
      log(`ошибка протокола: ${e.message}`);
    }
  }

  #stopSpeech(now: number): void {
    const events = this.#speech.stopNow(now);
    this.#options.player.stop();
    this.#emit(events);
  }

  #emit(events: Message[]): void {
    for (const e of events) this.#send(e);
  }

  #startTicking(): void {
    this.#next = Date.now();
    this.#tickTimer ??= setInterval(() => this.#step(), TICK_MS);
  }

  #stopTicking(): void {
    if (this.#tickTimer) clearInterval(this.#tickTimer);
    this.#tickTimer = null;
  }

  // Кадры выдаются по виртуальным часам с шагом 20 мс, как в режиме эха.
  #step(): void {
    const now = Date.now();
    if (now - this.#next > MAX_LAG_MS) this.#next = now;
    while (this.#next <= now) {
      const { frame, events } = this.#speech.pull(this.#next);
      if (frame) this.#options.player.write(monoToStereo(frame));
      this.#emit(events);
      this.#next += FRAME_MS;
    }
  }
}
