// Запись раздельных потоков участников из whitelist: data/recordings/<Сессия>/<user_id>.wav
// (PCM 48 кГц, моно, 16 бит) и рядом <user_id>.segments.jsonl — по строке на фрагмент речи:
// {start_ms, duration_ms, sample_offset}. start_ms — от начала Сессии, sample_offset — сэмпл в WAV.
// Тишина между фрагментами в файл не пишется.
//
// В заголовке WAV сначала нулевые размеры; они исправляются в конце каждого фрагмента и при закрытии,
// поэтому файл остаётся корректным и при штатной остановке, и при обрыве посреди Сессии.
// После закрытия (переподключение) запись продолжается в тот же файл — дописыванием в конец.

import {
  appendFileSync,
  closeSync,
  existsSync,
  fstatSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readSync,
  writeSync,
} from "node:fs";
import { join } from "node:path";

import { SAMPLE_RATE } from "./mixer.ts";
import type { Frame, SegmentEdge } from "./receiver.ts";

const HEADER_BYTES = 44;
const BYTES_PER_SAMPLE = 2;

function wavHeader(dataBytes: number): Buffer {
  const h = Buffer.alloc(HEADER_BYTES);
  h.write("RIFF", 0, "ascii");
  h.writeUInt32LE(36 + dataBytes, 4);
  h.write("WAVE", 8, "ascii");
  h.write("fmt ", 12, "ascii");
  h.writeUInt32LE(16, 16); // размер блока fmt
  h.writeUInt16LE(1, 20); // PCM
  h.writeUInt16LE(1, 22); // моно
  h.writeUInt32LE(SAMPLE_RATE, 24);
  h.writeUInt32LE(SAMPLE_RATE * BYTES_PER_SAMPLE, 28); // байт в секунду
  h.writeUInt16LE(BYTES_PER_SAMPLE, 32); // выравнивание блока
  h.writeUInt16LE(16, 34); // бит на сэмпл
  h.write("data", 36, "ascii");
  h.writeUInt32LE(dataBytes, 40);
  return h;
}

export class WavWriter {
  readonly path: string;
  #fd: number | null;
  #dataBytes: number;

  // Если файл уже есть (его писали до переподключения), дописываем в конец.
  constructor(path: string) {
    this.path = path;
    if (existsSync(path)) {
      this.#fd = openSync(path, "r+");
      const head = Buffer.alloc(4);
      readSync(this.#fd, head, 0, 4, 0);
      if (head.toString("ascii") !== "RIFF") throw new Error(`${path}: не WAV, дописывать не буду`);
      this.#dataBytes = fstatSync(this.#fd).size - HEADER_BYTES;
    } else {
      this.#fd = openSync(path, "w");
      this.#dataBytes = 0;
      writeSync(this.#fd, wavHeader(0), 0, HEADER_BYTES, 0);
    }
  }

  get samples(): number {
    return this.#dataBytes / BYTES_PER_SAMPLE;
  }

  write(pcm: Int16Array): void {
    if (this.#fd === null) throw new Error(`${this.path}: файл уже закрыт`);
    const bytes = Buffer.from(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    writeSync(this.#fd, bytes, 0, bytes.length, HEADER_BYTES + this.#dataBytes);
    this.#dataBytes += bytes.length;
  }

  // Записать в заголовок текущие размеры.
  flush(): void {
    if (this.#fd === null) return;
    writeSync(this.#fd, wavHeader(this.#dataBytes), 0, HEADER_BYTES, 0);
    fsyncSync(this.#fd);
  }

  close(): void {
    if (this.#fd === null) return;
    this.flush();
    closeSync(this.#fd);
    this.#fd = null;
  }
}

export interface Segment {
  start_ms: number;
  duration_ms: number;
  sample_offset: number;
}

interface Track {
  wav: WavWriter;
  segmentsPath: string;
  // Начало текущего фрагмента, если он идёт.
  open: { startMs: number; sampleOffset: number } | null;
}

export class Recorder {
  readonly dir: string;
  readonly #whitelist: ReadonlySet<string>;
  readonly #sessionStart: number;
  #tracks = new Map<string, Track>();

  constructor(dir: string, whitelist: readonly string[], sessionStart: number) {
    this.dir = dir;
    this.#whitelist = new Set(whitelist);
    this.#sessionStart = sessionStart;
  }

  segmentStart({ userId, t }: SegmentEdge): void {
    const track = this.#track(userId);
    if (!track || track.open) return;
    track.open = { startMs: t - this.#sessionStart, sampleOffset: track.wav.samples };
  }

  frame({ userId, pcm, t }: Frame): void {
    const track = this.#track(userId);
    if (!track) return;
    if (!track.open) this.segmentStart({ userId, t });
    track.wav.write(pcm);
  }

  segmentEnd({ userId }: SegmentEdge): void {
    const track = this.#tracks.get(userId);
    if (!track?.open) return;
    const samples = track.wav.samples - track.open.sampleOffset;
    const segment: Segment = {
      start_ms: track.open.startMs,
      duration_ms: Math.round((samples * 1000) / SAMPLE_RATE),
      sample_offset: track.open.sampleOffset,
    };
    track.open = null;
    if (samples > 0) appendFileSync(track.segmentsPath, JSON.stringify(segment) + "\n");
    track.wav.flush();
  }

  // Закрыть все файлы: открытые фрагменты завершаются, заголовки исправляются.
  // Можно продолжать запись и после этого: файлы откроются снова и будут дописаны.
  closeAll(): void {
    for (const [userId, track] of this.#tracks) {
      this.segmentEnd({ userId, t: Date.now() });
      track.wav.close();
    }
    this.#tracks.clear();
  }

  #track(userId: string): Track | null {
    if (!this.#whitelist.has(userId)) return null;
    let track = this.#tracks.get(userId);
    if (!track) {
      mkdirSync(this.dir, { recursive: true });
      track = {
        wav: new WavWriter(join(this.dir, `${userId}.wav`)),
        segmentsPath: join(this.dir, `${userId}.segments.jsonl`),
        open: null,
      };
      this.#tracks.set(userId, track);
    }
    return track;
  }
}

// Имя каталога Сессии: локальное время начала, без двоеточий.
export function sessionDirName(date: Date): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return (
    `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())}_` +
    `${p(date.getHours())}-${p(date.getMinutes())}-${p(date.getSeconds())}`
  );
}
