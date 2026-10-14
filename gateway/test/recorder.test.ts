import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { FRAME_SAMPLES, SAMPLE_RATE } from "../src/mixer.ts";
import { Recorder, type Segment } from "../src/recorder.ts";

const A = "111111111111111111";
const GUEST = "999999999999999999";
const T0 = 1_000_000;

const tmp = () => mkdtempSync(join(tmpdir(), "terebot-rec-"));

// Синусоида, чтобы в файле был слышимый сигнал, а не нули.
function tone(frameIndex: number): Int16Array {
  const pcm = new Int16Array(FRAME_SAMPLES);
  for (let i = 0; i < FRAME_SAMPLES; i++) {
    pcm[i] = Math.round(8000 * Math.sin((2 * Math.PI * 440 * (frameIndex * FRAME_SAMPLES + i)) / SAMPLE_RATE));
  }
  return pcm;
}

function speak(rec: Recorder, userId: string, startT: number, frames: number): void {
  rec.segmentStart({ userId, t: startT });
  for (let i = 0; i < frames; i++) rec.frame({ userId, pcm: tone(i), t: startT + i * 20 });
  rec.segmentEnd({ userId, t: startT + frames * 20 });
}

// Разбор WAV стандартным модулем Python: тот же путь, по которому файлы будет читать ядро.
function readWav(path: string): { channels: number; width: number; rate: number; frames: number } {
  const code =
    "import json, sys, wave\n" +
    "w = wave.open(sys.argv[1])\n" +
    "print(json.dumps({'channels': w.getnchannels(), 'width': w.getsampwidth(), " +
    "'rate': w.getframerate(), 'frames': w.getnframes()}))";
  return JSON.parse(execFileSync("python3", ["-c", code, path], { encoding: "utf8" }));
}

function readSegments(dir: string, userId: string): Segment[] {
  return readFileSync(join(dir, `${userId}.segments.jsonl`), "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
}

test("два фрагмента с паузой: WAV открывается, длительность = сумма фрагментов", () => {
  const dir = tmp();
  const rec = new Recorder(dir, [A], T0);
  speak(rec, A, T0 + 1000, 50); // 1 с
  speak(rec, A, T0 + 600_000, 75); // 1,5 с через 10 минут молчания
  rec.closeAll();

  const wav = readWav(join(dir, `${A}.wav`));
  assert.deepEqual({ ...wav, frames: undefined }, { channels: 1, width: 2, rate: 48_000, frames: undefined });
  assert.equal(wav.frames, 125 * FRAME_SAMPLES);
  assert.equal(wav.frames / wav.rate, 2.5);

  const segments = readSegments(dir, A);
  assert.deepEqual(segments, [
    { start_ms: 1000, duration_ms: 1000, sample_offset: 0 },
    { start_ms: 600_000, duration_ms: 1500, sample_offset: 50 * FRAME_SAMPLES },
  ]);
  const total = segments.reduce((s, x) => s + x.duration_ms, 0);
  assert.equal(total, (wav.frames / wav.rate) * 1000);
});

test("файл корректен и посреди фрагмента, если закрыть его (SIGINT)", () => {
  const dir = tmp();
  const rec = new Recorder(dir, [A], T0);
  rec.segmentStart({ userId: A, t: T0 });
  for (let i = 0; i < 30; i++) rec.frame({ userId: A, pcm: tone(i), t: T0 + i * 20 });
  rec.closeAll();
  assert.equal(readWav(join(dir, `${A}.wav`)).frames, 30 * FRAME_SAMPLES);
  assert.deepEqual(readSegments(dir, A), [{ start_ms: 0, duration_ms: 600, sample_offset: 0 }]);
});

test("после закрытия (переподключение) запись продолжается в тот же файл", () => {
  const dir = tmp();
  const rec = new Recorder(dir, [A], T0);
  speak(rec, A, T0, 10);
  rec.closeAll();
  speak(rec, A, T0 + 5000, 20);
  rec.closeAll();
  assert.equal(readWav(join(dir, `${A}.wav`)).frames, 30 * FRAME_SAMPLES);
  assert.deepEqual(readSegments(dir, A).map((s) => s.sample_offset), [0, 10 * FRAME_SAMPLES]);
  assert.deepEqual(readdirSync(dir).sort(), [`${A}.segments.jsonl`, `${A}.wav`]);
});

test("гость не из whitelist нигде не сохраняется", () => {
  const dir = join(tmp(), "session");
  const rec = new Recorder(dir, [A], T0);
  speak(rec, GUEST, T0, 50);
  rec.closeAll();
  assert.equal(existsSync(dir), false);
});
