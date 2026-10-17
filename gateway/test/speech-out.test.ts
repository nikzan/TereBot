import assert from "node:assert/strict";
import { test } from "node:test";

import { FRAME_MS, FRAME_SAMPLES } from "../src/mixer.ts";
import { SpeechOut } from "../src/speech-out.ts";

const frames = (n: number, value = 1) => new Int16Array(n * FRAME_SAMPLES).fill(value);

// Тянет кадры по 20 мс, пока очередь отдаёт звук или события; возвращает кадры и события.
function drain(out: SpeechOut, start: number, maxTicks = 100) {
  const played: Int16Array[] = [];
  const events: Record<string, unknown>[] = [];
  let now = start;
  for (let i = 0; i < maxTicks; i++, now += FRAME_MS) {
    const pulled = out.pull(now);
    events.push(...pulled.events);
    if (pulled.frame) played.push(pulled.frame);
    else if (pulled.events.length === 0) break;
  }
  return { played, events, now };
}

function fullResponse(out: SpeechOut, id: number, phrases: number[]) {
  phrases.forEach((n, i) => {
    out.phraseBegin(id, i);
    out.audio(id, i, frames(n, i + 1));
    out.phraseEndMarker(id, i);
  });
  out.responseEnd(id);
}

test("ответ из двух фраз: начало, конец каждой фразы, конец ответа", () => {
  const out = new SpeechOut();
  fullResponse(out, 1, [2, 3]);
  const { played, events } = drain(out, 0);
  assert.equal(played.length, 5);
  assert.deepEqual(
    events.map((e) => [e.type, e.phrase ?? null, e.t]),
    [
      ["playback_started", 0, 0],
      ["phrase_ended", 0, 40],
      ["phrase_ended", 1, 100],
      ["playback_finished", null, 100],
    ],
  );
  assert.equal(events.at(-1)!.interrupted, false);
});

test("договорить фразу: первая из трёх звучит до конца, остальные — нет", () => {
  const out = new SpeechOut();
  fullResponse(out, 1, [3, 3, 3]);
  const first = out.pull(0);
  assert.equal(first.events[0]!.type, "playback_started");
  assert.deepEqual(out.stopAfterPhrase(1, 20), []);
  const { played, events } = drain(out, 20);
  assert.equal(played.length, 2);
  assert.ok(played.every((f) => f[0] === 1), "звучит только первая фраза");
  assert.deepEqual(
    events.map((e) => e.type),
    ["phrase_ended", "playback_finished"],
  );
  assert.equal(events[1]!.interrupted, true);
});

test("фраза ещё не пришла целиком: играет по мере поступления, конец — после последнего куска", () => {
  const out = new SpeechOut();
  out.phraseBegin(1, 0);
  out.audio(1, 0, frames(1));
  let r = drain(out, 0);
  assert.equal(r.played.length, 1);
  assert.deepEqual(r.events.map((e) => e.type), ["playback_started"]);
  // Звук кончился, а конца фразы ещё нет: очередь ждёт.
  assert.deepEqual(out.pull(r.now), { frame: null, events: [] });
  // Кусок не кратен кадру: хвост ждёт следующего куска.
  out.audio(1, 0, new Int16Array(FRAME_SAMPLES + 100));
  r = drain(out, r.now);
  assert.equal(r.played.length, 1);
  assert.deepEqual(r.events, []);
  out.phraseEndMarker(1, 0);
  out.responseEnd(1);
  r = drain(out, r.now);
  assert.equal(r.played.length, 1, "хвост добит тишиной до кадра");
  assert.deepEqual(r.events.map((e) => e.type), ["phrase_ended", "playback_finished"]);
});

test("следующая фраза ещё синтезируется: конец ответа не объявляется", () => {
  const out = new SpeechOut();
  out.phraseBegin(1, 0);
  out.audio(1, 0, frames(1));
  out.phraseEndMarker(1, 0);
  const r = drain(out, 0);
  assert.deepEqual(r.events.map((e) => e.type), ["playback_started", "phrase_ended"]);
  out.responseEnd(1);
  assert.deepEqual(out.pull(r.now).events.map((e) => e.type), ["playback_finished"]);
});

test("замолчать сейчас: очередь пуста, запоздавший звук выброшен", () => {
  const out = new SpeechOut();
  fullResponse(out, 1, [5]);
  out.pull(0);
  const events = out.stopNow(20);
  assert.deepEqual(events, [{ type: "playback_finished", response_id: 1, t: 20, interrupted: true }]);
  out.audio(1, 1, frames(2));
  assert.deepEqual(out.pull(40), { frame: null, events: [] });
  assert.equal(out.playing, false);
});

test("договорить фразу до начала ответа: ответ выброшен целиком", () => {
  const out = new SpeechOut();
  fullResponse(out, 1, [2]);
  const events = out.stopAfterPhrase(1, 0);
  assert.equal(events[0]!.type, "playback_finished");
  assert.deepEqual(out.pull(0), { frame: null, events: [] });
});

test("ответы играют по очереди", () => {
  const out = new SpeechOut();
  fullResponse(out, 1, [1]);
  fullResponse(out, 2, [1]);
  const { events } = drain(out, 0);
  assert.deepEqual(
    events.map((e) => `${e.type}:${e.response_id}`),
    [
      "playback_started:1",
      "phrase_ended:1",
      "playback_finished:1",
      "playback_started:2",
      "phrase_ended:2",
      "playback_finished:2",
    ],
  );
});
