import assert from "node:assert/strict";
import { test } from "node:test";

import { FRAME_SAMPLES, Mixer, monoToStereo, stereoToMono } from "../src/mixer.ts";

const A = "111111111111111111";
const B = "222222222222222222";

const frame = (value: number) => new Int16Array(FRAME_SAMPLES).fill(value);

test("один поток: кадр выходит не раньше задержки и без изменений", () => {
  const mixer = new Mixer(2000);
  mixer.push(A, frame(1000), 0);
  assert.equal(mixer.pull(1999), null);
  const out = mixer.pull(2000);
  assert.ok(out);
  assert.equal(out.length, FRAME_SAMPLES);
  assert.ok(out.every((s) => s === 1000));
  assert.equal(mixer.pull(2020), null);
});

test("один поток: за один pull выходит один кадр, порядок сохраняется", () => {
  const mixer = new Mixer(0);
  mixer.push(A, frame(1), 0);
  mixer.push(A, frame(2), 20);
  assert.equal(mixer.pull(100)![0], 1);
  assert.equal(mixer.pull(100)![0], 2);
  assert.equal(mixer.pull(100), null);
});

test("два потока складываются", () => {
  const mixer = new Mixer(2000);
  mixer.push(A, frame(1000), 0);
  mixer.push(B, frame(-300), 10);
  assert.ok(mixer.pull(2010)!.every((s) => s === 700));
});

test("два потока: созревший выходит один, второй ждёт своей задержки", () => {
  const mixer = new Mixer(2000);
  mixer.push(A, frame(1000), 0);
  mixer.push(B, frame(500), 500);
  assert.ok(mixer.pull(2000)!.every((s) => s === 1000));
  assert.equal(mixer.pull(2020), null);
  assert.ok(mixer.pull(2500)!.every((s) => s === 500));
});

test("сумма насыщается до границ int16", () => {
  const mixer = new Mixer(0);
  mixer.push(A, frame(30_000), 0);
  mixer.push(B, frame(10_000), 0);
  assert.ok(mixer.pull(0)!.every((s) => s === 32767));
  mixer.push(A, frame(-30_000), 0);
  mixer.push(B, frame(-10_000), 0);
  assert.ok(mixer.pull(0)!.every((s) => s === -32768));
});

test("нет созревших кадров → нет вывода", () => {
  const mixer = new Mixer(2000);
  assert.equal(mixer.pull(10_000), null);
  mixer.push(A, frame(1), 9_000);
  assert.equal(mixer.pull(10_000), null);
});

test("моно → стерео → моно без потерь", () => {
  const mono = Int16Array.from([0, 1, -1, 32767, -32768, 1234]);
  const stereo = monoToStereo(mono);
  assert.equal(stereo.length, mono.length * 4);
  assert.deepEqual(stereoToMono(stereo), mono);
});
