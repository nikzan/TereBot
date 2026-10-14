import assert from "node:assert/strict";
import { test } from "node:test";

import { Watchdog } from "../src/watchdog.ts";

const A = "111111111111111111";
const B = "222222222222222222";
const options = { silenceMs: 3000, maxResubscribes: 3 };

test("ошибка расшифровки у говорящего → немедленная переподписка", () => {
  const wd = new Watchdog(options);
  wd.speakingStart(A, 0);
  wd.frame(A, 20);
  const actions = wd.streamError(A, 40);
  assert.equal(actions.length, 1);
  assert.equal(actions[0]!.kind, "resubscribe");
  assert.equal(actions[0]!.userId, A);
});

test("говорит, но кадров нет дольше порога → переподписка", () => {
  const wd = new Watchdog(options);
  wd.speakingStart(A, 0);
  assert.deepEqual(wd.tick(2999), []);
  const actions = wd.tick(3000);
  assert.deepEqual(actions.map((a) => [a.kind, a.userId]), [["resubscribe", A]]);
  // После переподписки отсчёт начинается заново.
  assert.deepEqual(wd.tick(4000), []);
});

test("3 неудачные переподписки за 60 с → переподключение", () => {
  const wd = new Watchdog(options);
  wd.speakingStart(A, 0);
  assert.equal(wd.streamError(A, 100)[0]!.kind, "resubscribe");
  assert.equal(wd.tick(3100)[0]!.kind, "resubscribe");
  assert.equal(wd.tick(6100)[0]!.kind, "resubscribe");
  const actions = wd.tick(9100);
  assert.equal(actions.length, 1);
  assert.equal(actions[0]!.kind, "reconnect");
  // Состояние сброшено: после переподключения счёт неудач начинается с нуля.
  wd.speakingStart(A, 10_000);
  assert.equal(wd.tick(13_000)[0]!.kind, "resubscribe");
});

test("кадр после переподписки — она удачная, счётчик неудач обнуляется", () => {
  const wd = new Watchdog(options);
  wd.speakingStart(A, 0);
  wd.streamError(A, 100);
  wd.tick(3100);
  wd.frame(A, 3200);
  wd.streamError(A, 4000);
  wd.tick(7000);
  // Неудач подряд только две — переподключения нет.
  assert.equal(wd.tick(10_000)[0]!.kind, "resubscribe");
});

test("неудачи, разнесённые больше чем на 60 с, не ведут к переподключению", () => {
  const wd = new Watchdog(options);
  wd.speakingStart(A, 0);
  wd.streamError(A, 0);
  wd.streamError(A, 30_000);
  wd.streamError(A, 61_000);
  assert.equal(wd.streamError(A, 62_000)[0]!.kind, "resubscribe");
});

test("участник молчит → ничего", () => {
  const wd = new Watchdog(options);
  wd.speakingStart(A, 0);
  wd.frame(A, 20);
  wd.speakingEnd(A, 1000);
  assert.deepEqual(wd.tick(10_000), []);
  assert.deepEqual(wd.tick(120_000), []);
  // Ошибка потока у молчащего (поток просто закрылся) тоже не повод переподписываться.
  assert.deepEqual(wd.streamError(A, 120_100), []);
});

test("приём идёт — переподписок нет", () => {
  const wd = new Watchdog(options);
  wd.speakingStart(A, 0);
  for (let t = 20; t <= 10_000; t += 20) {
    wd.frame(A, t);
    assert.deepEqual(wd.tick(t), []);
  }
});

test("участники независимы: пропал приём у одного — переподписка только его", () => {
  const wd = new Watchdog(options);
  wd.speakingStart(A, 0);
  wd.speakingStart(B, 0);
  for (let t = 20; t <= 3000; t += 20) wd.frame(B, t);
  assert.deepEqual(wd.tick(3000).map((a) => a.userId), [A]);
});
