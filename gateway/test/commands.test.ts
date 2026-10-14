import assert from "node:assert/strict";
import { mock, test } from "node:test";

import { AutoLeave, decideCommand } from "../src/commands.ts";

const WL = ["111111111111111111", "222222222222222222"];
const GUEST = "999999999999999999";
const CHANNEL = "555555555555555555";

test("участник из whitelist в канале: /join подключает к его каналу", () => {
  assert.deepEqual(
    decideCommand({ command: "join", userId: WL[1]!, voiceChannelId: CHANNEL }, WL),
    { kind: "join", channelId: CHANNEL },
  );
});

test("участник из whitelist не в канале: /join просит зайти в канал", () => {
  assert.deepEqual(
    decideCommand({ command: "join", userId: WL[0]!, voiceChannelId: null }, WL),
    { kind: "not-in-voice" },
  );
});

test("участник не из whitelist: отказ на /join и /leave", () => {
  for (const command of ["join", "leave"]) {
    assert.deepEqual(
      decideCommand({ command, userId: GUEST, voiceChannelId: CHANNEL }, WL),
      { kind: "deny" },
    );
  }
});

test("/leave из whitelist работает и вне голосового канала", () => {
  assert.deepEqual(
    decideCommand({ command: "leave", userId: WL[0]!, voiceChannelId: null }, WL),
    { kind: "leave" },
  );
});

function withClock(fn: () => void) {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    fn();
  } finally {
    mock.timers.reset();
  }
}

test("автовыход: все вышли — выход через autoleave_s", () =>
  withClock(() => {
    let left = 0;
    const al = new AutoLeave(30_000, () => left++);
    al.update(0);
    mock.timers.tick(29_999);
    assert.equal(left, 0);
    mock.timers.tick(1);
    assert.equal(left, 1);
    assert.equal(al.pending, false);
  }));

test("автовыход: кто-то вернулся до истечения — Клон остаётся", () =>
  withClock(() => {
    let left = 0;
    const al = new AutoLeave(30_000, () => left++);
    al.update(0);
    mock.timers.tick(20_000);
    al.update(1);
    mock.timers.tick(60_000);
    assert.equal(left, 0);
  }));

test("автовыход: повторные события «пусто» не сдвигают таймер", () =>
  withClock(() => {
    let left = 0;
    const al = new AutoLeave(30_000, () => left++);
    al.update(0);
    mock.timers.tick(20_000);
    al.update(0);
    mock.timers.tick(10_000);
    assert.equal(left, 1);
  }));

test("автовыход: в канале есть люди — таймер не запускается", () =>
  withClock(() => {
    let left = 0;
    const al = new AutoLeave(30_000, () => left++);
    al.update(2);
    mock.timers.tick(60_000);
    assert.equal(left, 0);
    assert.equal(al.pending, false);
  }));
