import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { ProtocolError, decodeAudio, decodeMessage, encodeAudio, encodeMessage } from "../src/core-protocol.ts";

// Общие с pytest (tests/core/test_protocol.py) байтовые примеры.
const FIXTURES = fileURLToPath(new URL("../../tests/fixtures/core-protocol/", import.meta.url));
const CASES = JSON.parse(readFileSync(join(FIXTURES, "cases.json"), "utf8"));

for (const c of CASES) {
  test(`пример ${c.file}: декодируется в те же значения и кодируется байт в байт`, () => {
    const data = readFileSync(join(FIXTURES, c.file));
    if (c.audio) {
      const frame = decodeAudio(data);
      const { kind, t, pcm } = c.audio;
      assert.equal(frame.kind, kind);
      assert.equal(frame.t, t);
      assert.deepEqual(Array.from(frame.pcm), pcm);
      if (frame.kind === "participant") assert.equal(frame.userId, c.audio.user_id);
      else {
        assert.equal(frame.responseId, c.audio.response_id);
        assert.equal(frame.phrase, c.audio.phrase);
      }
      assert.deepEqual(encodeAudio(frame), data);
    } else {
      const text = data.toString("utf8");
      assert.deepEqual(decodeMessage(text), c.message);
      assert.deepEqual(Buffer.from(encodeMessage(decodeMessage(text))), data);
    }
  });
}

test("поля кодируются в порядке схемы", () => {
  assert.equal(
    encodeMessage({ t: 5, user_id: "1", type: "segment_end" } as never),
    '{"type":"segment_end","user_id":"1","t":5}',
  );
});

test("ошибки сообщений", () => {
  assert.throws(() => decodeMessage('{"type":"nope"}'), ProtocolError);
  assert.throws(() => decodeMessage('{"type":"segment_start","t":1}'), /нет полей user_id/);
  assert.throws(() => decodeAudio(Buffer.from([1])), /короче заголовка/);
});
