// Протокол шлюз ↔ ядро (ADR 0002): JSON-сообщения и бинарные кадры звука.
// Та же схема — в src/terebot/core/protocol.py; обе стороны сверяются с байтовыми примерами
// tests/fixtures/core-protocol/. Время — Date.now() шлюза в мс.
//
// Бинарный кадр: заголовок 16 байт — тип (uint8: 1 — звук участника, 2 — звук ответа),
// 7 байт резерва (нули), t в мс (float64 LE). Дальше у звука участника user ID (uint64 LE),
// у звука ответа response_id и номер фразы (uint32 LE каждый). Затем PCM s16le моно 48 кГц.

export const PARTICIPANT_AUDIO = 1;
export const RESPONSE_AUDIO = 2;
const HEADER_BYTES = 16;

// Поля JSON-сообщений в порядке кодирования: так обе стороны пишут одинаковые байты.
export const MESSAGE_FIELDS = {
  // шлюз → ядро
  hello: ["session", "participants", "t"],
  session_start: ["session", "t"],
  session_end: ["t"],
  participant_joined: ["user_id", "name", "t"],
  participant_left: ["user_id", "t"],
  segment_start: ["user_id", "t"],
  segment_end: ["user_id", "t"],
  playback_started: ["response_id", "phrase", "t"],
  phrase_ended: ["response_id", "phrase", "t"],
  playback_finished: ["response_id", "t", "interrupted"],
  // ядро → шлюз
  phrase_begin: ["response_id", "phrase"],
  phrase_end_marker: ["response_id", "phrase"],
  response_end: ["response_id"],
  stop_after_phrase: ["response_id"],
  stop_now: [],
} as const satisfies Record<string, readonly string[]>;

export type MessageType = keyof typeof MESSAGE_FIELDS;
export type Message = { type: MessageType } & Record<string, unknown>;

export class ProtocolError extends Error {
  override name = "ProtocolError";
}

export interface ParticipantAudio {
  kind: "participant";
  userId: string;
  t: number;
  pcm: Int16Array;
}

export interface ResponseAudio {
  kind: "response";
  responseId: number;
  phrase: number;
  t: number;
  pcm: Int16Array;
}

export function encodeAudio(frame: ParticipantAudio | ResponseAudio): Buffer {
  // После заголовка 8 байт: user ID или response_id + номер фразы.
  const head = Buffer.alloc(HEADER_BYTES + 8);
  head.writeUInt8(frame.kind === "participant" ? PARTICIPANT_AUDIO : RESPONSE_AUDIO, 0);
  head.writeDoubleLE(frame.t, 8);
  if (frame.kind === "participant") {
    head.writeBigUInt64LE(BigInt(frame.userId), HEADER_BYTES);
  } else {
    head.writeUInt32LE(frame.responseId, HEADER_BYTES);
    head.writeUInt32LE(frame.phrase, HEADER_BYTES + 4);
  }
  const pcm = Buffer.from(frame.pcm.buffer, frame.pcm.byteOffset, frame.pcm.byteLength);
  return Buffer.concat([head, pcm]);
}

export function decodeAudio(data: Buffer): ParticipantAudio | ResponseAudio {
  if (data.length < HEADER_BYTES + 8) {
    throw new ProtocolError(`кадр звука короче заголовка: ${data.length} байт`);
  }
  const kind = data.readUInt8(0);
  const t = data.readDoubleLE(8);
  const body = data.subarray(HEADER_BYTES + 8);
  if (body.length % 2) throw new ProtocolError("PCM s16 нечётной длины");
  // Копия: Int16Array требует выравнивания по 2 байта, а смещение в буфере может быть нечётным.
  const pcm = new Int16Array(body.length / 2);
  for (let i = 0; i < pcm.length; i++) pcm[i] = body.readInt16LE(i * 2);
  if (kind === PARTICIPANT_AUDIO) {
    return { kind: "participant", userId: data.readBigUInt64LE(HEADER_BYTES).toString(), t, pcm };
  }
  if (kind === RESPONSE_AUDIO) {
    return {
      kind: "response",
      responseId: data.readUInt32LE(HEADER_BYTES),
      phrase: data.readUInt32LE(HEADER_BYTES + 4),
      t,
      pcm,
    };
  }
  throw new ProtocolError(`неизвестный тип кадра звука: ${kind}`);
}

function fieldsOf(message: Record<string, unknown>): readonly string[] {
  const type = message.type;
  if (typeof type !== "string" || !(type in MESSAGE_FIELDS)) {
    throw new ProtocolError(`неизвестный тип сообщения: ${String(type)}`);
  }
  const fields = MESSAGE_FIELDS[type as MessageType];
  const missing = fields.filter((f) => !(f in message));
  if (missing.length) throw new ProtocolError(`${type}: нет полей ${missing.join(", ")}`);
  return fields;
}

export function encodeMessage(message: Message): string {
  const fields = fieldsOf(message);
  const ordered: Record<string, unknown> = { type: message.type };
  for (const f of fields) ordered[f] = message[f];
  return JSON.stringify(ordered);
}

export function decodeMessage(text: string): Message {
  let message: unknown;
  try {
    message = JSON.parse(text);
  } catch (e) {
    throw new ProtocolError(`некорректный JSON: ${(e as Error).message}`);
  }
  if (typeof message !== "object" || message === null || Array.isArray(message)) {
    throw new ProtocolError("сообщение должно быть объектом");
  }
  fieldsOf(message as Record<string, unknown>);
  return message as Message;
}
