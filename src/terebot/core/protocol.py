"""Протокол шлюз ↔ ядро (ADR 0002): JSON-сообщения и бинарные кадры звука.

Та же схема — в gateway/src/core-protocol.ts; обе стороны сверяются с байтовыми примерами
tests/fixtures/core-protocol/. Время — `Date.now()` шлюза в мс.

Бинарный кадр: заголовок 16 байт — тип (uint8: 1 — звук участника, 2 — звук ответа),
7 байт резерва (нули), t в мс (float64 LE). Дальше у звука участника user ID (uint64 LE),
у звука ответа response_id и номер фразы (uint32 LE каждый). Затем PCM s16le моно 48 кГц.
"""

import json
import struct
from dataclasses import dataclass

PARTICIPANT_AUDIO = 1
RESPONSE_AUDIO = 2

_HEADER = struct.Struct("<B7xd")
_USER = struct.Struct("<Q")
_RESPONSE = struct.Struct("<II")

# Поля JSON-сообщений в порядке кодирования: так обе стороны пишут одинаковые байты.
MESSAGE_FIELDS: dict[str, tuple[str, ...]] = {
    # шлюз → ядро
    "hello": ("session", "participants", "t"),
    "session_start": ("session", "t"),
    "session_end": ("t",),
    "participant_joined": ("user_id", "name", "t"),
    "participant_left": ("user_id", "t"),
    "segment_start": ("user_id", "t"),
    "segment_end": ("user_id", "t"),
    "playback_started": ("response_id", "phrase", "t"),
    "phrase_ended": ("response_id", "phrase", "t"),
    "playback_finished": ("response_id", "t", "interrupted"),
    # ядро → шлюз
    "phrase_begin": ("response_id", "phrase"),
    "phrase_end_marker": ("response_id", "phrase"),
    "response_end": ("response_id",),
    "stop_after_phrase": ("response_id",),
    "stop_now": (),
}


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ParticipantAudio:
    user_id: str
    t: float
    pcm: bytes


@dataclass(frozen=True)
class ResponseAudio:
    response_id: int
    phrase: int
    t: float
    pcm: bytes


def encode_audio(frame: ParticipantAudio | ResponseAudio) -> bytes:
    if isinstance(frame, ParticipantAudio):
        head = _HEADER.pack(PARTICIPANT_AUDIO, frame.t) + _USER.pack(int(frame.user_id))
    else:
        head = _HEADER.pack(RESPONSE_AUDIO, frame.t) + _RESPONSE.pack(
            frame.response_id, frame.phrase
        )
    return head + frame.pcm


def decode_audio(data: bytes) -> ParticipantAudio | ResponseAudio:
    # После заголовка всегда 8 байт: user ID или response_id + номер фразы.
    if len(data) < _HEADER.size + 8:
        raise ProtocolError(f"кадр звука короче заголовка: {len(data)} байт")
    kind, t = _HEADER.unpack_from(data)
    offset = _HEADER.size
    if kind == PARTICIPANT_AUDIO:
        (user_id,) = _USER.unpack_from(data, offset)
        pcm = data[offset + _USER.size :]
        result: ParticipantAudio | ResponseAudio = ParticipantAudio(str(user_id), t, pcm)
    elif kind == RESPONSE_AUDIO:
        response_id, phrase = _RESPONSE.unpack_from(data, offset)
        pcm = data[offset + _RESPONSE.size :]
        result = ResponseAudio(response_id, phrase, t, pcm)
    else:
        raise ProtocolError(f"неизвестный тип кадра звука: {kind}")
    if len(pcm) % 2:
        raise ProtocolError("PCM s16 нечётной длины")
    return result


def encode_message(message: dict) -> str:
    kind = message.get("type")
    fields = MESSAGE_FIELDS.get(kind)
    if fields is None:
        raise ProtocolError(f"неизвестный тип сообщения: {kind}")
    missing = [f for f in fields if f not in message]
    if missing:
        raise ProtocolError(f"{kind}: нет полей {', '.join(missing)}")
    ordered = {"type": kind, **{f: message[f] for f in fields}}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def decode_message(text: str) -> dict:
    try:
        message = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"некорректный JSON: {e}") from None
    if not isinstance(message, dict):
        raise ProtocolError("сообщение должно быть объектом")
    kind = message.get("type")
    fields = MESSAGE_FIELDS.get(kind)
    if fields is None:
        raise ProtocolError(f"неизвестный тип сообщения: {kind}")
    missing = [f for f in fields if f not in message]
    if missing:
        raise ProtocolError(f"{kind}: нет полей {', '.join(missing)}")
    return message
