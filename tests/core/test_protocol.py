"""Кодек протокола против общих байтовых примеров (их же проверяет gateway/test)."""

import json
import struct
from pathlib import Path

import pytest

from terebot.core.protocol import (
    ParticipantAudio,
    ProtocolError,
    ResponseAudio,
    decode_audio,
    decode_message,
    encode_audio,
    encode_message,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "core-protocol"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))


def expected_audio(spec: dict) -> ParticipantAudio | ResponseAudio:
    pcm = struct.pack(f"<{len(spec['pcm'])}h", *spec["pcm"])
    if spec["kind"] == "participant":
        return ParticipantAudio(spec["user_id"], spec["t"], pcm)
    return ResponseAudio(spec["response_id"], spec["phrase"], spec["t"], pcm)


@pytest.mark.parametrize("case", [c for c in CASES if "audio" in c], ids=lambda c: c["file"])
def test_audio_fixture(case):
    data = (FIXTURES / case["file"]).read_bytes()
    frame = decode_audio(data)
    assert frame == expected_audio(case["audio"])
    assert encode_audio(frame) == data


@pytest.mark.parametrize("case", [c for c in CASES if "message" in c], ids=lambda c: c["file"])
def test_message_fixture(case):
    text = (FIXTURES / case["file"]).read_text(encoding="utf-8")
    assert decode_message(text) == case["message"]
    assert encode_message(decode_message(text)).encode() == text.encode()


def test_encode_orders_fields():
    a = encode_message({"t": 5, "user_id": "1", "type": "segment_end"})
    assert a == '{"type":"segment_end","user_id":"1","t":5}'


@pytest.mark.parametrize(
    "text, problem",
    [
        ('{"type":"nope"}', "неизвестный тип"),
        ('{"type":"segment_start","t":1}', "нет полей user_id"),
        ("[1]", "объектом"),
        ("{", "некорректный JSON"),
    ],
)
def test_bad_messages(text, problem):
    with pytest.raises(ProtocolError, match=problem):
        decode_message(text)


def test_bad_audio():
    with pytest.raises(ProtocolError, match="короче заголовка"):
        decode_audio(b"\x01")
    with pytest.raises(ProtocolError, match="тип кадра"):
        decode_audio(bytes([9]) + bytes(23))
