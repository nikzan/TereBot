"""Исполнение решений Политики об ответе (спека clone-response, «Отмена и остановка по Политике»).

LLM и Голос подменены: проверяется, что ядро говорит шлюзу, а не синтез.
"""

import asyncio

import numpy as np

from terebot.config import Persona, VoiceSettings
from terebot.core.prompt import SaidByParticipant
from terebot.core.protocol import ResponseAudio
from terebot.core.response import Marks, Responder
from terebot.dialogue import ResponseFailed, Utterance

PERSONA = Persona("Паспорт.", (), ())
SETTINGS = VoiceSettings(min_phrase_chars=5)


class FakeLlm:
    async def tokenize(self, text):
        return len(text.split())

    async def stream(self, messages):
        yield "Первая фраза ответа. Вторая фраза ответа."


class FakeVoice:
    """Синтез каждой фразы ждёт разрешения из теста."""

    sample_rate = 48_000

    def __init__(self):
        self.go = asyncio.Event()

    async def synthesize(self, text):
        await self.go.wait()
        yield np.zeros(960, dtype=np.float32)


class Journal:
    def __getattr__(self, name):
        return lambda *a, **k: None


def make():
    sent, events = [], []
    voice = FakeVoice()
    r = Responder(FakeLlm(), voice, PERSONA, SETTINGS, 1000, sent.append, lambda: 1.0,
                  events.append, Journal())  # fmt: skip
    return r, voice, sent, events


def respond(r: Responder):
    u = Utterance("1", "Теребот, привет", 0.0, 0.5)
    return r.respond((SaidByParticipant("Вася", u.text),), u, Marks(t_end=0.5))


def types(sent):
    return [m["type"] if isinstance(m, dict) else "audio" for m in sent]


async def until(predicate):
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("не дождались")


def test_cancel_after_phrase_begin_drops_response_in_gateway():
    async def run():
        r, voice, sent, _ = make()
        resp = respond(r)
        await until(lambda: "phrase_begin" in types(sent))
        assert "audio" not in types(sent)
        r.cancel()
        await asyncio.sleep(0)
        assert {"type": "stop_after_phrase", "response_id": resp.id} in sent

    asyncio.run(run())


def test_cancel_before_announce_sends_nothing():
    async def run():
        r, _, sent, _ = make()
        respond(r)
        r.cancel()
        await asyncio.sleep(0)
        assert sent == []

    asyncio.run(run())


def test_phrases_audio_and_end_in_order():
    async def run():
        r, voice, sent, _ = make()
        resp = respond(r)
        voice.go.set()
        await until(lambda: "response_end" in types(sent))
        assert types(sent) == [
            "phrase_begin", "audio", "phrase_end_marker",
            "phrase_begin", "audio", "phrase_end_marker",
            "response_end",
        ]  # fmt: skip
        assert all(m.response_id == resp.id for m in sent if isinstance(m, ResponseAudio))

    asyncio.run(run())


def test_llm_failure_reports_response_failed():
    class BrokenLlm(FakeLlm):
        async def stream(self, messages):
            from terebot.core.llm import LlmError

            raise LlmError("llama-server не отвечает")
            yield  # noqa: B901 — делает функцию генератором

    async def run():
        r, _, sent, events = make()
        r._llm = BrokenLlm()
        respond(r)
        await until(lambda: events)
        assert isinstance(events[0], ResponseFailed)
        assert sent == []

    asyncio.run(run())
