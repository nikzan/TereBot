"""Ответ Клона: LLM → фразы → текст для Голоса → синтез → звук фраз в шлюз.

Один ответ за раз (кроме приветствия, которое Политика не видит). Решения Политики
об ответе исполняются здесь: отменить подготовку, замолчать, договорить фразу. Отчёты шлюза
о воспроизведении превращаются в события для Политики и в историю Сессии: в неё ответ
попадает только прозвучавшей частью.
"""

import asyncio
import logging
import random
import threading
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from terebot.config import Persona, VoiceSettings
from terebot.core.audio import Upsampler24to48, float_to_pcm
from terebot.core.llm import Llm, LlmError
from terebot.core.prompt import HistoryItem, SaidByClone, SaidByParticipant, build_messages
from terebot.core.protocol import ResponseAudio
from terebot.core.text import PhraseSplitter, for_voice
from terebot.dialogue import (
    CloneFinished,
    ClonePhraseEnded,
    CloneStarted,
    Event,
    ResponseFailed,
    Utterance,
)
from terebot.voice import MlxVoice

log = logging.getLogger(__name__)

OUT_RATE = 48_000
# Служебные токены на сообщение в шаблоне чата: роль и разделители.
MESSAGE_OVERHEAD_TOKENS = 5


class VoiceWorker:
    """Голос в одном выделенном потоке: MLX держит состояние на поток, синтез не параллелится."""

    def __init__(self, settings: VoiceSettings):
        self._settings = settings
        self._voice = MlxVoice(settings.streaming_interval, settings.cache_limit_gb)
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="voice")

    @property
    def sample_rate(self) -> int:
        return self._voice.sample_rate

    def memory(self) -> int | None:
        m = self._voice.backend_memory()
        return m[0] if m else None

    async def load(self) -> None:
        """Загрузка и прогрев до приёма первой Реплики."""
        s = self._settings

        def work() -> None:
            self._voice.load(s.model)
            if s.speaker:
                self._voice.use_speaker(s.speaker)
            else:
                ref_text = s.ref_text or Path(s.ref_audio).with_suffix(".txt").read_text(
                    encoding="utf-8"
                )
                self._voice.prepare_voice(s.ref_audio, ref_text.strip())

        await asyncio.get_running_loop().run_in_executor(self._executor, work)

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        """Звук фразы кусками по мере синтеза. Выход из цикла останавливает синтез."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def work() -> None:
            try:
                for chunk in self._voice.synthesize(text):
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:  # noqa: BLE001 — передаётся в asyncio и там поднимается
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(self._executor, work)
        try:
            while (item := await queue.get()) is not None:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()


@dataclass
class Phrase:
    n: int
    llm_text: str
    voice_text: str
    # Отправлено звука, сэмплов 48 кГц.
    samples: int = 0


@dataclass
class Marks:
    """Метки задержки ответа, секунды на шкале Сессии."""

    t_end: float | None = None
    turn_decided: float | None = None
    text_ready: float | None = None
    llm_request: float | None = None
    first_token: float | None = None
    first_phrase: float | None = None
    first_voice: float | None = None
    playback_started: float | None = None


@dataclass
class Response:
    id: int
    greeting: bool
    question: tuple[SaidByParticipant, ...] = ()
    marks: Marks = field(default_factory=Marks)
    llm_text: str = ""
    phrases: list[Phrase] = field(default_factory=list)
    # Шлюз уже знает об ответе (получил phrase_begin) и держит его в очереди фраз.
    announced: bool = False
    sent_audio: bool = False
    started: bool = False
    # Фраз дозвучало; время конца последней (или начала ответа).
    ended: int = 0
    last_edge: float = 0.0
    # Договорить фразу с этим номером и закончить.
    stop_after: int | None = None
    cancelled: bool = False
    error: str | None = None
    # Индекс записи ответа в истории Сессии, пока он звучит.
    history_index: int | None = None

    def spoken(self, t_finish: float, interrupted: bool) -> str:
        """Прозвучавшая часть: дозвучавшие фразы и начало оборванной — пропорционально времени."""
        words = [p.llm_text for p in self.phrases[: self.ended]]
        if interrupted and self.ended < len(self.phrases):
            p = self.phrases[self.ended]
            duration = p.samples / OUT_RATE
            if duration > 0:
                share = min(1.0, max(0.0, (t_finish - self.last_edge) / duration))
                head = p.llm_text.split()
                words.append(" ".join(head[: round(share * len(head))]))
        return " ".join(w for w in words if w)


class Responder:
    def __init__(
        self,
        llm: Llm,
        voice: VoiceWorker,
        persona: Persona,
        settings: VoiceSettings,
        context_tokens: int,
        send: Callable[[dict | bytes], None],
        clock: Callable[[], float],
        on_event: Callable[[Event], None],
        journal,
    ):
        self._llm = llm
        self._voice = voice
        self._persona = persona
        self._settings = settings
        self._budget = context_tokens
        self._send = send
        self._clock = clock
        self._on_event = on_event
        self._journal = journal
        self.history: list[HistoryItem] = []
        self._responses: dict[int, Response] = {}
        self._current: Response | None = None
        self._task: asyncio.Task | None = None
        self._next_id = 1
        self._line_tokens: dict[str, int] = {}

    # --- Решения Политики ---

    def add_context(self, item: SaidByParticipant) -> None:
        self.history.append(item)

    def respond(
        self, question: tuple[SaidByParticipant, ...], utterance: Utterance, marks: Marks
    ) -> Response:
        resp = self._new(greeting=False)
        resp.question = question
        resp.marks = marks
        self._current = resp
        self._task = self._spawn(self._run_answer(resp))
        return resp

    def greet(self) -> Response:
        resp = self._new(greeting=True)
        self._spawn(self._run_greeting(resp, random.choice(self._settings.greetings)))
        return resp

    def cancel(self) -> None:
        """Бросить подготовку: ничего из ответа не звучит."""
        resp = self._take_current()
        if resp and resp.announced:
            # Ответ уже в очереди шлюза, хотя бы как начатая фраза без звука: пусть выбросит,
            # иначе он навсегда встанет в голове очереди и следующие ответы не зазвучат.
            self._send({"type": "stop_after_phrase", "response_id": resp.id})

    def interrupt(self) -> None:
        """Замолчать немедленно."""
        resp = self._take_current(keep_events=True)
        if resp or self._responses:
            self._send({"type": "stop_now"})
        if resp:
            log.info("ответ %s оборван", resp.id)

    def stop_after_phrase(self) -> None:
        """Договорить звучащую фразу, остальное не произносить."""
        resp = self._current
        if resp is None:
            return
        self._send({"type": "stop_after_phrase", "response_id": resp.id})
        if not resp.started:
            self._take_current()
            return
        # Генерация новых фраз прекращается, но звучащую фразу досинтезируем до конца.
        resp.stop_after = resp.ended

    # --- Отчёты шлюза ---

    def on_playback(self, message: dict, t: float) -> None:
        resp = self._responses.get(message["response_id"])
        if resp is None:
            return
        kind = message["type"]
        if kind == "playback_started":
            resp.started = True
            resp.last_edge = t
            resp.marks.playback_started = t
            if not resp.greeting and not resp.cancelled:
                # Вопрос и (пока пустой) ответ — в историю, ответ дополнится по его концу.
                self.history.extend(resp.question)
                resp.history_index = len(self.history)
                self.history.append(SaidByClone(""))
                self._journal.latency(resp)
                self._on_event(CloneStarted(t))
        elif kind == "phrase_ended":
            resp.ended += 1
            resp.last_edge = t
            if not resp.greeting and not resp.cancelled:
                self._on_event(ClonePhraseEnded(t))
        elif kind == "playback_finished":
            interrupted = bool(message["interrupted"])
            spoken = resp.spoken(t, interrupted)
            if resp.history_index is not None:
                self.history[resp.history_index] = SaidByClone(spoken, interrupted)
            self._journal.response(resp, spoken, interrupted)
            del self._responses[resp.id]
            if self._current is resp:
                self._current = None
            if not resp.greeting and not resp.cancelled and resp.started:
                self._on_event(CloneFinished(t))

    # --- Подготовка ответа ---

    def _new(self, greeting: bool) -> Response:
        resp = Response(self._next_id, greeting)
        self._next_id += 1
        self._responses[resp.id] = resp
        return resp

    def _take_current(self, keep_events: bool = False) -> Response | None:
        resp, self._current = self._current, None
        if self._task:
            self._task.cancel()
            self._task = None
        if resp:
            # После «замолчать» отчёты шлюза об ответе ещё нужны: по ним — прозвучавшая часть.
            resp.cancelled = not keep_events or not resp.started
            if not resp.announced:
                self._responses.pop(resp.id, None)
        return resp

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coro)
        task.add_done_callback(_log_crash)
        return task

    async def _messages(self, question: tuple[SaidByParticipant, ...]) -> list[dict]:
        # Подсчёт по строкам с кэшем: каждая новая строка — один запрос /tokenize.
        full, _ = build_messages(self._persona, self.history, question, 10**9, lambda m: 0)
        for m in full:
            for line in m["content"].split("\n"):
                if line not in self._line_tokens:
                    self._line_tokens[line] = await self._llm.tokenize(line)

        def count(messages: list[dict]) -> int:
            total = 0
            for m in messages:
                lines = m["content"].split("\n")
                total += MESSAGE_OVERHEAD_TOKENS + sum(self._line_tokens[x] + 1 for x in lines)
            return total

        messages, dropped = build_messages(
            self._persona, self.history, question, self._budget, count
        )
        if dropped:
            log.info("из промпта отброшено старых записей: %d", dropped)
        return messages

    async def _run_answer(self, resp: Response) -> None:
        phrases: asyncio.Queue[Phrase | None] = asyncio.Queue()
        synth = asyncio.create_task(self._synthesize(resp, phrases))
        try:
            messages = await self._messages(resp.question)
            s = self._settings
            splitter = PhraseSplitter(s.min_phrase_chars, s.max_phrase_chars)
            resp.marks.llm_request = self._clock()
            async for chunk in self._llm.stream(messages):
                if resp.marks.first_token is None:
                    resp.marks.first_token = self._clock()
                resp.llm_text += chunk
                for text in splitter.feed(chunk):
                    self._enqueue(resp, text, phrases)
                if resp.stop_after is not None:
                    break
            if resp.stop_after is None:
                for text in splitter.flush():
                    self._enqueue(resp, text, phrases)
            log.info("ответ %s: LLM закончила, фраз %d", resp.id, len(resp.phrases))
            phrases.put_nowait(None)
            await synth
        except asyncio.CancelledError:
            synth.cancel()
            raise
        except LlmError as e:
            synth.cancel()
            self._fail(resp, f"LLM: {e}")
        except Exception as e:  # noqa: BLE001 — сбой Голоса не должен ронять ядро
            synth.cancel()
            log.exception("сбой ответа %s", resp.id)
            self._fail(resp, f"{type(e).__name__}: {e}")

    async def _run_greeting(self, resp: Response, text: str) -> None:
        phrases: asyncio.Queue[Phrase | None] = asyncio.Queue()
        resp.llm_text = text
        self._enqueue(resp, text, phrases)
        phrases.put_nowait(None)
        try:
            await self._synthesize(resp, phrases)
        except Exception as e:  # noqa: BLE001
            log.exception("сбой приветствия")
            self._journal.error("greeting", f"{type(e).__name__}: {e}")
            self._responses.pop(resp.id, None)

    def _enqueue(self, resp: Response, text: str, phrases: asyncio.Queue) -> None:
        voice_text = for_voice(text, self._persona.lexicon)
        if not voice_text:
            return
        if resp.marks.first_phrase is None:
            resp.marks.first_phrase = self._clock()
        phrase = Phrase(len(resp.phrases), text, voice_text)
        resp.phrases.append(phrase)
        phrases.put_nowait(phrase)

    async def _synthesize(self, resp: Response, phrases: asyncio.Queue) -> None:
        """Фразы синтезируются по очереди: следующая — сразу после синтеза предыдущей."""
        up = Upsampler24to48()
        sent = 0
        while (phrase := await phrases.get()) is not None:
            if resp.stop_after is not None and phrase.n > resp.stop_after:
                break
            self._send({"type": "phrase_begin", "response_id": resp.id, "phrase": phrase.n})
            resp.announced = True
            try:
                async for chunk in self._voice.synthesize(phrase.voice_text):
                    if resp.stop_after is not None and phrase.n > resp.stop_after:
                        break
                    if resp.marks.first_voice is None:
                        resp.marks.first_voice = self._clock()
                    audio = up(chunk) if self._voice.sample_rate != OUT_RATE else chunk
                    phrase.samples += len(audio)
                    resp.sent_audio = True
                    self._send(
                        ResponseAudio(resp.id, phrase.n, self._clock() * 1000, float_to_pcm(audio))
                    )
            finally:
                # Даже при сбое шлюз должен знать, что звука этой фразы больше не будет.
                self._send(
                    {"type": "phrase_end_marker", "response_id": resp.id, "phrase": phrase.n}
                )
            sent += 1
        if sent == 0:
            raise LlmError("в ответе нет текста для Голоса")
        self._send({"type": "response_end", "response_id": resp.id})

    def _fail(self, resp: Response, error: str) -> None:
        resp.error = error
        self._journal.error("response", error)
        if self._current is resp:
            self._current = None
            self._task = None
        if resp.sent_audio:
            # Часть ответа уже у шлюза: пусть доиграет её и закончит ответ.
            self._send({"type": "response_end", "response_id": resp.id})
            return
        if resp.announced:
            # Звука нет, но фраза объявлена: шлюз должен выбросить ответ из очереди.
            self._send({"type": "stop_after_phrase", "response_id": resp.id})
        self._responses.pop(resp.id, None)
        # Ответ не прозвучал, но Реплики, на которые он готовился, остаются в контексте.
        self.history.extend(resp.question)
        self._on_event(ResponseFailed(self._clock()))


def _log_crash(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception():
        log.error("задача ответа упала", exc_info=task.exception())
