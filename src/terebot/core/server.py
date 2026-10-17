"""Цикл ядра: WebSocket-сервер для шлюза, восприятие речи, Политика Диалога и её исполнитель.

Один asyncio-цикл. Распознавание и Smart Turn — в пуле из двух потоков, Голос — в своём потоке.
Каждое событие идёт в `DialoguePolicy.handle`, раз в 100 мс — `tick`; решения исполняются
здесь и пишутся в Журнал Сессии. Время шлюза (`Date.now()`, мс) переводится в секунды
от начала Сессии.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from terebot.config import Persona, Settings
from terebot.core.journal import Journal
from terebot.core.listener import Listener, UtteranceMarks
from terebot.core.llm import Llm
from terebot.core.metrics import MemorySampler
from terebot.core.models import Asr, SileroVad, SmartTurn
from terebot.core.prompt import SaidByParticipant
from terebot.core.protocol import (
    ParticipantAudio,
    ProtocolError,
    ResponseAudio,
    decode_audio,
    decode_message,
    encode_audio,
    encode_message,
)
from terebot.core.response import Marks, Responder, VoiceWorker
from terebot.dialogue import (
    AddToContext,
    CancelPendingResponse,
    Decision,
    DialoguePolicy,
    Event,
    InterruptClone,
    ParticipantLeft,
    Respond,
    StopAfterCurrentPhrase,
    Utterance,
)

log = logging.getLogger("terebot.core")

TICK_S = 0.1
MEMORY_S = 1.0
PLAYBACK = ("playback_started", "phrase_ended", "playback_finished")


class Core:
    def __init__(self, settings: Settings, persona: Persona):
        self._settings = settings
        self._whitelist = frozenset(settings.discord.whitelist)
        self._executor = ThreadPoolExecutor(2, thread_name_prefix="speech")
        self._journal = Journal(settings.journal)
        self._llm = Llm(settings.llm)
        self._voice = VoiceWorker(settings.voice)
        self._memory = MemorySampler(self._voice.memory, lambda: self._llm.pid)
        self._responder = Responder(
            self._llm,
            self._voice,
            persona,
            settings.voice,
            settings.llm.context_tokens,
            self._send,
            self.clock,
            self._handle,
            self._journal,
        )
        self._policy = self._new_policy()
        # Сессия: имя каталога и начало по часам шлюза, мс.
        self._session: tuple[str, float] | None = None
        self._names: dict[str, str] = {}
        self._listeners: dict[str, Listener] = {}
        self._marks: dict[Utterance, UtteranceMarks] = {}
        self._outbox: asyncio.Queue[str | bytes] | None = None
        self._started = time.time()

    def _new_policy(self) -> DialoguePolicy:
        return DialoguePolicy(self._settings.dialogue, self._settings.discord.whitelist)

    # --- Время ---

    def clock(self) -> float:
        """Сейчас, секунды на шкале Сессии."""
        start = self._session[1] / 1000 if self._session else self._started
        return time.time() - start

    def _to_session(self, t_ms: float) -> float:
        return (t_ms - self._session[1]) / 1000 if self._session else self.clock()

    # --- Запуск ---

    async def load(self) -> None:
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        log.info("загрузка моделей восприятия речи…")
        self._vad, self._turn, self._asr = await asyncio.gather(
            loop.run_in_executor(self._executor, SileroVad),
            loop.run_in_executor(self._executor, SmartTurn),
            loop.run_in_executor(self._executor, Asr),
        )
        log.info("LLM…")
        await self._llm.start()
        log.info("Голос: загрузка и прогрев…")
        await self._voice.load()
        log.info("модели готовы за %.1f с", time.monotonic() - t0)

    async def serve(self) -> None:
        port = self._settings.core.port
        async with serve(self._connection, "127.0.0.1", port, max_size=None):
            log.info("Ядро готово: ждёт шлюз на ws://127.0.0.1:%d", port)
            ticker = asyncio.create_task(self._tick_loop())
            sampler = asyncio.create_task(self._memory_loop())
            try:
                await asyncio.Future()
            finally:
                ticker.cancel()
                sampler.cancel()
                await self.close()

    async def close(self) -> None:
        await self._end_session("остановка ядра")
        await self._llm.close()

    # --- Связь со шлюзом ---

    async def _connection(self, ws: ServerConnection) -> None:
        log.info("шлюз подключился")
        outbox: asyncio.Queue[str | bytes] = asyncio.Queue()
        self._outbox = outbox

        async def writer() -> None:
            while True:
                await ws.send(await outbox.get())

        writing = asyncio.create_task(writer())
        try:
            async for data in ws:
                try:
                    if isinstance(data, bytes):
                        self._on_audio(decode_audio(data))
                    else:
                        await self._on_message(decode_message(data))
                except ProtocolError as e:
                    log.warning("ошибка протокола: %s", e)
        except ConnectionClosed:
            pass
        finally:
            writing.cancel()
            if self._outbox is outbox:
                self._outbox = None
                # Шлюз пропал: звук ответа больше не звучит, ответ бросаем.
                self._responder.interrupt()
            log.info("шлюз отключился")

    def _send(self, message: dict | ResponseAudio) -> None:
        if self._outbox is None:
            return
        if isinstance(message, ResponseAudio):
            self._outbox.put_nowait(encode_audio(message))
        else:
            self._outbox.put_nowait(encode_message(message))

    async def _on_message(self, m: dict) -> None:
        match m["type"]:
            case "hello":
                self._names |= {p["user_id"]: p["name"] for p in m["participants"]}
                session = m["session"]
                if session is None:
                    await self._end_session("у шлюза нет Сессии")
                elif self._session is None or self._session[0] != session["name"]:
                    # Ядро перезапустили посреди Сессии: продолжаем её без приветствия.
                    await self._start_session(session["name"], session["t"], greet=False)
            case "session_start":
                await self._start_session(m["session"]["name"], m["session"]["t"], greet=True)
            case "session_end":
                await self._end_session("конец Сессии в шлюзе")
            case "participant_joined":
                self._names[m["user_id"]] = m["name"]
            case "participant_left":
                listener = self._listeners.pop(m["user_id"], None)
                if listener:
                    await listener.close()
                self._handle(ParticipantLeft(m["user_id"], self._to_session(m["t"])))
            case kind if kind in PLAYBACK:
                self._responder.on_playback(m, self._to_session(m["t"]))

    def _on_audio(self, frame: ParticipantAudio | ResponseAudio) -> None:
        if not isinstance(frame, ParticipantAudio) or self._session is None:
            return
        if frame.user_id not in self._whitelist:
            return
        listener = self._listeners.get(frame.user_id)
        if listener is None:
            listener = Listener(
                frame.user_id,
                self._settings.speech,
                self._vad,
                self._turn,
                self._asr,
                self._executor,
                self.clock,
                on_activity=self._handle,
                on_utterance=self._on_utterance,
            )
            self._listeners[frame.user_id] = listener
        listener.frame(frame.pcm, self._to_session(frame.t))

    # --- Сессия ---

    async def _start_session(self, name: str, t_ms: float, greet: bool) -> None:
        await self._end_session("новая Сессия")
        self._session = (name, t_ms)
        self._policy = self._new_policy()
        self._responder.history.clear()
        self._journal.open(name)
        log.info("Сессия %s, Журнал — %s", name, self._journal.path)
        if greet:
            # Приветствие вне Диалога: Политика о нём не знает.
            self._responder.greet()

    async def _end_session(self, reason: str) -> None:
        if self._session is None:
            return
        log.info("Сессия %s закончена: %s", self._session[0], reason)
        self._responder.interrupt()
        for listener in self._listeners.values():
            await listener.close()
        self._listeners.clear()
        self._marks.clear()
        self._journal.close()
        self._session = None

    # --- Политика ---

    def _on_utterance(self, u: Utterance, marks: UtteranceMarks) -> None:
        name = self._names.get(u.user_id)
        log.info("Реплика %s: %s", name or u.user_id, u.text)
        self._journal.utterance(u, name)
        self._marks[u] = marks
        self._handle(u)

    def _handle(self, event: Event) -> None:
        if self._session is None:
            return
        self._execute(self._policy.handle(event))

    def _execute(self, decisions: list[Decision]) -> None:
        for d in decisions:
            self._journal.decision(d, self._policy.state)
            if not isinstance(d, AddToContext):
                log.info("%s: %s", type(d).__name__, d.reason)
            match d:
                case Respond():
                    last = d.utterances[-1]
                    um = self._marks.get(last)
                    marks = Marks(
                        t_end=last.t_end,
                        turn_decided=um.turn_decided if um else None,
                        text_ready=um.text_ready if um else None,
                    )
                    question = tuple(self._said(u) for u in d.utterances)
                    self._responder.respond(question, last, marks)
                case AddToContext():
                    self._responder.add_context(self._said(d.utterance))
                case InterruptClone():
                    self._responder.interrupt()
                case StopAfterCurrentPhrase():
                    self._responder.stop_after_phrase()
                case CancelPendingResponse():
                    self._responder.cancel()

    def _said(self, u: Utterance) -> SaidByParticipant:
        return SaidByParticipant(self._names.get(u.user_id, "Участник"), u.text)

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(TICK_S)
            if self._session is None:
                continue
            now = self.clock()
            for listener in list(self._listeners.values()):
                listener.tick(now)
            self._execute(self._policy.tick(now))

    async def _memory_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(MEMORY_S)
            if self._session is None or not self._settings.journal.metrics:
                continue
            sample = await loop.run_in_executor(None, self._memory.sample)
            self._journal.memory(self.clock(), sample)
