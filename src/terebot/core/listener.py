"""Восприятие речи одного участника: кадры 48 кГц → активность речи и готовые Реплики.

Звук прореживается до 16 кГц, Silero VAD размечает окна по 32 мс, автомат `TurnDetector`
решает, где конец Реплики (с Smart Turn), а GigaAM распознаёт её целиком в пуле потоков.
Время — секунды на шкале Сессии.
"""

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass

import numpy as np

from terebot.config import SpeechSettings
from terebot.core.audio import Downsampler48to16, pcm_to_float
from terebot.core.models import SAMPLE_RATE, Asr, SileroVad, SmartTurn
from terebot.core.turn import Activity, CheckTurn, EndUtterance, TurnDetector, split_by_pauses
from terebot.dialogue import SpeechActivity, Utterance

log = logging.getLogger(__name__)

WINDOW = SileroVad.WINDOW
WINDOW_S = WINDOW / SAMPLE_RATE
# Звук до первого окна речи: начало слова VAD часто пропускает, а распознаванию оно нужно.
PREROLL_WINDOWS = 10
# Звук после конца речи, который ещё идёт в распознавание.
TAIL_S = 0.1


@dataclass(frozen=True)
class UtteranceMarks:
    """Метки задержки Реплики для Журнала: решение о конце хода и готовый текст."""

    turn_decided: float
    text_ready: float


@dataclass
class _Window:
    audio: np.ndarray
    prob: float
    t_start: float


class Listener:
    def __init__(
        self,
        user_id: str,
        settings: SpeechSettings,
        vad: SileroVad,
        turn: SmartTurn,
        asr: Asr,
        executor: Executor,
        clock: Callable[[], float],
        on_activity: Callable[[SpeechActivity], None],
        on_utterance: Callable[[Utterance, UtteranceMarks], None],
    ):
        self.user_id = user_id
        self._settings = settings
        self._vad = vad.stream()
        self._turn = turn
        self._asr = asr
        self._executor = executor
        self._clock = clock
        self._on_activity = on_activity
        self._on_utterance = on_utterance
        self._detector = TurnDetector(settings)
        self._down = Downsampler48to16()
        self._pending = np.zeros(0, dtype=np.float32)
        self._windows: list[_Window] = []
        self._preroll: deque[_Window] = deque(maxlen=PREROLL_WINDOWS)
        self._tasks: set[asyncio.Task] = set()

    def frame(self, pcm: bytes, t: float) -> None:
        """Кадр 20 мс, полученный шлюзом в момент `t` (конец кадра)."""
        self._pending = np.concatenate([self._pending, self._down(pcm_to_float(pcm))])
        while len(self._pending) >= WINDOW:
            audio, self._pending = self._pending[:WINDOW], self._pending[WINDOW:]
            t_end = t - len(self._pending) / SAMPLE_RATE
            window = _Window(audio, self._vad(audio), t_end - WINDOW_S)
            if self._detector.in_utterance or window.prob > self._settings.vad_threshold:
                if not self._windows:
                    self._windows.extend(self._preroll)
                    self._preroll.clear()
                self._windows.append(window)
            else:
                self._preroll.append(window)
            self._apply(self._detector.vad(window.t_start, t_end, window.prob))

    def tick(self, now: float) -> None:
        self._apply(self._detector.tick(now))

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def _apply(self, actions) -> None:
        for a in actions:
            match a:
                case Activity():
                    self._on_activity(SpeechActivity(self.user_id, a.duration_ms, a.t))
                case CheckTurn():
                    audio = np.concatenate([w.audio for w in self._windows])
                    self._spawn(self._check_turn(a.request_id, audio))
                case EndUtterance():
                    windows = [w for w in self._windows if w.t_start <= a.t_end + TAIL_S]
                    self._windows = []
                    self._spawn(self._recognize(a, windows, self._clock()))

    def _spawn(self, coro) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _check_turn(self, request_id: int, audio: np.ndarray) -> None:
        loop = asyncio.get_running_loop()
        prob = await loop.run_in_executor(self._executor, self._turn, audio)
        self._apply(self._detector.turn_result(request_id, prob > self._settings.turn_threshold))

    async def _recognize(self, end: EndUtterance, windows: list[_Window], decided: float) -> None:
        if not windows:
            return
        loop = asyncio.get_running_loop()
        max_windows = max(1, int(self._settings.asr_max_s / WINDOW_S))
        parts = split_by_pauses(
            [w.prob for w in windows], max_windows, self._settings.vad_threshold
        )
        texts = []
        for a, b in parts:
            audio = np.concatenate([w.audio for w in windows[a:b]])
            texts.append(await loop.run_in_executor(self._executor, self._asr, audio))
        text = " ".join(t for t in texts if t)
        # На кашле и шуме распознавание возвращает пустую строку или одну пунктуацию.
        if not any(c.isalnum() for c in text):
            log.info(
                "user=%s: пустой текст Реплики %.2f–%.2f, отброшена",
                self.user_id,
                end.t_start,
                end.t_end,
            )
            return
        utterance = Utterance(self.user_id, text, end.t_start, end.t_end)
        self._on_utterance(utterance, UtteranceMarks(decided, self._clock()))
