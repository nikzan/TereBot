"""Конец Реплики и активность речи одного участника: чистый конечный автомат.

На входе — вероятности VAD по окнам звука, результаты Smart Turn и ход времени (`tick`:
кадры могут перестать приходить, ведь Discord-клиент не шлёт тишину). На выходе — действия:
сообщить Политике длительность речи, запустить Smart Turn, закончить Реплику.
Время — секунды на шкале Сессии; конец окна VAD — его время.
"""

from dataclasses import dataclass

from terebot.config import SpeechSettings

# Как часто сообщать Политике длительность непрерывной речи.
ACTIVITY_INTERVAL_S = 0.1


@dataclass(frozen=True)
class Activity:
    """Участник говорит непрерывно `duration_ms` к моменту `t`."""

    duration_ms: int
    t: float


@dataclass(frozen=True)
class CheckTurn:
    """Тишина после речи: спросить Smart Turn, договорил ли участник (звук до `t_end`)."""

    request_id: int
    t_end: float


@dataclass(frozen=True)
class EndUtterance:
    t_start: float
    t_end: float
    # "smart_turn" — модель решила, что участник договорил; "max_pause" — предельная пауза.
    reason: str


type Action = Activity | CheckTurn | EndUtterance


class TurnDetector:
    def __init__(self, settings: SpeechSettings):
        self._vad_threshold = settings.vad_threshold
        self._silence = settings.silence_ms / 1000
        self._continuity = settings.continuity_ms / 1000
        self._max_pause = settings.max_pause_ms / 1000
        self._requests = 0
        self._reset()

    def _reset(self) -> None:
        # Начало Реплики и конец последнего окна с речью; None — Реплики нет.
        self._t_start: float | None = None
        self._last_speech: float | None = None
        # Начало непрерывной речи (паузы короче порога непрерывности не прерывают её).
        self._run_start: float | None = None
        self._last_activity: float | None = None
        # Ожидаемый ответ Smart Turn и конец речи, по которому он уже спрошен.
        self._pending: int | None = None
        self._checked_for: float | None = None

    @property
    def in_utterance(self) -> bool:
        return self._t_start is not None

    def vad(self, t_start: float, t_end: float, prob: float) -> list[Action]:
        """Окно звука [t_start, t_end] и вероятность речи в нём."""
        if prob <= self._vad_threshold:
            return self._advance(t_end)
        actions: list[Action] = []
        if self._t_start is None:
            self._t_start = t_start
        gap = None if self._last_speech is None else t_start - self._last_speech
        if self._run_start is None or (gap is not None and gap >= self._continuity):
            self._run_start = t_start
            self._last_activity = None
        self._last_speech = t_end
        # Речь продолжилась: прежний вопрос к Smart Turn больше не актуален.
        self._pending = None
        since = self._last_activity if self._last_activity is not None else self._run_start
        if t_end - since >= ACTIVITY_INTERVAL_S - 1e-9:
            self._last_activity = t_end
            actions.append(Activity(round((t_end - self._run_start) * 1000), t_end))
        return actions

    def tick(self, now: float) -> list[Action]:
        return self._advance(now)

    def turn_result(self, request_id: int, complete: bool) -> list[Action]:
        if request_id != self._pending:
            return []
        self._pending = None
        if not complete:
            # Ждём продолжения не дольше предельной паузы — это решит _advance.
            return []
        return [self._end("smart_turn")]

    def _advance(self, now: float) -> list[Action]:
        if self._last_speech is None:
            return []
        silence = now - self._last_speech
        if silence >= self._max_pause - 1e-9:
            return [self._end("max_pause")]
        if silence >= self._silence - 1e-9 and self._checked_for != self._last_speech:
            self._requests += 1
            self._pending = self._requests
            self._checked_for = self._last_speech
            return [CheckTurn(self._pending, self._last_speech)]
        return []

    def _end(self, reason: str) -> EndUtterance:
        end = EndUtterance(self._t_start, self._last_speech, reason)
        self._reset()
        return end


def split_by_pauses(
    probs: list[float], max_windows: int, threshold: float
) -> list[tuple[int, int]]:
    """Делит Реплику (вероятности VAD по окнам) на части не длиннее `max_windows` окон.

    Резать стараемся посреди самой длинной паузы во второй половине допустимой части,
    а если пауз нет — в окне с наименьшей вероятностью речи. Возвращает [начало, конец) в окнах.
    """
    parts = []
    start, n = 0, len(probs)
    while n - start > max_windows:
        lo, hi = start + max_windows // 2, start + max_windows
        best, best_len, run = None, 0, 0
        for i in range(lo, hi):
            run = run + 1 if probs[i] <= threshold else 0
            if run > best_len:
                best_len, best = run, i - (run - 1) // 2
        if best is None:
            best = min(range(lo, hi), key=lambda i: probs[i])
        cut = max(best, start + 1)
        parts.append((start, cut))
        start = cut
    parts.append((start, n))
    return parts
