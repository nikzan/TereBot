"""Политика Диалога: правила MVP из docs/PLAN.md, сценарии — tests/dialogue/scenarios/."""

from collections.abc import Iterable

from terebot.config import DialogueSettings
from terebot.dialogue.events import (
    AddToContext,
    CancelPendingResponse,
    CloneFinished,
    ClonePhraseEnded,
    CloneStarted,
    CloneState,
    Decision,
    EndDialogue,
    Event,
    InterruptClone,
    ParticipantLeft,
    PolicyState,
    Respond,
    ResponseFailed,
    SpeechActivity,
    StartDialogue,
    StopAfterCurrentPhrase,
    SwitchInterlocutor,
    Utterance,
)
from terebot.dialogue.wake_word import WakeMatch, WakeWordMatcher


def _quote(utterance: Utterance) -> str:
    return f"«{utterance.text}»"


def _wake_reason(match: WakeMatch, utterance: Utterance) -> str:
    return (
        f"найдено Слово-вызов «{match.candidate}» (≈ «{match.wake_word}», {match.score:.0f}) "
        f"в Реплике {_quote(utterance)}"
    )


class DialoguePolicy:
    """Детерминированная Политика: решения зависят только от событий и переданного времени."""

    def __init__(self, settings: DialogueSettings, whitelist: Iterable[str]):
        self._settings = settings
        self._whitelist = frozenset(whitelist)
        self._wake = WakeWordMatcher(
            settings.wake_words, settings.wake_threshold, settings.wake_blocklist
        )
        self._now = 0.0
        self._interlocutor: str | None = None
        self._clone = CloneState.IDLE
        # Реплики, на которые готовится ответ: новая Реплика до начала ответа к ним добавляется.
        self._pending: tuple[Utterance, ...] = ()
        # Клон договаривает фразу, а затем отвечает на `_queued` новому Собеседнику.
        self._stopping = False
        self._queued: tuple[Utterance, ...] = ()
        # Конец ответа Клона или последняя речь Собеседника — от этого идёт окно ожидания.
        self._last_activity = 0.0

    @property
    def state(self) -> PolicyState:
        window_left = None
        if self._window_running():
            elapsed = self._now - self._last_activity
            window_left = max(0.0, self._settings.window_s - elapsed)
        return PolicyState(
            now=self._now,
            in_dialogue=self._interlocutor is not None,
            interlocutor=self._interlocutor,
            clone=self._clone,
            stopping=self._stopping,
            window_left_s=window_left,
        )

    def tick(self, now: float) -> list[Decision]:
        self._now = max(self._now, now)
        return self._expire(now)

    def handle(self, event: Event) -> list[Decision]:
        # Окно проверяется на момент начала речи: Реплика, начатая после истечения окна,
        # уже не продолжает Диалог, даже если tick не вызывали.
        started = event.t_start if isinstance(event, Utterance | SpeechActivity) else event.t
        decisions = self._expire(started)
        self._now = max(self._now, event.t)
        match event:
            case Utterance():
                decisions += self._on_utterance(event)
            case SpeechActivity():
                decisions += self._on_speech(event)
            case CloneStarted():
                if self._clone is CloneState.PREPARING:
                    self._clone = CloneState.SPEAKING
            case ClonePhraseEnded():
                if self._stopping:
                    decisions += self._respond_queued("Клон договорил фразу")
            case CloneFinished():
                if self._stopping:
                    decisions += self._respond_queued("Клон договорил ответ")
                elif self._clone is CloneState.SPEAKING:
                    self._finish_response(event.t)
            case ResponseFailed():
                if self._clone is CloneState.PREPARING:
                    self._finish_response(event.t)
            case ParticipantLeft():
                if event.user_id == self._interlocutor:
                    decisions += self._on_interlocutor_left()
        return decisions

    # --- Реплики ---

    def _on_utterance(self, u: Utterance) -> list[Decision]:
        if u.user_id not in self._whitelist:
            return []
        match = self._wake.find(u.text)

        if self._interlocutor is None:
            if match is None:
                return [AddToContext(u, "Диалога нет, Слово-вызов не найдено")]
            self._interlocutor = u.user_id
            self._last_activity = u.t_end
            reason = _wake_reason(match, u)
            return [StartDialogue(u.user_id, u, reason), *self._respond((u,), reason)]

        if u.user_id != self._interlocutor:
            if match is None:
                return [AddToContext(u, "Реплика не Собеседника без Слова-вызова")]
            return self._switch(u, match)

        self._last_activity = u.t_end
        if self._stopping:
            self._queued += (u,)
            return []
        match self._clone:
            case CloneState.IDLE:
                reason = "повторный вызов Собеседником" if match else "Реплика Собеседника"
                return self._respond((u,), f"{reason}: {_quote(u)}")
            case CloneState.PREPARING:
                return [
                    CancelPendingResponse("Собеседник добавил Реплику до начала ответа"),
                    *self._respond(
                        self._pending + (u,),
                        f"ответ на все Реплики Собеседника, последняя {_quote(u)}",
                    ),
                ]
            case CloneState.SPEAKING:
                # Даже со Словом-вызовом: замолчать Собеседник может, только перебив Клона.
                return [AddToContext(u, "Реплика Собеседника поверх Клона без перебивания")]

    def _switch(self, u: Utterance, match: WakeMatch) -> list[Decision]:
        reason = _wake_reason(match, u)
        self._interlocutor = u.user_id
        self._last_activity = u.t_end
        if self._stopping:
            # Клон и так договаривает фразу; отвечать теперь новому вызвавшему.
            self._queued = (u,)
            return [SwitchInterlocutor(u.user_id, reason)]
        match self._clone:
            case CloneState.IDLE:
                return [SwitchInterlocutor(u.user_id, reason), *self._respond((u,), reason)]
            case CloneState.PREPARING:
                self._pending = ()
                return [
                    CancelPendingResponse(f"переключение Собеседника: {reason}"),
                    SwitchInterlocutor(u.user_id, reason),
                    *self._respond((u,), reason),
                ]
            case CloneState.SPEAKING:
                self._stopping = True
                self._queued = (u,)
                return [
                    StopAfterCurrentPhrase(f"переключение Собеседника: {reason}"),
                    SwitchInterlocutor(u.user_id, reason),
                ]

    # --- Речь и ответ ---

    def _on_speech(self, s: SpeechActivity) -> list[Decision]:
        if s.user_id != self._interlocutor:
            # Речь остальных и участников не из whitelist не перебивает Клона.
            return []
        self._last_activity = s.t
        speaking = self._clone is CloneState.SPEAKING
        if speaking and s.duration_ms >= self._settings.barge_in_ms:
            # После перебивания Клон молчит; ответ (с `_queued`, если было переключение)
            # будет на Реплику, которую Собеседник сейчас договаривает.
            self._clone = CloneState.IDLE
            self._stopping = False
            self._pending = ()
            return [InterruptClone(f"Собеседник говорит {s.duration_ms} мс подряд")]
        return []

    def _respond(self, utterances: tuple[Utterance, ...], reason: str) -> list[Decision]:
        utterances = self._queued + utterances
        self._queued = ()
        self._pending = utterances
        self._clone = CloneState.PREPARING
        return [Respond(utterances, reason)]

    def _respond_queued(self, why: str) -> list[Decision]:
        self._stopping = False
        queued, self._queued = self._queued, ()
        return self._respond(queued, f"{why}, ответ Собеседнику на {_quote(queued[-1])}")

    def _finish_response(self, t: float) -> None:
        self._clone = CloneState.IDLE
        self._pending = ()
        self._last_activity = max(self._last_activity, t)

    def _on_interlocutor_left(self) -> list[Decision]:
        decisions: list[Decision] = []
        if self._clone is CloneState.SPEAKING:
            decisions.append(InterruptClone("Собеседник вышел из канала"))
        elif self._clone is CloneState.PREPARING:
            decisions.append(CancelPendingResponse("Собеседник вышел из канала"))
        decisions.append(EndDialogue("Собеседник вышел из канала"))
        self._reset()
        return decisions

    # --- Окно ожидания ---

    def _window_running(self) -> bool:
        return (
            self._interlocutor is not None and self._clone is CloneState.IDLE and not self._stopping
        )

    def _expire(self, now: float) -> list[Decision]:
        if not self._window_running() or now - self._last_activity <= self._settings.window_s:
            return []
        silence = now - self._last_activity
        self._reset()
        return [EndDialogue(f"окно ожидания истекло: Собеседник молчит {silence:.1f} с")]

    def _reset(self) -> None:
        self._interlocutor = None
        self._clone = CloneState.IDLE
        self._pending = ()
        self._stopping = False
        self._queued = ()
