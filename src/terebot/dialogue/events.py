"""События на входе Политики Диалога, решения на выходе и снимок её состояния.

Время — секунды на общей шкале Сессии (float). Откуда оно берётся, решает вызывающий код.
"""

from dataclasses import dataclass
from enum import StrEnum

# --- События ---


@dataclass(frozen=True)
class Utterance:
    """Готовая Реплика: участник договорил, распознавание выдало текст."""

    user_id: str
    text: str
    t_start: float
    t_end: float

    @property
    def t(self) -> float:
        return self.t_end


@dataclass(frozen=True)
class SpeechActivity:
    """Участник говорит прямо сейчас: непрерывная речь длится `duration_ms` к моменту `t`."""

    user_id: str
    duration_ms: int
    t: float

    @property
    def t_start(self) -> float:
        return self.t - self.duration_ms / 1000


@dataclass(frozen=True)
class CloneStarted:
    """Голос начал произносить подготовленный ответ."""

    t: float


@dataclass(frozen=True)
class ClonePhraseEnded:
    """Голос договорил очередную фразу ответа."""

    t: float


@dataclass(frozen=True)
class CloneFinished:
    """Голос договорил ответ целиком."""

    t: float


@dataclass(frozen=True)
class ResponseFailed:
    """LLM или Голос не смогли подготовить ответ."""

    t: float


@dataclass(frozen=True)
class ParticipantLeft:
    """Участник вышел из голосового канала."""

    user_id: str
    t: float


type Event = (
    Utterance
    | SpeechActivity
    | CloneStarted
    | ClonePhraseEnded
    | CloneFinished
    | ResponseFailed
    | ParticipantLeft
)


# --- Решения. У каждого есть `reason` для Журнала Сессии. ---


@dataclass(frozen=True)
class StartDialogue:
    user_id: str
    utterance: Utterance
    reason: str


@dataclass(frozen=True)
class Respond:
    """Подготовить и произнести ответ на эти Реплики Собеседника."""

    utterances: tuple[Utterance, ...]
    reason: str


@dataclass(frozen=True)
class AddToContext:
    """Учесть Реплику в контексте, не отвечая на неё."""

    utterance: Utterance
    reason: str


@dataclass(frozen=True)
class InterruptClone:
    """Замолчать немедленно."""

    reason: str


@dataclass(frozen=True)
class StopAfterCurrentPhrase:
    """Договорить звучащую фразу, остаток ответа не произносить."""

    reason: str


@dataclass(frozen=True)
class CancelPendingResponse:
    """Бросить подготовку ответа, который ещё не начал звучать."""

    reason: str


@dataclass(frozen=True)
class SwitchInterlocutor:
    user_id: str
    reason: str


@dataclass(frozen=True)
class EndDialogue:
    reason: str


type Decision = (
    StartDialogue
    | Respond
    | AddToContext
    | InterruptClone
    | StopAfterCurrentPhrase
    | CancelPendingResponse
    | SwitchInterlocutor
    | EndDialogue
)


# --- Состояние ---


class CloneState(StrEnum):
    IDLE = "idle"
    # Ответ готовится, Клон ещё не говорит.
    PREPARING = "preparing"
    SPEAKING = "speaking"


@dataclass(frozen=True)
class PolicyState:
    """Снимок для Журнала Сессии на момент `now` — время последнего события или `tick`."""

    now: float
    in_dialogue: bool
    interlocutor: str | None
    clone: CloneState
    # Клон договаривает фразу перед ответом новому Собеседнику.
    stopping: bool
    # None, пока окно не идёт: Диалога нет, Клон говорит или готовит ответ.
    window_left_s: float | None
