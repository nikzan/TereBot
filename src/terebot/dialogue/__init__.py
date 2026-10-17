"""Политика Диалога: события канала → решения, кого слушать и когда говорить.

Чистая детерминированная логика без звука, моделей и часов: время приходит в событиях
и в `DialoguePolicy.tick`. Сценарии поведения — tests/dialogue/scenarios/*.yaml.
"""

from terebot.dialogue.events import (
    AddToContext,
    CancelPendingResponse,
    CloneFinished,
    ClonePhraseEnded,
    CloneStarted,
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
from terebot.dialogue.policy import DialoguePolicy
from terebot.dialogue.wake_word import WakeMatch, WakeWordMatcher

__all__ = [
    "AddToContext",
    "CancelPendingResponse",
    "CloneFinished",
    "ClonePhraseEnded",
    "CloneStarted",
    "Decision",
    "DialoguePolicy",
    "EndDialogue",
    "Event",
    "InterruptClone",
    "ParticipantLeft",
    "PolicyState",
    "Respond",
    "ResponseFailed",
    "SpeechActivity",
    "StartDialogue",
    "StopAfterCurrentPhrase",
    "SwitchInterlocutor",
    "Utterance",
    "WakeMatch",
    "WakeWordMatcher",
]
