import pytest

from terebot.config import DialogueSettings
from terebot.dialogue.wake_word import WakeWordMatcher

DEFAULTS = DialogueSettings()


def matcher(wake_words=DEFAULTS.wake_words) -> WakeWordMatcher:
    return WakeWordMatcher(wake_words, DEFAULTS.wake_threshold, DEFAULTS.wake_blocklist)


@pytest.mark.parametrize(
    "text",
    ["тере бот, привет", "Теребод, ты тут?", "теребот", "слушай, теребот, а ты в кс играешь?",
     "Терёбот!", "Теребот?"],
)  # fmt: skip
def test_found(text):
    assert matcher().find(text) is not None


@pytest.mark.parametrize(
    "text",
    ["Терем, ты где был?", "Терешкин, иди сюда", "это же терабайт", "не тереблю я", "теремом",
     "да он опять за своё", ""],
)  # fmt: skip
def test_not_found(text):
    assert matcher().find(text) is None


def test_blocklist_protects_with_lower_threshold():
    loose = WakeWordMatcher(DEFAULTS.wake_words, 60, DEFAULTS.wake_blocklist)
    assert loose.find("Терем, ты где был?") is None
    assert WakeWordMatcher(DEFAULTS.wake_words, 60).find("Терем, ты где был?") is not None


def test_changed_wake_word():
    m = matcher(["Тереша"])
    assert m.find("Тереша, привет") is not None
    assert m.find("Теребот, привет") is None


def test_match_reports_candidate_and_score():
    match = matcher().find("слушай, тере бот")
    assert match.candidate == "теребот"
    assert match.score == 100
