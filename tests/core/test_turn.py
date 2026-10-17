"""Конец Реплики и активность речи на синтетических последовательностях (спека speech-input)."""

from terebot.config import SpeechSettings
from terebot.core.turn import Activity, CheckTurn, EndUtterance, TurnDetector

WINDOW = 0.032  # окно Silero VAD: 512 сэмплов при 16 кГц
SETTINGS = SpeechSettings(silence_ms=200, continuity_ms=300, max_pause_ms=1500)


class Run:
    """Подаёт окна VAD подряд и собирает действия автомата."""

    def __init__(self, settings: SpeechSettings = SETTINGS):
        self.d = TurnDetector(settings)
        self.t = 0.0
        self.actions = []

    def speech(self, seconds: float, prob: float = 0.9) -> "Run":
        for _ in range(round(seconds / WINDOW)):
            self.actions += self.d.vad(self.t, self.t + WINDOW, prob)
            self.t += WINDOW
        return self

    def silence(self, seconds: float) -> "Run":
        return self.speech(seconds, prob=0.1)

    def no_frames(self, seconds: float, step: float = 0.1) -> "Run":
        """Кадры не приходят: только tick."""
        for _ in range(round(seconds / step)):
            self.t += step
            self.actions += self.d.tick(self.t)
        return self

    def answer(self, complete: bool) -> "Run":
        checks = [a for a in self.actions if isinstance(a, CheckTurn)]
        self.actions += self.d.turn_result(checks[-1].request_id, complete)
        return self

    def of(self, kind):
        return [a for a in self.actions if isinstance(a, kind)]


def test_pause_mid_thought_keeps_one_utterance():
    r = Run().speech(1.5).silence(0.3).answer(complete=False).silence(0.3).speech(1.5).silence(0.3)
    r.answer(complete=True)
    ends = r.of(EndUtterance)
    assert len(ends) == 1
    assert ends[0].t_start == 0
    assert abs(ends[0].t_end - 3.6) < WINDOW * 2
    assert ends[0].reason == "smart_turn"


def test_finished_question_ends_without_max_pause():
    r = Run().speech(1.5).silence(0.25)
    assert len(r.of(CheckTurn)) == 1
    r.answer(complete=True)
    [end] = r.of(EndUtterance)
    assert end.reason == "smart_turn"
    # Реплика кончилась через ~0,25 с тишины, а не через 1,5 с, и t_end — конец речи.
    assert r.t - end.t_end < 0.3
    assert abs(end.t_end - 1.504) < 1e-6


def test_half_word_ends_by_max_pause():
    r = Run().speech(1.0).silence(0.25).answer(complete=False).silence(1.0)
    assert r.of(EndUtterance) == []
    r.silence(0.3)
    [end] = r.of(EndUtterance)
    assert end.reason == "max_pause"
    assert abs(end.t_end - 0.992) < 1e-6


def test_frames_stop_arriving_counts_as_silence():
    r = Run().speech(1.0).no_frames(0.3)
    assert len(r.of(CheckTurn)) == 1
    r.answer(complete=False).no_frames(1.3)
    [end] = r.of(EndUtterance)
    assert end.reason == "max_pause"


def test_stale_turn_result_ignored_after_new_speech():
    r = Run().speech(1.0).silence(0.25)
    stale = r.of(CheckTurn)[-1].request_id
    r.speech(0.5)
    assert r.d.turn_result(stale, complete=True) == []
    assert r.d.in_utterance


def test_activity_every_100ms_and_reaches_1s_on_time():
    r = Run().speech(1.2)
    acts = r.of(Activity)
    gaps = [b.t - a.t for a, b in zip(acts, acts[1:], strict=False)]
    assert max(gaps) <= 0.2
    first_over_1s = next(a for a in acts if a.duration_ms >= 1000)
    assert first_over_1s.t <= 1.0 + 0.1


def test_short_pause_does_not_reset_activity():
    r = Run().speech(0.6).silence(0.2).speech(0.6)
    assert r.of(Activity)[-1].duration_ms >= 1300


def test_long_pause_resets_activity():
    r = Run().speech(0.6).silence(0.4)
    r.actions.clear()
    r.speech(0.6)
    assert r.of(Activity)[-1].duration_ms <= 600


def test_no_activity_without_speech():
    r = Run().silence(2.0).no_frames(2.0)
    assert r.actions == []


def test_blip_shorter_than_interval_gives_no_activity():
    r = Run().speech(0.064).silence(0.5)
    assert r.of(Activity) == []


def test_split_short_utterance_is_one_part():
    from terebot.core.turn import split_by_pauses

    assert split_by_pauses([0.9] * 10, 20, 0.5) == [(0, 10)]


def test_split_cuts_inside_longest_pause():
    from terebot.core.turn import split_by_pauses

    probs = [0.9] * 12 + [0.1] * 4 + [0.9] * 3 + [0.1] + [0.9] * 10
    parts = split_by_pauses(probs, 20, 0.5)
    assert parts[0] == (0, 14)  # середина паузы 12–15
    assert parts[-1][1] == len(probs)
    assert all(b - a <= 20 for a, b in parts)


def test_split_without_pauses_by_min_probability():
    from terebot.core.turn import split_by_pauses

    probs = [0.9] * 50
    probs[15] = 0.6
    parts = split_by_pauses(probs, 20, 0.5)
    assert parts[0] == (0, 15)
    assert [b - a for a, b in parts] and all(b - a <= 20 for a, b in parts)
    assert parts[-1][1] == 50
