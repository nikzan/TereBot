import numpy as np
import pytest
from bench_voice import Phrase, load_phrases, measure_phrases, percentile, summarize

SR = 100


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class FakeAdapter:
    """Каждый чанк — 1 с звука, на него уходит chunk_s секунд «синтеза»."""

    sample_rate = SR
    packages = ()

    def __init__(self, clock: Clock, streaming: bool, chunk_s: dict[str, float], chunks: int = 2):
        self.clock = clock
        self.streaming = streaming
        self.chunk_s = chunk_s
        self.chunks = chunks
        self.seeds = []
        self.texts = []

    def seed(self, seed):
        self.seeds.append(seed)

    def synthesize(self, text):
        self.texts.append(text)
        for _ in range(self.chunks):
            self.clock.t += self.chunk_s[text]
            yield np.zeros(SR, dtype=np.float32)


def run(streaming: bool):
    clock = Clock()
    speeds = {"прогрев": 100.0, "a": 0.5, "b": 1.0, "c": 2.0}
    adapter = FakeAdapter(clock, streaming, speeds)
    phrases = [Phrase(i, "short", t) for i, t in enumerate("abc", 1)]
    saved = []
    records = measure_phrases(
        adapter,
        phrases,
        [Phrase(0, "warmup", "прогрев")],
        seed=7,
        on_audio=lambda p, audio: saved.append(len(audio)) or f"{p.idx:04d}.wav",
        clock=clock,
    )
    return adapter, records, saved


def test_warmup_excluded_and_seed_before_each_phrase():
    adapter, records, _ = run(streaming=True)
    assert adapter.texts == ["прогрев", "a", "b", "c"]
    assert adapter.seeds == [7, 7, 7, 7]
    assert [r["text"] for r in records] == ["a", "b", "c"]


def test_rtf_and_first_audio_with_streaming():
    _, records, saved = run(streaming=True)
    assert saved == [2 * SR] * 3
    a = records[0]
    assert a["audio_s"] == 2
    assert a["synth_s"] == pytest.approx(1.0)
    assert a["rtf"] == pytest.approx(0.5)
    assert a["first_audio_s"] == pytest.approx(0.5)
    assert a["first_audio_is_full_synthesis"] is False
    assert a["chunks"] == 2


def test_first_audio_is_full_synthesis_without_streaming():
    _, records, _ = run(streaming=False)
    c = records[2]
    assert c["first_audio_s"] == pytest.approx(c["synth_s"]) == pytest.approx(4.0)
    assert c["first_audio_is_full_synthesis"] is True


def test_summary_median_and_p90():
    _, records, _ = run(streaming=True)
    s = summarize(records)
    # RTF фраз: 0.5, 1.0, 2.0; первый звук: 0.5, 1.0, 2.0.
    assert s["rtf"]["median"] == pytest.approx(1.0)
    assert s["rtf"]["p90"] == pytest.approx(1.8)
    assert s["first_audio_s"]["median"] == pytest.approx(1.0)
    assert s["synth_s_total"] == pytest.approx(7.0)
    assert s["audio_s_total"] == 6


@pytest.mark.parametrize("q", [0, 10, 50, 90, 100])
def test_percentile_matches_numpy(q):
    values = [3.0, 0.2, 7.5, 1.1, 4.4, 2.0, 9.9]
    assert percentile(values, q) == pytest.approx(np.percentile(values, q))


def test_load_phrases_splits_warmup_and_numbers_set(tmp_path):
    path = tmp_path / "phrases.txt"
    path.write_text(
        "# комментарий\n\nwarmup\tпрогрев\nshort\tАга.\nnumbers\tВ 19:30.\n", encoding="utf-8"
    )
    warmup, phrases = load_phrases(path)
    assert [p.text for p in warmup] == ["прогрев"]
    assert phrases == [Phrase(1, "short", "Ага."), Phrase(2, "numbers", "В 19:30.")]
