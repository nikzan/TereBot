import numpy as np
import pytest

from terebot.voice_train.blind import (
    Pair,
    make_pairs,
    score,
    trim_silence,
    used_seg_ids,
)


def holdout(n=40):
    # Длительности 2–14 с: часть сегментов вне 4–12 с.
    return [
        {"seg_id": f"s{i:03d}", "text": f"текст {i}", "duration": 2.0 + (i % 13)} for i in range(n)
    ]


def test_pairs_reproducible_by_seed():
    a = make_pairs(holdout(), set(), 10, seed=7)
    assert a == make_pairs(holdout(), set(), 10, seed=7)
    assert a != make_pairs(holdout(), set(), 10, seed=8)
    assert {p.synthetic for p in a} == {"A", "B"}
    assert [p.pair for p in a] == [f"{i:02d}" for i in range(1, 11)]


def test_pairs_only_4_to_12_seconds():
    rows = {r["seg_id"]: r for r in holdout()}
    for p in make_pairs(holdout(), set(), 15, seed=0):
        assert 4.0 <= rows[p.seg_id]["duration"] <= 12.0


def test_used_segments_not_repeated():
    journal = {"e8": {"model": "m", "speaker": "prototype", "seg_ids": ["s004", "s005", "s006"]}}
    used = used_seg_ids(journal)
    for seed in range(20):
        assert not {p.seg_id for p in make_pairs(holdout(), used, 10, seed)} & used
    # Повторная подготовка того же теста свои сегменты не исключает.
    assert used_seg_ids(journal, except_test="e8") == set()


def test_not_enough_segments():
    with pytest.raises(ValueError, match="исчерпана"):
        make_pairs(holdout(10), set(), 20, seed=0)


def key(n=20, seed=0):
    return [Pair(f"{i:02d}", f"s{i}", "т", "AB"[(i + seed) % 2]) for i in range(1, n + 1)]


def judge(k, right):
    """Ответы судьи, угадавшего синтез в первых `right` парах."""
    flip = {"A": "B", "B": "A"}
    return {p.pair: p.synthetic if i < right else flip[p.synthetic] for i, p in enumerate(k)}


def test_spec_example_three_judges_pass():
    k = key()
    s = score(k, {"аня": judge(k, 14), "боря": judge(k, 12), "вика": judge(k, 13)})
    assert (s.correct, s.total) == (39, 60)
    assert s.by_judge["аня"] == (14, 20)
    assert s.share == pytest.approx(0.65)
    assert s.passed


def test_easy_to_spot_fails():
    k = key()
    s = score(k, {"аня": judge(k, 16), "боря": judge(k, 16), "вика": judge(k, 16)})
    assert s.share == pytest.approx(0.8)
    assert not s.passed


def test_blank_answers_not_counted_and_case_ignored():
    k = key(4)
    ans = {"01": k[0].synthetic.lower(), "02": " ", "03": k[2].synthetic, "04": ""}
    assert score(k, {"аня": ans}).by_judge["аня"] == (2, 2)


def test_bad_answers_rejected():
    k = key(2)
    with pytest.raises(ValueError, match="A или B"):
        score(k, {"аня": {"01": "C"}})
    with pytest.raises(ValueError, match="нет в ключе"):
        score(k, {"аня": {"07": "A"}})


def test_trim_silence_same_margins():
    sr = 48000
    speech = np.sin(np.linspace(0, 2000, sr)).astype(np.float32) * 0.5
    for lead, tail in [(0.2, 0.5), (0.12, 0.12), (0.0, 0.0), (0.02, 0.0)]:
        audio = np.concatenate([np.zeros(int(lead * sr)), speech, np.zeros(int(tail * sr))])
        out = trim_silence(audio.astype(np.float32), sr)
        # Речь целиком, по 50 мс (±1 рамка) по краям.
        assert abs(len(out) - (sr + 2 * sr // 20)) <= sr // 100
