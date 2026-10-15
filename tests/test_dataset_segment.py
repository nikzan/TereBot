import json

import numpy as np
import pytest
import soundfile as sf

from terebot.dataset.jsonl import read_jsonl
from terebot.dataset.segment import (
    CLIPPED,
    SAMPLE_RATE,
    SHORT,
    Params,
    loudness,
    segment_all,
    split_fragment,
)

SR = SAMPLE_RATE
PROTOTYPE = "111111111111111111"
OTHER = "222222222222222222"


def energy_vad(audio, sr):
    """Речь — 10-мс окна с RMS выше порога; в синтетике паузы — точные нули."""
    win = sr // 100
    speech, start = [], None
    for i in range(0, len(audio), win):
        loud = np.sqrt(np.mean(audio[i : i + win] ** 2)) > 1e-3
        if loud and start is None:
            start = i
        if not loud and start is not None:
            speech.append((start, i))
            start = None
    if start is not None:
        speech.append((start, len(audio)))
    return speech


def speech(seconds, level=0.05, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(seconds * SR)) * level).astype(
        np.float32
    )


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def write_session(root, name, user_id, fragments):
    """fragments — [(start_ms, audio)]; пишет WAV без тишины между фрагментами, как шлюз."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    lines, offset = [], 0
    for start_ms, audio in fragments:
        lines.append(
            {
                "start_ms": start_ms,
                "duration_ms": round(len(audio) * 1000 / SR),
                "sample_offset": offset,
            }
        )
        offset += len(audio)
    sf.write(d / f"{user_id}.wav", np.concatenate([a for _, a in fragments]), SR, "PCM_16")
    (d / f"{user_id}.segments.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8"
    )
    return lines


@pytest.fixture
def recordings(tmp_path):
    root = tmp_path / "recordings"
    long_phrase = np.concatenate(
        [
            speech(6, seed=1),
            silence(0.3),
            speech(5, seed=2),
            silence(0.8),
            speech(9, seed=3),
            silence(0.4),
            speech(5, seed=4),
        ]
    )
    frags = write_session(
        root,
        "2026-10-20_20-00-00",
        PROTOTYPE,
        [
            (1000, np.concatenate([silence(0.5), speech(2, seed=5), silence(0.3)])),  # 2 с речи
            (
                61000,
                np.concatenate([silence(0.4), speech(4, seed=6), silence(0.4)]),
            ),  # через минуту
            (70000, long_phrase),  # 26,5 с одним фрагментом
        ],
    )
    write_session(root, "2026-10-20_20-00-00", OTHER, [(5000, speech(8, level=0.2, seed=9))])
    clipped = np.sign(speech(5, seed=7)).astype(np.float32)
    write_session(root, "2026-10-21_20-00-00", PROTOTYPE, [(2000, clipped)])
    return root, frags


def run(tmp_path, root):
    dataset = tmp_path / "dataset"
    segment_all(root, dataset, PROTOTYPE, energy_vad)
    return dataset, read_jsonl(dataset / "segments.jsonl")


def test_good_segments_within_limits_and_fragments(tmp_path, recordings):
    root, frags = recordings
    _, rows = run(tmp_path, root)
    good = [r for r in rows if not r["reject"]]
    assert len(good) >= 3
    for r in good:
        assert 3.0 <= r["duration"] <= 15.0
        start, end = r["start_ms"], r["start_ms"] + r["duration"] * 1000
        assert any(
            f["start_ms"] <= start and end <= f["start_ms"] + f["duration_ms"] + 1 for f in frags
        ), r


def test_fragments_never_glued(tmp_path, recordings):
    root, _ = recordings
    _, rows = run(tmp_path, root)
    session = [r for r in rows if r["session"] == "2026-10-20_20-00-00"]
    first, second = session[0], session[1]
    assert first["reject"] == SHORT and first["start_ms"] < 61000
    assert second["reject"] is None and second["start_ms"] >= 61000


def test_other_participant_ignored_and_session_kept(tmp_path, recordings):
    root, _ = recordings
    _, rows = run(tmp_path, root)
    assert {r["session"] for r in rows} == {"2026-10-20_20-00-00", "2026-10-21_20-00-00"}
    # Фрагмент другого участника начинался на 5000 мс — у Прототипа там ничего нет.
    assert not any(
        4000 <= r["start_ms"] <= 13000 for r in rows if r["session"].startswith("2026-10-20")
    )


def test_clipped_fragment_rejected(tmp_path, recordings):
    root, _ = recordings
    dataset, rows = run(tmp_path, root)
    clipped = [r for r in rows if r["session"] == "2026-10-21_20-00-00"]
    assert [r["reject"] for r in clipped] == [CLIPPED]
    assert not (dataset / "segments" / f"{clipped[0]['seg_id']}.wav").exists()


def test_loudness_normalized(tmp_path, recordings):
    root, _ = recordings
    dataset, rows = run(tmp_path, root)
    for r in rows:
        if not r["reject"]:
            audio, sr = sf.read(dataset / r["audio"], dtype="float32")
            assert loudness(audio, sr) == pytest.approx(-23.0, abs=0.5)


def test_resegmentation_keeps_ids(tmp_path, recordings):
    root, _ = recordings
    dataset, rows = run(tmp_path, root)
    segment_all(root, dataset, PROTOTYPE, energy_vad, force=True)
    again = read_jsonl(dataset / "segments.jsonl")
    assert [r["seg_id"] for r in again] == [r["seg_id"] for r in rows]
    assert sorted(p.name for p in (dataset / "segments").iterdir()) == sorted(
        f"{r['seg_id']}.wav" for r in rows if not r["reject"]
    )


def test_long_fragment_split_at_longest_pause():
    # 6 с речи, пауза 0,3 с, 5 с, пауза 0,8 с, 9 с: делим по паузе 0,8 с.
    s = [(0, 6 * SR), (int(6.3 * SR), int(11.3 * SR)), (int(12.1 * SR), int(21.1 * SR))]
    pieces = split_fragment(int(21.1 * SR), s, SR, Params())
    assert len(pieces) == 2
    assert int(11.3 * SR) < pieces[0][1] <= pieces[1][0] < int(12.1 * SR)
