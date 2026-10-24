import numpy as np
import pytest

from terebot.dataset.jsonl import write_jsonl
from terebot.voice_train.data import pad_silence, pick_reference, training_rows


def seg(sid, duration=5.0):
    return {
        "seg_id": sid,
        "text": f"текст {sid}",
        "duration": duration,
        "audio": f"segments/{sid}.wav",
    }


@pytest.fixture
def manifest():
    train = [seg("s03", 9.0), seg("s01", 12.0), seg("s02", 12.0), seg("s04")]
    holdout = [seg("h01", 14.0), seg("h02")]
    return train, holdout


def test_holdout_never_in_training_set(manifest, tmp_path):
    train, holdout = manifest
    rows = training_rows(train, holdout, tmp_path, pick_reference(train, holdout))
    used = " ".join(r["audio"] + r["ref_audio"] for r in rows)
    assert len(rows) == len(train)
    assert "h01" not in used and "h02" not in used


def test_overlapping_manifest_is_rejected(manifest, tmp_path):
    train, holdout = manifest
    with pytest.raises(ValueError, match="s04"):
        training_rows(train, [*holdout, seg("s04")], tmp_path, train[0])


def test_single_reference_longest_then_first_by_seg_id(manifest, tmp_path):
    train, holdout = manifest
    ref = pick_reference(train, holdout)
    assert ref["seg_id"] == "s01"
    rows = training_rows(train, holdout, tmp_path, ref)
    assert {r["ref_audio"] for r in rows} == {str(tmp_path.resolve() / "s01.wav")}


def test_explicit_reference(manifest):
    train, holdout = manifest
    assert pick_reference(train, holdout, "s04")["seg_id"] == "s04"


def test_holdout_reference_is_refused(manifest):
    train, holdout = manifest
    with pytest.raises(ValueError, match="отложенной"):
        pick_reference(train, holdout, "h01")


def test_unknown_reference_is_refused(manifest):
    train, holdout = manifest
    with pytest.raises(ValueError, match="нет в обучающей"):
        pick_reference(train, holdout, "s99")


def test_repeated_preparation_is_identical(manifest, tmp_path):
    train, holdout = manifest

    def prepare(order):
        rows = training_rows(order, holdout, tmp_path, pick_reference(order, holdout))
        path = tmp_path / f"train-{len(order)}-{order[0]['seg_id']}.jsonl"
        write_jsonl(path, rows)
        return path.read_bytes()

    assert prepare(train) == prepare(list(reversed(train)))


def test_pad_silence_adds_exact_silence_around_speech():
    speech = np.linspace(-0.5, 0.5, 4 * 24000, dtype=np.float32)
    padded = pad_silence(speech, 24000, 150, 300)
    assert len(padded) == int(4.45 * 24000)
    assert padded.dtype == speech.dtype
    assert not padded[:3600].any() and not padded[-7200:].any()
    assert np.array_equal(padded[3600:-7200], speech)
