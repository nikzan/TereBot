import pytest

from terebot.dataset.build import block_of, build
from terebot.dataset.jsonl import read_jsonl, write_jsonl
from terebot.dataset.review import ACCEPTED, REJECTED, UNCHECKED


@pytest.fixture
def dataset(tmp_path):
    """Три Сессии по 20 минут, сегмент 6 с каждые 8 с; почти все приняты."""
    segments, review = [], []
    for day in (20, 21, 22):
        session = f"2026-10-{day}_20-00-00"
        for start_ms in range(0, 20 * 60_000, 8000):
            sid = f"{session}_{start_ms:09d}"
            segments.append(
                {
                    "seg_id": sid,
                    "session": session,
                    "start_ms": start_ms,
                    "duration": 6.0,
                    "audio": f"segments/{sid}.wav",
                    "reject": None,
                }
            )
            status = REJECTED if start_ms % 80_000 == 0 else ACCEPTED
            review.append(
                {
                    "seg_id": sid,
                    "auto_text": "авто",
                    "text": f"текст {sid}",
                    "status": status,
                    "reason": "смех" if status == REJECTED else None,
                }
            )
    segments.append(
        {
            "seg_id": "2026-10-22_20-00-00_099999999",
            "session": "2026-10-22_20-00-00",
            "start_ms": 99999999,
            "duration": 1.2,
            "audio": None,
            "reject": "короткий",
        }
    )
    review[-1]["status"] = UNCHECKED
    write_jsonl(tmp_path / "segments.jsonl", segments)
    write_jsonl(tmp_path / "review.jsonl", review)
    return tmp_path


def files(d):
    return {
        p: (d / p).read_bytes()
        for p in ("manifest/train.jsonl", "manifest/holdout.jsonl", "report.md")
    }


def test_rebuild_is_identical(dataset):
    build(dataset)
    first = files(dataset)
    build(dataset)
    assert files(dataset) == first


def test_blocks_not_shared_between_splits(dataset):
    build(dataset)
    segments = {s["seg_id"]: s for s in read_jsonl(dataset / "segments.jsonl")}
    train = read_jsonl(dataset / "manifest" / "train.jsonl")
    holdout = read_jsonl(dataset / "manifest" / "holdout.jsonl")
    train_blocks = {block_of(segments[r["seg_id"]]) for r in train}
    holdout_blocks = {block_of(segments[r["seg_id"]]) for r in holdout}
    assert holdout_blocks and not train_blocks & holdout_blocks


def test_holdout_about_ten_percent_of_accepted(dataset):
    build(dataset)
    train = read_jsonl(dataset / "manifest" / "train.jsonl")
    holdout = read_jsonl(dataset / "manifest" / "holdout.jsonl")
    share = sum(r["duration"] for r in holdout) / sum(r["duration"] for r in train + holdout)
    assert 0.08 <= share <= 0.2
    assert set(train[0]) == {"audio", "text", "duration", "session", "seg_id"}


def test_only_accepted_in_manifest_and_report_counts(dataset):
    text = build(dataset)
    rows = read_jsonl(dataset / "manifest" / "train.jsonl") + read_jsonl(
        dataset / "manifest" / "holdout.jsonl"
    )
    review = {e["seg_id"]: e for e in read_jsonl(dataset / "review.jsonl")}
    assert all(review[r["seg_id"]]["status"] == ACCEPTED for r in rows)
    assert "| вычитка | смех |" in text and "| нарезка | короткий | 1 |" in text
    assert "не проверено 1." in text


def test_single_block_goes_to_train(tmp_path):
    sid = "2026-10-20_20-00-00_000001000"
    write_jsonl(
        tmp_path / "segments.jsonl",
        [
            {
                "seg_id": sid,
                "session": "2026-10-20_20-00-00",
                "start_ms": 1000,
                "duration": 5.0,
                "audio": f"segments/{sid}.wav",
                "reject": None,
            }
        ],
    )
    write_jsonl(
        tmp_path / "review.jsonl",
        [{"seg_id": sid, "auto_text": "а", "text": "а", "status": ACCEPTED, "reason": None}],
    )
    build(tmp_path)
    assert len(read_jsonl(tmp_path / "manifest" / "train.jsonl")) == 1
    assert read_jsonl(tmp_path / "manifest" / "holdout.jsonl") == []
