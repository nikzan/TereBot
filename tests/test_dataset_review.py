from terebot.dataset.jsonl import read_jsonl, write_jsonl
from terebot.dataset.review import ACCEPTED, REJECTED, UNCHECKED, Review
from terebot.dataset.transcribe import transcribe_all


def segment(i, reject=None):
    sid = f"2026-10-20_20-00-00_{i:09d}"
    return {
        "seg_id": sid,
        "session": "2026-10-20_20-00-00",
        "start_ms": i,
        "duration": 5.0,
        "audio": None if reject else f"segments/{sid}.wav",
        "reject": reject,
    }


def make_dataset(tmp_path, n=5):
    segments = [segment(i * 10000) for i in range(n)] + [segment(999999, reject="короткий")]
    write_jsonl(tmp_path / "segments.jsonl", segments)
    return segments


def fake_asr(calls):
    def transcribe(wav):
        calls.append(wav.name)
        return f"авто {wav.stem}"

    return transcribe


def test_transcribe_adds_unchecked_entries_for_good_segments_only(tmp_path):
    segments = make_dataset(tmp_path)
    calls = []
    assert transcribe_all(tmp_path, fake_asr(calls)) == 5
    review = read_jsonl(tmp_path / "review.jsonl")
    assert [e["seg_id"] for e in review] == [s["seg_id"] for s in segments[:5]]
    assert all(e["status"] == UNCHECKED and e["auto_text"].startswith("авто") for e in review)


def test_retranscription_keeps_manual_work(tmp_path):
    segments = make_dataset(tmp_path)
    transcribe_all(tmp_path, fake_asr([]))
    review = Review(tmp_path / "review.jsonl", segments)
    first, second = segments[0]["seg_id"], segments[1]["seg_id"]
    review.edit(first, "ну э-э короче")
    review.accept(first)
    review.reject(second, "смех")

    calls = []
    assert transcribe_all(tmp_path, fake_asr(calls)) == 0
    assert calls == []
    transcribe_all(tmp_path, lambda wav: "другой текст", force=True)

    entries = {e["seg_id"]: e for e in read_jsonl(tmp_path / "review.jsonl")}
    assert (entries[first]["text"], entries[first]["status"]) == ("ну э-э короче", ACCEPTED)
    assert (entries[second]["status"], entries[second]["reason"]) == (REJECTED, "смех")
    assert entries[first]["auto_text"] == "другой текст"


def test_interrupted_review_resumes_after_last_decision(tmp_path):
    segments = make_dataset(tmp_path)
    transcribe_all(tmp_path, fake_asr([]))
    review = Review(tmp_path / "review.jsonl", segments)
    for _ in range(3):
        review.accept(review.next()["seg_id"])

    resumed = Review(tmp_path / "review.jsonl", segments)
    assert resumed.next()["seg_id"] == segments[3]["seg_id"]
    assert resumed.counts() == {UNCHECKED: 2, ACCEPTED: 3, REJECTED: 0}


def test_accept_without_edit_takes_auto_text(tmp_path):
    segments = make_dataset(tmp_path)
    transcribe_all(tmp_path, fake_asr([]))
    review = Review(tmp_path / "review.jsonl", segments)
    sid = review.next()["seg_id"]
    review.accept(sid)
    assert Review(tmp_path / "review.jsonl", segments).entries[sid]["text"] == f"авто {sid}"


def test_edit_alone_is_not_a_decision(tmp_path):
    segments = make_dataset(tmp_path)
    transcribe_all(tmp_path, fake_asr([]))
    review = Review(tmp_path / "review.jsonl", segments)
    sid = review.next()["seg_id"]
    review.edit(sid, "  щас  ")

    resumed = Review(tmp_path / "review.jsonl", segments)
    assert resumed.next()["seg_id"] == sid
    assert resumed.shown_text(sid) == "щас"


def test_entries_of_vanished_segments_are_kept_but_hidden(tmp_path):
    segments = make_dataset(tmp_path)
    transcribe_all(tmp_path, fake_asr([]))
    review = Review(tmp_path / "review.jsonl", segments)
    review.accept(segments[0]["seg_id"])

    shrunk = Review(tmp_path / "review.jsonl", segments[1:])
    assert segments[0]["seg_id"] not in [e["seg_id"] for e in shrunk.current()]
    shrunk.accept(shrunk.next()["seg_id"])
    assert segments[0]["seg_id"] in {e["seg_id"] for e in read_jsonl(tmp_path / "review.jsonl")}


def test_commands_work_in_russian_layout():
    from terebot.dataset.review import ACCEPT, EDIT, PLAY, QUIT, REJECT, parse_command

    assert [parse_command(k) for k in ("", "e", "у", "R", "к", "p", "з", "q", "й")] == [
        ACCEPT,
        EDIT,
        EDIT,
        REJECT,
        REJECT,
        PLAY,
        PLAY,
        QUIT,
        QUIT,
    ]
    assert parse_command("x") is None


def test_reason_is_number_or_words():
    from terebot.dataset.review import parse_reason

    assert parse_reason("1") == "смех"
    assert parse_reason(" кашель ") == "кашель"
    assert parse_reason("к") is None
    assert parse_reason("9") is None


def test_latin_words_found():
    from terebot.dataset.review import latin_words

    assert latin_words("A-a, убрать из дискорда") == ["A", "a"]
    assert latin_words("а-а, убрать из дискорда") == []
