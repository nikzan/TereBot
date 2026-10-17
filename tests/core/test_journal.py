"""Сводка Журнала Сессии и тумблер транскриптов (спека session-journal)."""

import json

import pytest

from terebot.config import JournalSettings
from terebot.core.journal import Journal, summarize
from terebot.dialogue import PolicyState, Respond, Utterance
from terebot.dialogue.events import CloneState


def latency(to_sound: float, first_token: float) -> dict:
    stages = {"turn_decided": 0.3, "first_token": first_token, "playback_started": to_sound}
    return {"type": "latency", "stages": stages}


def memory(swap: float, system: float) -> dict:
    return {"type": "memory", "core_rss_gb": 1.0, "mlx_gb": 4.0, "llm_rss_gb": 3.0,
            "system_used_gb": system, "swap_gb": swap}  # fmt: skip


def test_summary_median_and_p90():
    lines = [latency(s, s / 2) for s in (1.0, 2.0, 3.0, 4.0, 10.0)]
    lines += [memory(2.0, 12.0), memory(2.5, 14.0), memory(2.25, 13.0)]
    s = summarize(lines)
    assert s["responses"] == 5
    assert s["to_first_sound_s"]["median"] == 3.0
    assert s["to_first_sound_s"]["p90"] == pytest.approx(7.6)
    assert s["stages_s"]["first_token"]["median"] == 1.5
    assert s["memory_peak_gb"]["system_used_gb"] == 14.0
    assert s["swap_gb"] == {"start": 2.0, "peak": 2.5, "growth": 0.5}


def test_summary_without_responses():
    assert summarize([]) == {"responses": 0}


STATE = PolicyState(1.0, True, "1", CloneState.PREPARING, False, None)


def test_transcripts_off_keeps_no_texts(tmp_path):
    journal = Journal(JournalSettings(transcripts=False, metrics=True, dir=tmp_path))
    journal.open("s")
    u = Utterance("1", "Теребот, секретный текст", 0.0, 1.0)
    journal.utterance(u, "Вася")
    journal.decision(Respond((u,), "Слово-вызов в Реплике «Теребот, секретный текст»"), STATE)
    journal.error("response", "сбой на «секретный текст»")
    journal.memory(1.0, memory(1.0, 10.0))
    journal.close()
    text = (tmp_path / "s" / "journal.jsonl").read_text(encoding="utf-8")
    assert "секретный" not in text
    kinds = [json.loads(line)["type"] for line in text.splitlines()]
    assert kinds == ["decision", "error", "memory", "summary"]
