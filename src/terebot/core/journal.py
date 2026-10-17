"""Журнал Сессии: data/sessions/<Сессия>/journal.jsonl, по строке JSON на событие.

Каждая строка пишется сразу с flush, чтобы при аварийной остановке ядра всё записанное
сохранилось. Тумблеры независимы: `transcripts` — Реплики, решения Политики, ответы и ошибки
(при выключенном вместо текстов — только длины), `metrics` — задержка, память и сводка.
"""

import dataclasses
import datetime as dt
import json
import re
import statistics
from pathlib import Path
from typing import Any

from terebot.config import JournalSettings
from terebot.dialogue import Decision, PolicyState, Utterance

_QUOTED = re.compile(r"«[^»]*»")

# Этапы задержки ответа в порядке цикла; отсчёт — от конца Реплики (`t_end`).
LATENCY_STAGES = (
    "turn_decided",
    "text_ready",
    "llm_request",
    "first_token",
    "first_phrase",
    "first_voice",
    "playback_started",
)


def percentile(values: list[float], q: float) -> float:
    """Перцентиль с линейной интерполяцией (как numpy.percentile по умолчанию)."""
    s = sorted(values)
    pos = (len(s) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(lines: list[dict]) -> dict:
    """Сводка Сессии по строкам Журнала: ответы, задержка до первого звука, память, swap."""
    latencies = [x for x in lines if x["type"] == "latency"]
    to_sound = [x["stages"]["playback_started"] for x in latencies]
    memory = [x for x in lines if x["type"] == "memory"]
    summary: dict[str, Any] = {"responses": len(latencies)}
    if to_sound:
        summary["to_first_sound_s"] = {
            "median": statistics.median(to_sound),
            "p90": percentile(to_sound, 90),
        }
        stages = {}
        for stage in LATENCY_STAGES:
            values = [x["stages"][stage] for x in latencies if x["stages"].get(stage) is not None]
            if values:
                stages[stage] = {"median": statistics.median(values), "p90": percentile(values, 90)}
        summary["stages_s"] = stages
    if memory:
        summary["memory_peak_gb"] = {
            key: max(m[key] for m in memory if m.get(key) is not None)
            for key in ("core_rss_gb", "mlx_gb", "llm_rss_gb", "system_used_gb")
            if any(m.get(key) is not None for m in memory)
        }
        summary["swap_gb"] = {
            "start": memory[0]["swap_gb"],
            "peak": max(m["swap_gb"] for m in memory),
            "growth": max(m["swap_gb"] for m in memory) - memory[0]["swap_gb"],
        }
    return summary


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, tuple | list):
        return [_jsonable(v) for v in value]
    return value


class Journal:
    def __init__(self, settings: JournalSettings):
        self._settings = settings
        self._file = None
        self._lines: list[dict] = []
        self.path: Path | None = None

    @property
    def is_open(self) -> bool:
        return self._file is not None

    def open(self, session: str) -> None:
        self.close()
        self.path = self._settings.dir / session / "journal.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Переподключение шлюза посреди Сессии: дописываем в тот же Журнал.
        self._file = open(self.path, "a", encoding="utf-8")  # noqa: SIM115
        self._lines = []

    def close(self) -> None:
        if self._file is None:
            return
        if self._settings.metrics:
            self._write("summary", None, summarize(self._lines))
        self._file.close()
        self._file = None

    def _write(self, kind: str, t: float | None, fields: dict) -> None:
        if self._file is None:
            return
        line = {"type": kind, "t": t, "wall": dt.datetime.now().astimezone().isoformat(), **fields}
        self._file.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._file.flush()
        if kind in ("latency", "memory"):
            self._lines.append(line)

    # --- Транскрипты ---

    def utterance(self, u: Utterance, name: str | None) -> None:
        if self._settings.transcripts:
            self._write(
                "utterance",
                u.t_end,
                {"user_id": u.user_id, "name": name, "text": u.text, "t_start": u.t_start},
            )

    def decision(self, decision: Decision, state: PolicyState) -> None:
        fields = _jsonable(decision)
        if not self._settings.transcripts:
            # Причина цитирует Реплику, а в решениях есть сами Реплики: остаются только длины.
            fields["reason"] = _QUOTED.sub("«…»", fields["reason"])
            for key in ("utterance", "utterances"):
                if key in fields:
                    items = fields[key] if key == "utterances" else [fields[key]]
                    for item in items:
                        item["text_len"] = len(item.pop("text"))
        self._write(
            "decision",
            state.now,
            {"decision": type(decision).__name__, **fields, "state": _jsonable(state)},
        )

    def response(self, resp, spoken: str, interrupted: bool) -> None:
        phrases = [
            {"llm": p.llm_text, "voice": p.voice_text, "audio_s": p.samples / 48_000}
            for p in resp.phrases
        ]
        fields = {
            "response_id": resp.id,
            "greeting": resp.greeting,
            "interrupted": interrupted,
            "phrases_played": resp.ended,
            "error": resp.error,
        }
        if self._settings.transcripts:
            fields |= {"llm_text": resp.llm_text, "phrases": phrases, "spoken": spoken}
        else:
            fields |= {
                "llm_text_len": len(resp.llm_text),
                "spoken_len": len(spoken),
                "phrases": [{"audio_s": p["audio_s"]} for p in phrases],
            }
        self._write("response", resp.marks.playback_started, fields)

    def error(self, where: str, message: str) -> None:
        if not self._settings.transcripts:
            message = _QUOTED.sub("«…»", message)
        self._write("error", None, {"where": where, "message": message})

    # --- Метрики ---

    def latency(self, resp) -> None:
        if not self._settings.metrics or resp.marks.t_end is None:
            return
        m = resp.marks
        stages = {
            s: (None if getattr(m, s) is None else getattr(m, s) - m.t_end) for s in LATENCY_STAGES
        }
        self._write("latency", m.playback_started, {"response_id": resp.id, "stages": stages})

    def memory(self, t: float, sample: dict) -> None:
        if self._settings.metrics:
            self._write("memory", t, sample)
