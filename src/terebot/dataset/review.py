"""Состояние вычитки: `review.jsonl` — по строке на годный сегмент нарезки.

Строка: {seg_id, auto_text, text, status, reason}. `auto_text` пишет транскрипция, `text`,
`status` и `reason` — только вычитка. Файл перезаписывается атомарно после каждого решения.
"""

import re
from pathlib import Path

from terebot.dataset.jsonl import read_jsonl, write_jsonl

UNCHECKED = "unchecked"
ACCEPTED = "accepted"
REJECTED = "rejected"
STATUS_RU = {UNCHECKED: "не проверен", ACCEPTED: "принят", REJECTED: "отбракован"}

# Причины отбраковки при вычитке — docs/transcription-guide.md.
REJECT_REASONS = ("смех", "чужой голос", "обрыв", "артефакты", "шум", "неразборчиво")

# Команды вычитки; клавиши работают и в русской раскладке.
ACCEPT, EDIT, REJECT, PLAY, QUIT = "accept", "edit", "reject", "play", "quit"
_COMMANDS = {"": ACCEPT, "e": EDIT, "у": EDIT, "r": REJECT, "к": REJECT, "p": PLAY, "з": PLAY}
_COMMANDS |= {"q": QUIT, "й": QUIT}
_LATIN = re.compile(r"[A-Za-z]+")


def parse_command(answer: str) -> str | None:
    return _COMMANDS.get(answer.strip().lower())


def parse_reason(answer: str) -> str | None:
    """Номер причины или своя причина от трёх символов; иначе None — переспросить."""
    answer = answer.strip()
    if answer.isdigit() and 1 <= int(answer) <= len(REJECT_REASONS):
        return REJECT_REASONS[int(answer) - 1]
    if len(answer) >= 3 and not answer.isdigit():
        return answer
    return None


def latin_words(text: str | None) -> list[str]:
    """Латиница в тексте: по руководству всё пишется кириллицей, часто это опечатка раскладки."""
    return _LATIN.findall(text or "")


def new_entry(seg_id: str) -> dict:
    return {"seg_id": seg_id, "auto_text": None, "text": None, "status": UNCHECKED, "reason": None}


class Review:
    """Вычитка сегментов из `segments`; строки review.jsonl для исчезнувших сегментов хранятся, но
    не показываются: после повторной нарезки с теми же seg_id ручная работа вернётся."""

    def __init__(self, path: Path, segments: list[dict]):
        self.path = path
        self.entries = {e["seg_id"]: e for e in read_jsonl(path)}
        self.queue = sorted(s["seg_id"] for s in segments if not s["reject"])

    def current(self) -> list[dict]:
        return [self.entries[sid] for sid in self.queue if sid in self.entries]

    def next(self) -> dict | None:
        """Первый непроверенный сегмент с автоматическим текстом."""
        return next(
            (e for e in self.current() if e["status"] == UNCHECKED and e["auto_text"] is not None),
            None,
        )

    def shown_text(self, seg_id: str) -> str:
        e = self.entries[seg_id]
        return e["text"] if e["text"] is not None else e["auto_text"]

    def edit(self, seg_id: str, text: str) -> None:
        self.entries[seg_id]["text"] = text.strip()
        self.save()

    def accept(self, seg_id: str) -> None:
        e = self.entries[seg_id]
        e.update(text=self.shown_text(seg_id), status=ACCEPTED, reason=None)
        self.save()

    def reject(self, seg_id: str, reason: str) -> None:
        self.entries[seg_id].update(status=REJECTED, reason=reason)
        self.save()

    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(STATUS_RU, 0)
        for e in self.current():
            counts[e["status"]] += 1
        counts[UNCHECKED] += len(self.queue) - len(self.current())
        return counts

    def save(self) -> None:
        write_jsonl(self.path, sorted(self.entries.values(), key=lambda e: e["seg_id"]))
