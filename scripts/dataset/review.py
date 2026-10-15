"""Шаг 3 датасета Голоса: ручная вычитка сегментов в терминале.
Правила — docs/transcription-guide.md.

    uv run --group data python scripts/dataset/review.py

Сегмент играет через afplay по кругу, пока не принято решение; под ним — текст.
Клавиши (затем Enter):
    Enter — принять, e — исправить текст, r — отбраковать, p — играть с начала, q — выйти.
Клавиши работают и в русской раскладке (у, к, з, й).
Каждое решение сразу сохраняется в data/dataset/review.jsonl; следующий запуск продолжит
с первого непроверенного сегмента.
"""

import argparse
import os
import subprocess
import tempfile
import threading
from pathlib import Path

from prompt_toolkit import prompt

from terebot.dataset import DATASET_DIR
from terebot.dataset.jsonl import read_jsonl
from terebot.dataset.review import (
    ACCEPT,
    ACCEPTED,
    EDIT,
    PLAY,
    QUIT,
    REJECT,
    REJECT_REASONS,
    REJECTED,
    UNCHECKED,
    Review,
    latin_words,
    parse_command,
    parse_reason,
)

DIM, BOLD, GREEN, RED, YELLOW = "\033[2m", "\033[1m", "\033[32m", "\033[31m", "\033[33m"
RESET = "\033[0m"
PROMPT = (
    f"{DIM}Enter — принять · e — исправить · r — отбраковать · p — с начала · q — выйти{RESET} > "
)


class Player:
    """Играет сегмент по кругу с короткой паузой, пока его не остановят."""

    GAP_S = 0.7

    def __init__(self, wav: Path):
        self._wav = wav
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, args=(self._wav,), daemon=True)
        self._thread.start()

    def _loop(self, wav: Path) -> None:
        while not self._stop.is_set():
            with self._lock:
                if self._stop.is_set():
                    return
                self._proc = subprocess.Popen(["afplay", str(wav)])
            self._proc.wait()
            self._stop.wait(self.GAP_S)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        self._thread.join()

    def restart(self) -> None:
        self.stop()
        self._start()


def edit_text(text: str) -> str:
    """В $EDITOR, если он задан, иначе — однострочный редактор с подставленным текстом.

    readline здесь не годится: в Python из uv он собран на libedit, и подставить текст нельзя.
    """
    editor = os.environ.get("EDITOR")
    if editor:
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", encoding="utf-8", delete=False) as f:
            f.write(text + "\n")
        try:
            subprocess.run([*editor.split(), f.name], check=True)
            return Path(f.name).read_text(encoding="utf-8").strip()
        finally:
            os.unlink(f.name)
    return prompt("текст > ", default=text).strip()


def ask_reason() -> str | None:
    for i, reason in enumerate(REJECT_REASONS, 1):
        print(f"  {i} — {reason}")
    while answer := input("причина (номер или своим текстом, Enter — отмена) > ").strip():
        if reason := parse_reason(answer):
            return reason
        print(f"{YELLOW}нужен номер от 1 до {len(REJECT_REASONS)} или причина словами{RESET}")
    return None


def decide(review: Review, sid: str, player: Player) -> str:
    """Спрашивает команды по сегменту, пока не будет решения; возвращает его."""
    warned = False
    while True:
        print(f"  {review.shown_text(sid)}")
        command = parse_command(input(PROMPT))
        latin = latin_words(review.shown_text(sid))
        if command == ACCEPT and latin and not warned:
            # Второй Enter подряд принимает как есть: латиница бывает и намеренной.
            print(f"{YELLOW}латиница в тексте: {' '.join(latin)} — e, чтобы исправить{RESET}")
            warned = True
        elif command == ACCEPT:
            review.accept(sid)
            print(f"{GREEN}принят{RESET}")
            return ACCEPT
        elif command == EDIT:
            review.edit(sid, edit_text(review.shown_text(sid)))
            warned = False
        elif command == REJECT:
            reason = ask_reason()
            if reason:
                review.reject(sid, reason)
                print(f"{RED}отбракован: {reason}{RESET}")
                return REJECT
        elif command == PLAY:
            player.restart()
        elif command == QUIT:
            return QUIT
        else:
            print(f"{YELLOW}нет такой команды{RESET}")


def run(review: Review, segments: dict[str, dict], dataset_dir: Path) -> None:
    while (entry := review.next()) is not None:
        sid = entry["seg_id"]
        counts = review.counts()
        checked = counts[ACCEPTED] + counts[REJECTED]
        print(
            f"\n{BOLD}[{checked + 1}/{len(review.queue)}] {sid}{RESET} "
            f"{DIM}{segments[sid]['duration']:.1f} с{RESET}"
        )
        player = Player(dataset_dir / segments[sid]["audio"])
        try:
            if decide(review, sid, player) == QUIT:
                return
        finally:
            player.stop()
    print(f"\n{GREEN}Все сегменты вычитаны.{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    args = ap.parse_args()

    segments = read_jsonl(args.dataset / "segments.jsonl")
    review = Review(args.dataset / "review.jsonl", segments)
    try:
        run(review, {s["seg_id"]: s for s in segments}, args.dataset)
    except (KeyboardInterrupt, EOFError):
        print()
    counts = review.counts()
    print(
        f"Принято: {counts[ACCEPTED]}, отбраковано: {counts[REJECTED]}, "
        f"не проверено: {counts[UNCHECKED]}. Решения сохранены."
    )


if __name__ == "__main__":
    main()
