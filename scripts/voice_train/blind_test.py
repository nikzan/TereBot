"""Слепой тест Голоса: пары «живая запись / синтез» для судей и подсчёт их ответов.

Запуск — `uv run --exact --group tts-mlx --with pyloudnorm==0.2.0 python <этот файл> …`
(группа data не подходит, см. README):

    … prepare --name final-e8 --run final-2026-10-24 \\
        --model data/models/final-2026-10-24-e8-mlx-8bit
    … listen --name final-e8 --judge аня
    … score --name final-e8 data/voice/blind_tests/final-e8/answers-*.csv

prepare берёт 20 сегментов 4–12 с из отложенной выборки прогона, не бывших в прошлых тестах,
синтезирует их тексты на Mac через MLX, как в Клоне, пропускает синтез через Opus 96 кбит/с
туда и обратно, обрезает края обоих файлов по одному правилу, нормализует к −23 LUFS и пишет
WAV 48 кГц 16 бит без метаданных в data/voice/blind_tests/<имя>/pairs/: NN-A.wav, NN-B.wav и
лист ответов answers.csv. Ключ — key.json рядом с pairs/, судьям его не отдавать. Повторный
prepare с тем же --name (например, с другой эпохой) заменяет тест целиком.

listen — проведение теста с одним компьютером (например, стрим экрана в Discord): пары по
очереди проигрываются через afplay, ведущий нажимает A или B за судью, ответ сразу пишется
в data/voice/blind_tests/<имя>/answers-<судья>.csv. Ключ не читается и на экран ничего
не выдаёт; прерванный тест продолжается с первой пары без ответа.

score: ответ судьи — копия answers.csv или файл listen, в колонке synthetic — A или B; имя
судьи — из имени файла после «answers-». Итог дописывается в docs/benchmarks/voice-finetune.md.
"""

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import termios
import time
import tty
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from terebot.dataset.jsonl import read_jsonl, write_text
from terebot.dataset.segment import normalize_loudness
from terebot.voice_train import SPEAKER, VOICE_DIR
from terebot.voice_train.blind import (
    PASS_SHARE,
    Pair,
    make_pairs,
    score,
    trim_silence,
    used_seg_ids,
)

ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = VOICE_DIR / "blind_tests"
REPORT = ROOT / "docs" / "benchmarks" / "voice-finetune.md"
SR = 48000


def opus_roundtrip(audio: np.ndarray, sr: int, tmp: Path) -> np.ndarray:
    """Синтез через Opus 96 кбит/с туда и обратно: живые записи прошли тот же кодек Discord."""
    src, enc, dec = tmp / "src.wav", tmp / "enc.opus", tmp / "dec.wav"
    sf.write(src, audio, sr, subtype="FLOAT")
    run = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    subprocess.run([*run, "-i", src, "-c:a", "libopus", "-b:a", "96k", enc], check=True)
    subprocess.run([*run, "-i", enc, "-ar", str(SR), "-ac", "1", dec], check=True)
    out, _ = sf.read(dec, dtype="float32")
    return out


def finish(audio: np.ndarray, dst: Path) -> None:
    """Общая обработка обоих файлов пары: края, −23 LUFS, WAV 48 кГц 16 бит без метаданных."""
    audio = normalize_loudness(trim_silence(audio, SR), SR)
    sf.write(dst, np.clip(audio, -1.0, 1.0), SR, subtype="PCM_16")


def prepare(args: argparse.Namespace) -> None:
    from terebot.voice import MlxVoice

    run_dir = VOICE_DIR / args.run
    holdout = read_jsonl(run_dir / "holdout.jsonl")
    if not holdout:
        raise SystemExit(f"в {run_dir} нет holdout.jsonl — отложенная выборка не зафиксирована")
    journal_path = args.out / "used.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.exists() else {}
    try:
        pairs = make_pairs(holdout, used_seg_ids(journal, args.name), args.n, args.seed)
    except ValueError as e:
        raise SystemExit(str(e)) from None

    voice = MlxVoice(streaming_interval=0.5, cache_limit_gb=1.0)
    voice.load(args.model)
    try:
        voice.use_speaker(args.speaker)
    except ValueError as e:
        raise SystemExit(f"{args.model}: {e}") from None

    test_dir = args.out / args.name
    shutil.rmtree(test_dir, ignore_errors=True)
    pairs_dir = test_dir / "pairs"
    pairs_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        for p in pairs:
            live, sr = sf.read(run_dir / "holdout" / f"{p.seg_id}.wav", dtype="float32")
            if sr != SR:
                raise SystemExit(f"{p.seg_id}: живая запись {sr} Гц, ожидается {SR}")
            voice.seed(args.seed)
            synth = np.concatenate(list(voice.synthesize(p.text)))
            synth = opus_roundtrip(synth, voice.sample_rate, Path(tmp))
            files = {"A": live, "B": synth} if p.synthetic == "B" else {"A": synth, "B": live}
            for side, audio in files.items():
                finish(audio, pairs_dir / f"{p.pair}-{side}.wav")
            print(f"  пара {p.pair}", flush=True)

    rows = "".join(f"{p.pair},\n" for p in pairs)
    write_text(pairs_dir / "answers.csv", "pair,synthetic\n" + rows)
    key = {
        "name": args.name,
        "model": args.model,
        "speaker": args.speaker,
        "run": args.run,
        "seed": args.seed,
        "date": dt.date.today().isoformat(),
        "pairs": [asdict(p) for p in pairs],
    }
    write_text(test_dir / "key.json", json.dumps(key, ensure_ascii=False, indent=2) + "\n")
    journal[args.name] = {
        "model": args.model,
        "speaker": args.speaker,
        "date": key["date"],
        "seg_ids": [p.seg_id for p in pairs],
    }
    write_text(journal_path, json.dumps(journal, ensure_ascii=False, indent=2) + "\n")
    print(f"{pairs_dir}: {len(pairs)} пар и answers.csv; ключ — {test_dir / 'key.json'}")


def read_answers(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8", newline="") as f:
        return {r["pair"].strip(): (r.get("synthetic") or "") for r in csv.DictReader(f)}


KEYS_A, KEYS_B = set("aAфФ"), set("bBиИ")
KEYS_QUIT = set("qQйЙ\x03")
LEFT, RIGHT, BACKSPACE = "\x1b[D", "\x1b[C", "\x7f"


def read_key() -> str:
    """Одна клавиша без Enter; в русской раскладке — те же физические клавиши."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return os.read(fd, 8).decode("utf-8", "ignore")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def play(path: Path, label: str) -> None:
    print(f"  ▶ {label}", end="", flush=True)
    subprocess.run(["afplay", str(path)], check=True)
    print("\r" + " " * 10 + "\r", end="", flush=True)


def listen(args: argparse.Namespace) -> None:
    test_dir = args.out / args.name
    pairs_dir = test_dir / "pairs"
    template = pairs_dir / "answers.csv"
    if not template.exists():
        raise SystemExit(f"нет {template} — сначала prepare")
    pairs = list(read_answers(template))
    out = test_dir / f"answers-{args.judge}.csv"
    answers = {p: "" for p in pairs} | (read_answers(out) if out.exists() else {})

    def save() -> None:
        rows = "".join(f"{p},{answers[p]}\n" for p in pairs)
        write_text(out, "pair,synthetic\n" + rows)

    i = next((k for k, p in enumerate(pairs) if not answers[p]), len(pairs))
    autoplay = True
    while i < len(pairs):
        p = pairs[i]
        done = sum(bool(v) for v in answers.values())
        print("\033[2J\033[H", end="")
        print(f"Слепой тест · судья {args.judge} · пара {p} из {len(pairs)} · отвечено {done}\n")
        print("Где синтез?  A / B — ответ    1 / 2 — послушать A / B    пробел — обе")
        print("← назад    → пропустить    q — выход (ответы сохранены)\n")
        if answers[p]:
            print(f"  ответ: {answers[p]}")
        if autoplay:
            play(pairs_dir / f"{p}-A.wav", "A")
            time.sleep(0.7)
            play(pairs_dir / f"{p}-B.wav", "B")
        autoplay = False
        key = read_key()
        if key in KEYS_A or key in KEYS_B:
            answers[p] = "A" if key in KEYS_A else "B"
            save()
            i, autoplay = i + 1, True
        elif key == "1":
            play(pairs_dir / f"{p}-A.wav", "A")
        elif key == "2":
            play(pairs_dir / f"{p}-B.wav", "B")
        elif key == " ":
            autoplay = True
        elif key in (LEFT, BACKSPACE):
            i, autoplay = max(0, i - 1), True
        elif key == RIGHT:
            i, autoplay = i + 1, True
        elif key in KEYS_QUIT:
            break
    save()
    done = sum(bool(v) for v in answers.values())
    print("\033[2J\033[H", end="")
    print(f"Судья {args.judge}: отвечено {done} из {len(pairs)}. Ответы — {out}")
    if done < len(pairs):
        print("Продолжить — та же команда, начнётся с первой пары без ответа.")


def score_cmd(args: argparse.Namespace) -> None:
    test_dir = args.out / args.name
    key = json.loads((test_dir / "key.json").read_text(encoding="utf-8"))
    pairs = [Pair(**p) for p in key["pairs"]]
    answers = {}
    for path in args.answers:
        judge = path.stem.removeprefix("answers-").removeprefix("answers") or path.stem
        answers[judge] = read_answers(path)
    try:
        s = score(pairs, answers)
    except ValueError as e:
        raise SystemExit(str(e)) from None

    lines = [f"- {j}: угадал синтез в {c} из {t}" for j, (c, t) in s.by_judge.items()]
    verdict = "пройден" if s.passed else "не пройден"
    total = (
        f"Итог: {s.correct} из {s.total} ({s.share:.0%}), порог — не больше {PASS_SHARE:.0%}: "
        f"**тест {verdict}**."
    )
    print("\n".join(lines) + "\n" + total)
    if args.no_report:
        return
    section = (
        f"\n\n### Слепой тест `{args.name}`\n\n"
        f"{dt.date.today().isoformat()}. Голос — `{key['model']}`, говорящий `{key['speaker']}`; "
        f"{len(pairs)} пар из отложенной выборки `{key['run']}`, сид {key['seed']}, "
        f"судей — {len(s.by_judge)}, ответов — {s.total}.\n\n" + "\n".join(lines) + f"\n\n{total}\n"
    )
    REPORT.write_text(REPORT.read_text(encoding="utf-8").rstrip("\n") + section, encoding="utf-8")
    print(f"дописано в {REPORT.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=TESTS_DIR, help="каталог тестов и used.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("prepare", help="пары для судей, лист ответов и ключ")
    pr.add_argument("--name", required=True, help="имя теста, например final-e8")
    pr.add_argument("--run", required=True, help="прогон с зафиксированной отложенной выборкой")
    pr.add_argument("--model", required=True, help="MLX-конвертация дообученного Голоса")
    pr.add_argument("--speaker", default=SPEAKER)
    pr.add_argument("--n", type=int, default=20, help="число пар")
    pr.add_argument("--seed", type=int, default=0)

    sc = sub.add_parser("score", help="подсчёт ответов судей")
    sc.add_argument("--name", required=True)
    sc.add_argument("answers", nargs="+", type=Path, help="answers-<судья>.csv")
    sc.add_argument("--no-report", action="store_true", help="не дописывать итог в отчёт")

    li = sub.add_parser("listen", help="провести тест: пары по очереди, ответ клавишей A/B")
    li.add_argument("--name", required=True)
    li.add_argument("--judge", required=True, help="имя судьи: файл answers-<судья>.csv")

    args = ap.parse_args()
    {"prepare": prepare, "score": score_cmd, "listen": listen}[args.cmd](args)


if __name__ == "__main__":
    main()
