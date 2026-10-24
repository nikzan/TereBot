"""Слепой тест Голоса: пары «живая запись / синтез» для судей и подсчёт их ответов.

    uv run --group tts-mlx --group data python scripts/voice_train/blind_test.py prepare \\
        --name final-e8 --run final-2026-10-24 --model data/models/final-2026-10-24-e8-mlx-8bit
    uv run --group data python scripts/voice_train/blind_test.py score --name final-e8 \\
        data/voice/blind_tests/final-e8/answers-аня.csv answers-боря.csv …

prepare берёт 20 сегментов 4–12 с из отложенной выборки прогона, не бывших в прошлых тестах,
синтезирует их тексты на Mac через MLX, как в Клоне, пропускает синтез через Opus 96 кбит/с
туда и обратно, обрезает края обоих файлов по одному правилу, нормализует к −23 LUFS и пишет
WAV 48 кГц 16 бит без метаданных в data/voice/blind_tests/<имя>/pairs/: NN-A.wav, NN-B.wav и
лист ответов answers.csv. Ключ — key.json рядом с pairs/, судьям его не отдавать. Повторный
prepare с тем же --name (например, с другой эпохой) заменяет тест целиком.

score: ответ судьи — копия answers.csv, в колонке synthetic — A или B; имя судьи — из имени
файла после «answers-». Итог дописывается в docs/benchmarks/voice-finetune.md.
"""

import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import tempfile
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

    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else score_cmd(args)


if __name__ == "__main__":
    main()
