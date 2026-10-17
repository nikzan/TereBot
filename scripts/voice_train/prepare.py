"""Данные для дообучения Голоса из манифеста датасета.

    uv run --group train python scripts/voice_train/prepare.py --run pilot-2026-10-17 --label pilot

Ресемплирует сегменты обучающей выборки 48 → 24 кГц в data/voice/<прогон>/audio/, пишет
train.jsonl `{audio, text, ref_audio}` с абсолютными путями и run.json с данными прогона.
Пути абсолютные, поэтому запускайте там же, где будет обучение (на кластере).
"""

import argparse
import json
from datetime import date
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from terebot.dataset import DATASET_DIR
from terebot.dataset.jsonl import read_jsonl, write_jsonl, write_text
from terebot.voice_train import SAMPLE_RATE, SPEAKER, VOICE_DIR
from terebot.voice_train.data import pick_reference, training_rows


def resample(src: Path, dst: Path) -> None:
    audio, sr = sf.read(src, dtype="float32")
    if sr != SAMPLE_RATE:
        g = gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, np.clip(audio, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", required=True, help="имя прогона: каталог data/voice/<прогон>/")
    ap.add_argument("--label", required=True, choices=["pilot", "final"])
    ap.add_argument(
        "--ref", help="seg_id референса из обучающей выборки (по умолчанию — самый длинный)"
    )
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--out", type=Path, default=VOICE_DIR)
    args = ap.parse_args()

    train = read_jsonl(args.dataset / "manifest" / "train.jsonl")
    holdout = read_jsonl(args.dataset / "manifest" / "holdout.jsonl")
    try:
        ref = pick_reference(train, holdout, args.ref)
    except ValueError as e:
        raise SystemExit(f"референс: {e}") from None

    run_dir = args.out / args.run
    audio_dir = run_dir / "audio"
    rows = training_rows(train, holdout, audio_dir, ref)
    for r in train:
        resample(args.dataset / r["audio"], audio_dir / f"{r['seg_id']}.wav")
    write_jsonl(run_dir / "train.jsonl", rows)

    minutes = round(sum(r["duration"] for r in train) / 60, 2)
    run = {
        "run": args.run,
        "label": args.label,
        "prepared": date.today().isoformat(),
        "speaker": SPEAKER,
        "segments": len(rows),
        "minutes": minutes,
        "holdout_segments": len(holdout),
        "holdout_minutes": round(sum(r["duration"] for r in holdout) / 60, 2),
        "sessions": sorted({r["session"] for r in train}),
        "ref": {"seg_id": ref["seg_id"], "duration": ref["duration"], "text": ref["text"]},
        "sample_rate": SAMPLE_RATE,
    }
    write_text(run_dir / "run.json", json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print(
        f"{run_dir}: {len(rows)} сегментов, {minutes} мин, "
        f"референс {ref['seg_id']} ({ref['duration']} с)"
    )


if __name__ == "__main__":
    main()
