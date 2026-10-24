"""Синтез набора оценки для чекпоинтов прогона и zero-shot (PyTorch, CUDA).

    CUDA_VISIBLE_DEVICES=1 HF_HOME=$PWD/data/hf .venv/bin/qwen-lora \\
        scripts/voice_train/synth_eval.py pilot-2026-10-18 --epochs 5 9 --zero-shot

Набор — тексты отложенной выборки прогона (категория `holdout`, из data/voice/<прогон>/
holdout.jsonl) и config/eval/phrases_ru.txt без прогрева (категории бенчмарка). Чекпоинт синтезирует
`generate_custom_voice` с говорящим prototype, zero-shot — Base `generate_voice_clone` с референсом
и его текстом из run.json. Сид (0) ставится перед каждой фразой.

Результат — data/voice/<прогон>/eval/<checkpoint-epoch-N | zero-shot>/ с WAV и results.json
в формате бенчмарка. Метрики — на Mac после rsync:

    uv run --group eval python -m terebot.voice_eval data/voice/<прогон>/eval/* \\
        --refs data/voice/<прогон>/holdout

У прогонов, подготовленных до фиксации отложенной выборки, holdout.jsonl нет: передайте
--holdout с holdout.jsonl того манифеста, на котором прогон обучен (у pilot-2026-10-18 —
data/voice/pilot-2026-10-18/dataset-manifest/holdout.jsonl), и она будет скопирована в каталог
прогона. Аудио берётся из data/dataset/ (--dataset).
"""

import argparse
import datetime as dt
import importlib.metadata
import json
import re
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_voice import load_phrases  # noqa: E402

from terebot.dataset import DATASET_DIR  # noqa: E402
from terebot.dataset.jsonl import read_jsonl  # noqa: E402
from terebot.voice_train import SPEAKER, VOICE_DIR  # noqa: E402
from terebot.voice_train.curve import ZERO_SHOT  # noqa: E402
from terebot.voice_train.data import freeze_holdout  # noqa: E402

BASE_DIR = ROOT / "data" / "models" / "Qwen3-TTS-12Hz-1.7B-Base"
FRAME_RATE = 12.5  # кадров кодека в секунду
HOLDOUT = "holdout"
PACKAGES = ("qwen-tts", "transformers", "torch")


def eval_set(holdout: list[dict]) -> list[dict]:
    _, bench = load_phrases()
    items = [{"category": HOLDOUT, "text": r["text"], "seg_id": r["seg_id"]} for r in holdout]
    items += [{"category": p.category, "text": p.text} for p in bench]
    return [{"idx": i, **it} for i, it in enumerate(items, start=1)]


def synth(
    tts, items: list[dict], out: Path, seed: int, max_new_tokens: int, **kw
) -> tuple[list[dict], int]:
    out.mkdir(parents=True, exist_ok=True)
    gen = tts.generate_custom_voice if "speaker" in kw else tts.generate_voice_clone
    records = []
    for it in items:
        torch.manual_seed(seed)
        t = time.perf_counter()
        wavs, sr = gen(it["text"], language="Russian", max_new_tokens=max_new_tokens, **kw)
        synth_s = time.perf_counter() - t
        name = f"{it['idx']:04d}.wav"
        sf.write(out / name, wavs[0], sr)
        audio_s = len(wavs[0]) / sr
        # На лимите кодек отдаёт max_new_tokens - 1 кадр (последний токен — не аудио).
        hit = audio_s * FRAME_RATE >= max_new_tokens - 1 - 0.5
        records.append(
            {
                **it,
                "wav": name,
                "synth_s": synth_s,
                "audio_s": audio_s,
                "rtf": synth_s / audio_s,
                "hit_limit": hit,
            }
        )
        print(f"  {name} {audio_s:6.2f} с{'  ДО ЛИМИТА' if hit else ''}", flush=True)
    return records, sr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", help="имя прогона: каталог data/voice/<прогон>/")
    ap.add_argument("--epochs", type=int, nargs="*", help="только эти эпохи (по умолчанию все)")
    ap.add_argument("--zero-shot", action="store_true", help="синтезировать и zero-shot Base")
    ap.add_argument("--holdout", type=Path, help="манифест holdout.jsonl для старых прогонов")
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR, help="корень аудио для --holdout")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = VOICE_DIR / args.run
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if (run_dir / "holdout.jsonl").exists():
        if args.holdout:
            raise SystemExit(f"{run_dir}/holdout.jsonl уже есть, --holdout не нужен")
        holdout = read_jsonl(run_dir / "holdout.jsonl")
    elif args.holdout:
        holdout = freeze_holdout(read_jsonl(args.holdout), args.dataset, run_dir)
    else:
        raise SystemExit(f"в {run_dir} нет holdout.jsonl — передайте --holdout <манифест>")
    items = eval_set(holdout)

    ckpts = sorted(
        (int(m.group(1)), p)
        for p in run_dir.glob("checkpoint-epoch-*")
        if (m := re.fullmatch(r"checkpoint-epoch-(\d+)", p.name))
    )
    if args.epochs is not None:
        missing = set(args.epochs) - {n for n, _ in ckpts}
        if missing:
            raise SystemExit(f"нет чекпоинтов эпох {sorted(missing)} в {run_dir}")
        ckpts = [(n, p) for n, p in ckpts if n in args.epochs]
    if not ckpts and not args.zero_shot:
        raise SystemExit(f"нечего синтезировать: нет чекпоинтов в {run_dir} и не задан --zero-shot")

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    def load(path: Path):
        return Qwen3TTSModel.from_pretrained(
            str(path), dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0"
        )

    ref_audio = run_dir / "audio" / f"{run['ref']['seg_id']}.wav"
    jobs = [(ZERO_SHOT, BASE_DIR)] if args.zero_shot else []
    jobs += [(p.name, p) for _, p in ckpts]
    env = {
        "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
        "gpu": torch.cuda.get_device_name(0),
    }
    for name, path in jobs:
        print(name, flush=True)
        tts = load(path)
        out = run_dir / "eval" / name
        if name == ZERO_SHOT:
            prompt = tts.create_voice_clone_prompt(
                ref_audio=str(ref_audio), ref_text=run["ref"]["text"]
            )
            kw, voice = {"voice_clone_prompt": prompt}, {"reference": run["ref"]}
        else:
            kw, voice = {"speaker": SPEAKER}, {"speaker": SPEAKER}
        records, sr = synth(tts, items, out, args.seed, args.max_new_tokens, **kw)
        results = {
            "config": {"id": name, "model": str(path.relative_to(ROOT)), **voice},
            "params": {"seed": args.seed, "max_new_tokens": args.max_new_tokens},
            "run": args.run,
            "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "environment": env,
            "sample_rate": sr,
            "phrases": records,
        }
        (out / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        hits = sum(r["hit_limit"] for r in records)
        print(f"{out}: {len(records)} фраз, до лимита — {hits}", flush=True)
        del tts
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
