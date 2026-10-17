"""Кривая переобучения: CE исправленной цели на отложенной выборке по чекпоинтам прогона.

    CUDA_VISIBLE_DEVICES=1 HF_HOME=$PWD/data/hf .venv/bin/qwen-lora \\
        scripts/voice_train/check_holdout.py pilot-2026-10-18 [--epochs 0 5 9]

Готовит коды отложенной выборки датасета тем же prepare_data.py из vendor, что и обучение
(holdout_audio/, holdout.jsonl, holdout.codes.jsonl в data/voice/<прогон>/; повторно не пересчитывает),
и в режиме no_grad считает CE talker и sub-talker на Base и на каждом checkpoint-epoch-N:
на holdout и на столько же первых клипах train.codes.jsonl. Цель — как в патче выравнивания,
loss — talker + 0,3 · sub-talker, как в sft_12hz.py. Минимум на holdout — эпоха, после которой
модель запоминает обучающие клипы, а не учится голосу. Итог — также holdout_ce.json.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_objective import BASE_DIR, ROOT, VENDOR, batch_losses  # noqa: E402
from prepare import resample  # noqa: E402

from terebot.dataset import DATASET_DIR  # noqa: E402
from terebot.dataset.jsonl import read_jsonl, write_jsonl  # noqa: E402

TOKENIZER = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
SPEAKER_ID = 3000  # строка говорящего в codec_embedding, её задаёт sft_12hz.py
SUB_WEIGHT = 0.3


def holdout_codes(run_dir: Path, ref_audio: str) -> Path:
    out = run_dir / "holdout.codes.jsonl"
    if out.is_file():
        return out
    audio_dir = run_dir / "holdout_audio"
    rows = []
    for r in read_jsonl(DATASET_DIR / "manifest" / "holdout.jsonl"):
        dst = audio_dir / f"{r['seg_id']}.wav"
        resample(DATASET_DIR / r["audio"], dst)
        rows.append({"audio": str(dst), "text": r["text"], "ref_audio": ref_audio})
    raw = run_dir / "holdout.jsonl"
    write_jsonl(raw, rows)
    subprocess.run(
        [sys.executable, "prepare_data.py", "--device", "cuda:0", "--tokenizer_model_path", TOKENIZER,
         "--input_jsonl", str(raw), "--output_jsonl", str(out)],
        cwd=VENDOR, check=True,
    )
    return out


def evaluate(model, dataset, speaker: torch.Tensor | None, batch_size: int) -> dict[str, float]:
    sums = {"talker": [0.0, 0], "sub-talker": [0.0, 0]}
    for start in range(0, len(dataset), batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        res = batch_losses(model, dataset.collate_fn(items), speaker, objectives=("исправленная",))
        for part in sums:
            s, n = res[("исправленная", part)]
            sums[part][0] += s
            sums[part][1] += n
    t = sums["talker"][0] / sums["talker"][1]
    s = sums["sub-talker"][0] / sums["sub-talker"][1]
    return {"talker": t, "sub-talker": s, "loss": t + SUB_WEIGHT * s}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", help="имя прогона: каталог data/voice/<прогон>/")
    ap.add_argument("--epochs", type=int, nargs="*", help="только эти эпохи (по умолчанию все)")
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    run_dir = ROOT / "data" / "voice" / args.run
    train_rows = read_jsonl(run_dir / "train.codes.jsonl")
    ref_audio = train_rows[0]["ref_audio"]
    hold_rows = read_jsonl(holdout_codes(run_dir, ref_audio))
    train_rows = train_rows[: len(hold_rows)]

    sys.path.insert(0, str(VENDOR))
    from dataset import TTSDataset
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from transformers import AutoConfig

    ckpts = sorted(
        (int(m.group(1)), p)
        for p in run_dir.glob("checkpoint-epoch-*")
        if (m := re.fullmatch(r"checkpoint-epoch-(\d+)", p.name))
    )
    if args.epochs is not None:
        ckpts = [(n, p) for n, p in ckpts if n in args.epochs]
    models = [("base", BASE_DIR)] + [(f"epoch-{n}", p) for n, p in ckpts]

    results = {}
    for name, path in models:
        tts = Qwen3TTSModel.from_pretrained(
            str(path), dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0"
        )
        model = tts.model.eval()
        config = AutoConfig.from_pretrained(str(path))
        speaker = None
        if name != "base":
            speaker = model.talker.model.codec_embedding.weight[SPEAKER_ID]
        results[name] = {
            split: evaluate(model, TTSDataset(rows, tts.processor, config), speaker, args.batch_size)
            for split, rows in (("holdout", hold_rows), ("train", train_rows))
        }
        h, t = results[name]["holdout"], results[name]["train"]
        print(f"{name}: holdout {h['loss']:.3f} (talker {h['talker']:.3f}, sub {h['sub-talker']:.3f}),"
              f" train {t['loss']:.3f}", flush=True)
        del tts, model
        torch.cuda.empty_cache()

    print(f"\nПрогон {args.run}: holdout {len(hold_rows)} клипов, train — первые {len(train_rows)}")
    print(f"{'модель':<10}{'h talker':>10}{'h sub':>8}{'h loss':>8}{'t talker':>10}{'t sub':>8}{'t loss':>8}")
    for name, r in results.items():
        h, t = r["holdout"], r["train"]
        print(f"{name:<10}{h['talker']:>10.3f}{h['sub-talker']:>8.3f}{h['loss']:>8.3f}"
              f"{t['talker']:>10.3f}{t['sub-talker']:>8.3f}{t['loss']:>8.3f}")
    epochs = {k: v for k, v in results.items() if k != "base"}
    if epochs:
        best = min(epochs, key=lambda k: epochs[k]["holdout"]["loss"])
        print(f"\nминимум loss на holdout: {best}")
    (run_dir / "holdout_ce.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
