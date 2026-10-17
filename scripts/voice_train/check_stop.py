"""Проверка остановки: заканчивает ли каждый чекпоинт прогона фразу сам, до лимита генерации.

    CUDA_VISIBLE_DEVICES=1 HF_HOME=data/hf .venv/bin/qwen-lora \\
        scripts/voice_train/check_stop.py pilot-2026-10-18 [--epochs 0 4 9]

Синтезирует фразы config/eval/stop_phrases_ru.txt на Base (zero-shot `generate_voice_clone`
с референсом и его текстом из run.json) и на каждом data/voice/<прогон>/checkpoint-epoch-N
(`generate_custom_voice`, говорящий prototype). Лимит `max_new_tokens=400` (~32 с), сид 0 перед
каждой фразой. Чекпоинт пригоден, если на каждой фразе синтез остановился сам и длительность —
0,5–2× от Base. WAV и summary.json — в data/voice/<прогон>/stop_check/.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "data" / "models" / "Qwen3-TTS-12Hz-1.7B-Base"
PHRASES = ROOT / "config" / "eval" / "stop_phrases_ru.txt"
FRAME_RATE = 12.5  # кадров кодека в секунду
SPEAKER = "prototype"
RATIO = (0.5, 2.0)


def read_phrases(path: Path) -> list[str]:
    lines = (s.strip() for s in path.read_text(encoding="utf-8").splitlines())
    return [s for s in lines if s and not s.startswith("#")]


def synth(tts, phrases: list[str], out_dir: Path, max_new_tokens: int, **kw) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, text in enumerate(phrases):
        torch.manual_seed(0)
        if "speaker" in kw:
            wavs, sr = tts.generate_custom_voice(
                text, language="Russian", max_new_tokens=max_new_tokens, **kw
            )
        else:
            wavs, sr = tts.generate_voice_clone(
                text, language="Russian", max_new_tokens=max_new_tokens, **kw
            )
        sf.write(out_dir / f"{i}.wav", wavs[0], sr)
        seconds = len(wavs[0]) / sr
        # На лимите кодек отдаёт max_new_tokens - 1 кадр (последний токен — не аудио).
        rows.append({
            "phrase": text,
            "seconds": round(seconds, 2),
            "hit_limit": seconds * FRAME_RATE >= max_new_tokens - 1 - 0.5,
        })
        print(f"  {i}: {seconds:6.2f} с{'  ДО ЛИМИТА' if rows[-1]['hit_limit'] else ''}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", help="имя прогона: каталог data/voice/<прогон>/")
    ap.add_argument("--epochs", type=int, nargs="*", help="только эти эпохи (по умолчанию все)")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--phrases", type=Path, default=PHRASES)
    args = ap.parse_args()

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    run_dir = ROOT / "data" / "voice" / args.run
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    ref_audio = run_dir / "audio" / f"{run['ref']['seg_id']}.wav"
    phrases = read_phrases(args.phrases)
    out = run_dir / "stop_check"

    ckpts = sorted(
        (int(m.group(1)), p)
        for p in run_dir.glob("checkpoint-epoch-*")
        if (m := re.fullmatch(r"checkpoint-epoch-(\d+)", p.name))
    )
    if args.epochs is not None:
        ckpts = [(n, p) for n, p in ckpts if n in args.epochs]
    if not ckpts:
        raise SystemExit(f"нет чекпоинтов в {run_dir}")

    def load(path: Path):
        return Qwen3TTSModel.from_pretrained(
            str(path), dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0"
        )

    results: dict[str, list[dict]] = {}
    print("base (zero-shot)", flush=True)
    tts = load(BASE_DIR)
    prompt = tts.create_voice_clone_prompt(ref_audio=str(ref_audio), ref_text=run["ref"]["text"])
    results["base"] = synth(tts, phrases, out / "base", args.max_new_tokens, voice_clone_prompt=prompt)
    del tts
    torch.cuda.empty_cache()

    for n, path in ckpts:
        name = f"epoch-{n}"
        print(name, flush=True)
        tts = load(path)
        results[name] = synth(tts, phrases, out / name, args.max_new_tokens, speaker=SPEAKER)
        del tts
        torch.cuda.empty_cache()

    base = results["base"]
    verdicts = {}
    print(f"\nПрогон {args.run}, лимит {args.max_new_tokens} токенов (~{args.max_new_tokens / FRAME_RATE:.0f} с)")
    head = "".join(f"{f'ф{i}':>10}" for i in range(len(phrases)))
    print(f"{'модель':<10}{head}  итог")
    for name, rows in results.items():
        cells, bad = [], []
        for i, r in enumerate(rows):
            mark = "*" if r["hit_limit"] else ""
            cells.append(f"{r['seconds']:>9.2f}{mark or ' '}")
            ratio = r["seconds"] / base[i]["seconds"] if base[i]["seconds"] else float("inf")
            if r["hit_limit"] or (name != "base" and not RATIO[0] <= ratio <= RATIO[1]):
                bad.append(i)
        ok = not bad
        verdicts[name] = {"ok": ok, "bad_phrases": [phrases[i] for i in bad]}
        verdict = "пригоден" if ok else "НЕПРИГОДЕН: ф" + ", ф".join(map(str, bad))
        if name == "base":
            verdict = "останавливается" if ok else "НЕ останавливается: ф" + ", ф".join(map(str, bad))
        print(f"{name:<10}{''.join(cells)}  {verdict}")
    print("\nдлительность в секундах; * — синтез дошёл до лимита")
    for i, p in enumerate(phrases):
        print(f"  ф{i}: {p}")
    for name, v in verdicts.items():
        if not v["ok"] and name != "base":
            print(f"{name}: не закончил или вне 0,5–2× от Base — {'; '.join(v['bad_phrases'])}")

    (out / "summary.json").write_text(
        json.dumps(
            {"max_new_tokens": args.max_new_tokens, "phrases": phrases, "results": results,
             "verdicts": verdicts},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    sys.exit(0 if all(v["ok"] for v in verdicts.values()) else 1)


if __name__ == "__main__":
    main()
