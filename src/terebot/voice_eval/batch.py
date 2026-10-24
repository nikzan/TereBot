"""Пакетный подсчёт метрик для каталога прогона бенчмарка Голоса.

Читает `<каталог>/results.json`, для каждой фразы считает похожесть на эталоны и CER
и дописывает их в тот же файл вместе с медианами по конфигурации. Эталоны по умолчанию —
`sim_*.wav` бенчмарка; для дообученного Голоса — записи Прототипа из отложенной выборки.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

from terebot.voice_eval.cer import ASR_MODEL, Transcriber, cer
from terebot.voice_eval.similarity import ECAPA_MODEL, SpeakerEncoder, baseline, similarity_to_refs

REFERENCE_DIR = Path(__file__).resolve().parents[3] / "data" / "bench" / "reference"
# Эталоны для похожести; ref.wav — промпт zero-shot, с ним не сравниваем.
REFS_GLOB = "sim_*.wav"


def reference_wavs(refs: Path) -> list[Path]:
    """Эталоны: каталог WAV или манифест датасета (`holdout.jsonl`).

    Пути в манифесте — относительно корня датасета, то есть на уровень выше `manifest/`.
    """
    if refs.is_dir():
        wavs = sorted(refs.glob("*.wav"))
    else:
        root = refs.resolve().parent.parent
        lines = refs.read_text(encoding="utf-8").splitlines()
        wavs = [root / json.loads(line)["audio"] for line in lines if line.strip()]
    if not wavs:
        raise FileNotFoundError(f"{refs}: нет эталонных записей")
    return wavs


def summarize(phrases: list[dict]) -> dict:
    by_category = defaultdict(list)
    for p in phrases:
        by_category[p["category"]].append(p["cer"])
    return {
        "similarity_median": statistics.median(p["similarity"] for p in phrases),
        "cer_median": statistics.median(p["cer"] for p in phrases),
        "cer_median_by_category": {c: statistics.median(v) for c, v in sorted(by_category.items())},
    }


def evaluate_run(
    run_dir: Path, reference_dir: Path = REFERENCE_DIR, refs: Path | None = None
) -> dict:
    """`refs` — каталог WAV или `holdout.jsonl`; без него — `sim_*.wav` из `reference_dir`."""
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if refs is not None:
        ref_wavs = reference_wavs(refs)
    else:
        ref_wavs = sorted(reference_dir.glob(REFS_GLOB))
        if not ref_wavs:
            raise FileNotFoundError(f"{reference_dir}: нет эталонных записей {REFS_GLOB}")

    encoder = SpeakerEncoder()
    transcriber = Transcriber()
    ref_embs = [encoder.embed(w) for w in ref_wavs]
    base = baseline(ref_embs) if len(ref_embs) > 1 else None

    for p in results["phrases"]:
        wav = run_dir / p["wav"]
        sim = similarity_to_refs(encoder.embed(wav), ref_embs)
        p["similarity"] = sim.mean
        p["similarity_std"] = sim.std
        p["asr_text"] = transcriber.transcribe(wav)
        p["cer"] = cer(p["text"], p["asr_text"])

    results["quality"] = {
        **summarize(results["phrases"]),
        "baseline_similarity": base.mean if base else None,
        "baseline_similarity_std": base.std if base else None,
        "refs": [w.name for w in ref_wavs],
        **({"refs_source": str(refs)} if refs is not None else {}),
        "speaker_model": ECAPA_MODEL,
        "asr_model": ASR_MODEL,
    }
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
