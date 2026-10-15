"""Автоматическая транскрипция годных сегментов в `review.jsonl`.

Пишет только `auto_text`; `text`, `status` и `reason` — ручная работа вычитки, их не трогает.
Сегменты с уже готовым `auto_text` пропускаются, с force распознаются заново.
"""

from collections.abc import Callable
from pathlib import Path

from terebot.dataset.jsonl import read_jsonl
from terebot.dataset.review import Review, new_entry

# e2e — с пунктуацией и регистром; RNNT точнее CTC, а скорость офлайн не важна.
ASR_MODEL = "gigaam-v3-e2e-rnnt"
# onnx-asr по умолчанию берёт CoreML, и gigaam-v3-e2e на нём падает (onnx-asr 0.12.0, 2026-10-15).
ASR_PROVIDERS = ["CPUExecutionProvider"]

Transcriber = Callable[[Path], str]


def transcribe_all(
    dataset_dir: Path, transcribe: Transcriber, force: bool = False, save_every: int = 20
) -> int:
    """Распознать сегменты без auto_text (с force — все). Возвращает число распознанных."""
    review = Review(dataset_dir / "review.jsonl", read_jsonl(dataset_dir / "segments.jsonl"))
    segments = {s["seg_id"]: s for s in read_jsonl(dataset_dir / "segments.jsonl")}
    done = 0
    for sid in review.queue:
        entry = review.entries.setdefault(sid, new_entry(sid))
        if entry["auto_text"] is not None and not force:
            continue
        entry["auto_text"] = transcribe(dataset_dir / segments[sid]["audio"])
        done += 1
        if done % save_every == 0:
            review.save()
    review.save()
    return done


class GigaAM:
    def __init__(self, model: str = ASR_MODEL):
        import onnx_asr

        self._asr = onnx_asr.load_model(model, providers=ASR_PROVIDERS)

    def __call__(self, wav: Path) -> str:
        import soundfile as sf

        audio, sr = sf.read(str(wav), dtype="float32")
        return self._asr.recognize(audio, sample_rate=sr)
