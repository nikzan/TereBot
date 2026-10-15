"""Разборчивость: CER распознанного текста относительно исходной фразы.

Распознавание — GigaAM-v3 e2e CTC через onnx-asr, та же модель, что будет в Клоне.
Перед сравнением оба текста нормализуются: регистр, пунктуация, ё → е.
"""

import re
from pathlib import Path

from terebot.voice_eval.audio import read_mono

ASR_MODEL = "gigaam-v3-e2e-ctc"
# onnx-asr по умолчанию берёт CoreML, и gigaam-v3-e2e-ctc на нём падает при распознавании
# (onnx-asr 0.12.0, 2026-10-15). Для метрик скорость не важна.
ASR_PROVIDERS = ["CPUExecutionProvider"]

# Всё, кроме букв, цифр и пробелов, — пунктуация. Дефис внутри слова («по-другому», «кто-нибудь»)
# тоже убираем: распознавание пишет такие слова то слитно, то через дефис.
_PUNCT = re.compile(r"[^\w\s]|_")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError("пустой исходный текст после нормализации")
    return edit_distance(ref, hyp) / len(ref)


class Transcriber:
    def __init__(self, model: str = ASR_MODEL):
        import onnx_asr

        self._asr = onnx_asr.load_model(model, providers=ASR_PROVIDERS)

    def transcribe(self, wav: Path) -> str:
        audio, sr = read_mono(wav)
        return self._asr.recognize(audio, sample_rate=sr)
