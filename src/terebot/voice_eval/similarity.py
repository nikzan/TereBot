"""Похожесть спикера: косинусное сходство эмбеддингов ECAPA-TDNN (SpeechBrain).

Синтез сравнивается с набором эталонных записей целевого говорящего; опорный уровень —
средняя попарная похожесть самих эталонов.
"""

import itertools
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from terebot.voice_eval.audio import read_mono

ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_SR = 16_000
MODELS_DIR = Path(__file__).resolve().parents[3] / "data" / "models"


@dataclass(frozen=True)
class Similarity:
    mean: float
    std: float


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def aggregate(values: list[float]) -> Similarity:
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return Similarity(mean=statistics.fmean(values), std=std)


def similarity_to_refs(emb: np.ndarray, refs: list[np.ndarray]) -> Similarity:
    return aggregate([cosine(emb, r) for r in refs])


def baseline(refs: list[np.ndarray]) -> Similarity:
    if len(refs) < 2:
        raise ValueError("для опорного уровня нужно минимум две эталонные записи")
    return aggregate([cosine(a, b) for a, b in itertools.combinations(refs, 2)])


class SpeakerEncoder:
    def __init__(self, model: str = ECAPA_MODEL):
        from speechbrain.inference.speaker import EncoderClassifier

        savedir = MODELS_DIR / model.replace("/", "--")
        self._model = EncoderClassifier.from_hparams(source=model, savedir=str(savedir))

    def embed(self, wav: Path) -> np.ndarray:
        import torch
        import torchaudio.functional as F

        audio, sr = read_mono(wav)
        signal = torch.from_numpy(audio)
        if sr != ECAPA_SR:
            signal = F.resample(signal, sr, ECAPA_SR)
        with torch.inference_mode():
            emb = self._model.encode_batch(signal.unsqueeze(0))
        return emb.squeeze().cpu().numpy()
