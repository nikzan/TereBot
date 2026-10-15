"""Чтение WAV в моно float32."""

from pathlib import Path

import numpy as np
import soundfile as sf


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr
