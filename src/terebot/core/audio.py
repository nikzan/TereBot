"""Преобразования звука ядра: PCM s16 ↔ float32 и ресемплинг."""

import numpy as np
from scipy.signal import resample_poly


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768


def float_to_pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def downsample_48k_to_16k(audio: np.ndarray) -> np.ndarray:
    return resample_poly(audio, 1, 3).astype(np.float32)


def upsample_24k_to_48k(audio: np.ndarray) -> np.ndarray:
    return resample_poly(audio, 2, 1).astype(np.float32)


class Downsampler48to16:
    """Потоковое 48 → 16 кГц: ФНЧ с сохранением состояния между кадрами и каждый третий сэмпл.

    Ресемплинг каждого 20-мс кадра по отдельности давал бы щелчки на стыках.
    """

    def __init__(self, taps: int = 63):
        from scipy.signal import firwin

        self._fir = firwin(taps, 7_500, fs=48_000).astype(np.float32)
        self._zi = np.zeros(taps - 1, dtype=np.float32)
        self._phase = 0

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter

        filtered, self._zi = lfilter(self._fir, 1.0, audio, zi=self._zi)
        out = filtered[self._phase :: 3]
        self._phase = (self._phase - len(audio)) % 3
        return out.astype(np.float32)


class Upsampler24to48:
    """Потоковое 24 → 48 кГц: нули между сэмплами и ФНЧ с состоянием между кусками синтеза."""

    def __init__(self, taps: int = 63):
        from scipy.signal import firwin

        # ×2: после вставки нулей энергия сигнала вдвое меньше.
        self._fir = (firwin(taps, 11_000, fs=48_000) * 2).astype(np.float32)
        self._zi = np.zeros(taps - 1, dtype=np.float32)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter

        stuffed = np.zeros(len(audio) * 2, dtype=np.float32)
        stuffed[::2] = audio
        out, self._zi = lfilter(self._fir, 1.0, stuffed, zi=self._zi)
        return out.astype(np.float32)
