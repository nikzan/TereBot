"""Голос Клона на MLX: Qwen3-TTS через mlx-audio со стримингом.

Общий для ядра и бенчмарка (scripts/bench_voice.py), чтобы они синтезировали одинаково.
Найдено на этапе 3 (docs/benchmarks/voice-zero-shot.md): лимит кэша MLX, прогрев
со стримингом и `mx.clear_cache()` после него держат пик в ~4,4 ГБ.
"""

import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np

GB = 1024**3
WARMUP_TEXT = "Да."


class MlxVoice:
    """Zero-shot по референсу (`prepare_voice`) или дообученный по говорящему (`use_speaker`)."""

    streaming = True
    packages = ("mlx-audio", "mlx", "mlx-metal", "transformers")
    language = "russian"

    def __init__(self, streaming_interval: float, cache_limit_gb: float | None):
        self._interval = streaming_interval
        self._cache_limit_gb = cache_limit_gb
        self._mx = None
        self._ref_audio = None
        self._ref_text: str | None = None
        self._speaker: str | None = None
        self.sample_rate = 0

    def load(self, model: str, precision: str | None = None) -> None:
        import mlx.core as mx
        from mlx_audio.tts.utils import load

        if self._cache_limit_gb is not None:
            mx.set_cache_limit(int(self._cache_limit_gb * GB))
        self._model = load(model)
        self._mx = mx
        self.sample_rate = self._model.sample_rate

    def prepare_voice(self, ref_wav: Path, ref_text: str) -> None:
        from mlx_audio.utils import load_audio

        self._ref_audio = load_audio(str(ref_wav), sample_rate=self.sample_rate)
        self._ref_text = ref_text
        # Коды референса mlx-audio кэширует по (ref_text, ref_audio): считаем их при прогреве.
        self._warmup()

    def use_speaker(self, speaker: str) -> None:
        self._speaker = speaker
        self._warmup()

    def _warmup(self) -> None:
        # Прогрев — со стримингом, как синтез фраз: без него mlx-audio декодирует коды референса
        # вместе с новыми одним проходом, и это +4 ГБ к пику.
        for _ in self._generate(WARMUP_TEXT):
            pass
        # Буферы кодирования референса остаются в кэше MLX (~1,4 ГБ), пока их не освободить.
        self._mx.clear_cache()

    def _generate(self, text: str):
        if self._speaker is not None:
            voice = {"voice": self._speaker}
        else:
            voice = {"ref_audio": self._ref_audio, "ref_text": self._ref_text}
        return self._model.generate(
            text,
            **voice,
            lang_code=self.language,
            stream=True,
            streaming_interval=self._interval,
        )

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        """Звук фразы кусками по мере синтеза: float32 моно, `sample_rate` Гц."""
        for r in self._generate(text):
            yield np.array(r.audio.astype(self._mx.float32))

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self._mx.random.seed(seed)

    def backend_memory(self) -> tuple[int, str] | None:
        if self._mx is None:
            return None
        # Кэш буферов MLX — тоже память процесса, а get_peak_memory считает только активную.
        return self._mx.get_active_memory() + self._mx.get_cache_memory(), "mlx.active+cache"

    def backend_peak_memory(self) -> int | None:
        return self._mx.get_peak_memory() if self._mx else None
