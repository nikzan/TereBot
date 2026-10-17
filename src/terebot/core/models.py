"""Обёртки моделей восприятия речи: Silero VAD, Smart Turn v3.2, GigaAM.

Версии — docs/REFERENCES.md.
"""

import urllib.request
from pathlib import Path

import numpy as np

from terebot.config import REPO_ROOT

SAMPLE_RATE = 16_000
MODELS_DIR = REPO_ROOT / "data" / "models"

SILERO_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/"
    "src/silero_vad/data/silero_vad.onnx"
)
SILERO_FILE = MODELS_DIR / "silero_vad-v6.2.3.onnx"
SILERO_BYTES = 2_327_524

SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_FILE = "smart-turn-v3.2-cpu.onnx"
SMART_TURN_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
SMART_TURN_SECONDS = 8


def _session(path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"], sess_options=opts)


def silero_path() -> Path:
    if not SILERO_FILE.exists():
        SILERO_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SILERO_FILE.with_suffix(".part")
        urllib.request.urlretrieve(SILERO_URL, tmp)
        if tmp.stat().st_size != SILERO_BYTES:
            raise RuntimeError(f"{SILERO_URL}: неожиданный размер {tmp.stat().st_size} байт")
        tmp.rename(SILERO_FILE)
    return SILERO_FILE


class SileroVad:
    """Одна сессия ONNX на всех; состояние — у каждого участника своё (`SileroVad.Stream`)."""

    WINDOW = 512
    CONTEXT = 64

    def __init__(self):
        self._session = _session(silero_path())

    def stream(self) -> "SileroStream":
        return SileroStream(self._session)


class SileroStream:
    def __init__(self, session):
        self._session = session
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, SileroVad.CONTEXT), dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def __call__(self, window: np.ndarray) -> float:
        """Вероятность речи в окне из 512 сэмплов float32 16 кГц."""
        x = np.concatenate([self._context, window.reshape(1, -1).astype(np.float32)], axis=1)
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[:, -SileroVad.CONTEXT :]
        return float(out[0][0])


class SmartTurn:
    """Smart Turn v3.2: вероятность, что участник договорил, по последним 8 с Реплики."""

    def __init__(self):
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        path = hf_hub_download(SMART_TURN_REPO, SMART_TURN_FILE, revision=SMART_TURN_REVISION)
        self._session = _session(Path(path))
        self._features = WhisperFeatureExtractor(chunk_length=SMART_TURN_SECONDS)

    def __call__(self, audio: np.ndarray) -> float:
        """audio — float32 16 кГц; как inference.py upstream: последние 8 с, тишина слева."""
        n = SMART_TURN_SECONDS * SAMPLE_RATE
        audio = audio[-n:]
        if len(audio) < n:
            audio = np.pad(audio, (n - len(audio), 0))
        inputs = self._features(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=n,
            truncation=True,
            do_normalize=True,
        )
        features = inputs.input_features.astype(np.float32)
        return float(self._session.run(None, {"input_features": features})[0][0].item())


ASR_MODEL = "gigaam-v3-e2e-ctc"


class Asr:
    """GigaAM-v3 e2e-ctc через onnx-asr, только на CPU: на CoreML модель падает."""

    def __init__(self, model: str = ASR_MODEL):
        import onnx_asr

        self._asr = onnx_asr.load_model(model, providers=["CPUExecutionProvider"])

    def __call__(self, audio: np.ndarray) -> str:
        return self._asr.recognize(audio, sample_rate=SAMPLE_RATE).strip()
