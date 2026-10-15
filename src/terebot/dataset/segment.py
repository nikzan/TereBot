"""Нарезка записей шлюза на сегменты 3–15 с.

Шлюз пишет в `data/recordings/<Сессия>/` файл `<user_id>.wav` (48 кГц, моно, 16 бит) без тишины
и рядом `<user_id>.segments.jsonl` — по строке на фрагмент речи: {start_ms, duration_ms,
sample_offset}. Берём только файлы Прототипа.

Весь WAV Сессии нормализуется к −23 LUFS без лимитера. Фрагменты — жёсткие границы: каждый
режется отдельно. Внутри фрагмента VAD находит речь; края обрезаются с отступом, а фрагмент длиннее
15 с рекурсивно делится по самой длинной паузе. Негодные сегменты не пишутся в WAV, а остаются
в `segments.jsonl` датасета с причиной в поле `reject`.
"""

import shutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf

from terebot.dataset.jsonl import read_jsonl, write_jsonl

SAMPLE_RATE = 48000
TARGET_LUFS = -23.0

# Интервалы речи [начало, конец) в сэмплах внутри переданного аудио.
Vad = Callable[[np.ndarray, int], list[tuple[int, int]]]

# Причины отбраковки при нарезке. Причины вычитки — в terebot.dataset.review.
SHORT = "короткий"
LONG = "длинный без пауз"
CLIPPED = "клиппинг"
QUIET = "тихий"
OVERLOAD = "перегруз после нормализации"
NO_SPEECH = "нет речи"


@dataclass(frozen=True)
class Params:
    min_s: float = 3.0
    max_s: float = 15.0
    # Отступ от края речи; берётся только внутри фрагмента.
    pad_ms: int = 120
    # Клиппинг: доля сэмплов с |x| ≥ clip_level в исходной записи больше clip_ratio.
    clip_level: float = 0.99
    clip_ratio: float = 0.001
    # Сегмент тише этого после нормализации Сессии отбраковывается.
    quiet_lufs: float = -40.0


DEFAULT_PARAMS = Params()


@dataclass(frozen=True)
class Fragment:
    start_ms: int
    duration_ms: int
    sample_offset: int


def seg_id(session: str, start_ms: int) -> str:
    return f"{session}_{start_ms:09d}"


def read_fragments(path: Path) -> list[Fragment]:
    return [Fragment(r["start_ms"], r["duration_ms"], r["sample_offset"]) for r in read_jsonl(path)]


def fragment_bounds(fragments: list[Fragment], total: int, sr: int) -> list[tuple[int, int]]:
    """Сэмплы [начало, конец) каждого фрагмента в WAV.

    Конец — начало следующего фрагмента, но не дальше `duration_ms`: если шлюз оборвался посреди
    фрагмента, его хвост без строки в segments.jsonl не приклеится к предыдущему.
    """
    bounds = []
    for i, f in enumerate(fragments):
        nxt = fragments[i + 1].sample_offset if i + 1 < len(fragments) else total
        end = min(nxt, f.sample_offset + round(f.duration_ms * sr / 1000), total)
        bounds.append((f.sample_offset, end))
    return bounds


def split_fragment(
    n: int, speech: list[tuple[int, int]], sr: int, params: Params
) -> list[tuple[int, int]]:
    """Границы сегментов [начало, конец) внутри фрагмента длиной n сэмплов по интервалам речи."""
    if not speech:
        return []
    pad = params.pad_ms * sr // 1000
    start = max(0, speech[0][0] - pad)
    end = min(n, speech[-1][1] + pad)
    pauses = [
        (a_end, b_start) for (_, a_end), (b_start, _) in zip(speech, speech[1:], strict=False)
    ]
    return _split(start, end, [p for p in pauses if p[1] > p[0]], sr, params, pad)


def _split(start, end, pauses, sr, params, pad) -> list[tuple[int, int]]:
    if end - start <= params.max_s * sr:
        return [(start, end)]
    inside = [p for p in pauses if start < p[0] and p[1] < end]
    if not inside:
        return [(start, end)]

    def cut(p):
        mid = (p[0] + p[1]) // 2
        return min(p[0] + pad, mid), max(p[1] - pad, mid)

    min_n = params.min_s * sr
    # Сначала паузы, после которых обе части не короче минимума, иначе — любые.
    viable = [p for p in inside if cut(p)[0] - start >= min_n and end - cut(p)[1] >= min_n]
    pause = max(viable or inside, key=lambda p: (p[1] - p[0], -p[0]))
    left_end, right_start = cut(pause)
    return _split(start, left_end, pauses, sr, params, pad) + _split(
        right_start, end, pauses, sr, params, pad
    )


def loudness(audio: np.ndarray, sr: int) -> float:
    with warnings.catch_warnings():
        # pyloudnorm предупреждает о тишине (−inf) — для нас это просто очень тихо.
        warnings.simplefilter("ignore")
        return pyloudnorm.Meter(sr).integrated_loudness(audio)


def normalize_loudness(audio: np.ndarray, sr: int, target: float = TARGET_LUFS) -> np.ndarray:
    """Усиление до целевой громкости без лимитера: пики выше 1 не срезаются."""
    current = loudness(audio, sr)
    if not np.isfinite(current):
        raise ValueError("запись беззвучна, нормализовать нечего")
    return (audio * 10 ** ((target - current) / 20)).astype(np.float32)


def reject_reason(raw: np.ndarray, norm: np.ndarray, sr: int, params: Params) -> str | None:
    seconds = len(raw) / sr
    if seconds < params.min_s:
        return SHORT
    if seconds > params.max_s:
        return LONG
    if np.mean(np.abs(raw) >= params.clip_level) > params.clip_ratio:
        return CLIPPED
    if loudness(norm, sr) < params.quiet_lufs:
        return QUIET
    if np.max(np.abs(norm)) > 1.0:
        return OVERLOAD
    return None


@dataclass
class Cut:
    record: dict
    # Нормализованное аудио годного сегмента; None у отбракованного.
    audio: np.ndarray | None


def segment_session(session_dir: Path, prototype_id: str, vad: Vad, params: Params) -> list[Cut]:
    session = session_dir.name
    raw, sr = sf.read(str(session_dir / f"{prototype_id}.wav"), dtype="float32")
    if raw.ndim != 1 or sr != SAMPLE_RATE:
        raise ValueError(f"{session_dir}: ожидался моно {SAMPLE_RATE} Гц от шлюза")
    norm = normalize_loudness(raw, sr)
    fragments = read_fragments(session_dir / f"{prototype_id}.segments.jsonl")

    cuts = []
    for frag, (f_start, f_end) in zip(
        fragments, fragment_bounds(fragments, len(raw), sr), strict=True
    ):
        frag_norm = norm[f_start:f_end]
        pieces = split_fragment(len(frag_norm), vad(frag_norm, sr), sr, params)
        if not pieces:
            cuts.append(Cut(_record(session, frag.start_ms, f_end - f_start, sr, NO_SPEECH), None))
            continue
        for a, b in pieces:
            start_ms = frag.start_ms + round(a * 1000 / sr)
            reason = reject_reason(raw[f_start + a : f_start + b], frag_norm[a:b], sr, params)
            record = _record(session, start_ms, b - a, sr, reason)
            cuts.append(Cut(record, None if reason else frag_norm[a:b]))
    return cuts


def _record(session: str, start_ms: int, samples: int, sr: int, reason: str | None) -> dict:
    sid = seg_id(session, start_ms)
    return {
        "seg_id": sid,
        "session": session,
        "start_ms": start_ms,
        "duration": round(samples / sr, 3),
        "audio": None if reason else f"segments/{sid}.wav",
        "reject": reason,
    }


def find_sessions(recordings_dir: Path, prototype_id: str) -> list[Path]:
    """Каталоги Сессий, где есть запись Прототипа; файлы других участников не читаются."""
    return sorted(
        d
        for d in recordings_dir.iterdir()
        if (d / f"{prototype_id}.wav").exists() and (d / f"{prototype_id}.segments.jsonl").exists()
    )


def segment_all(
    recordings_dir: Path,
    dataset_dir: Path,
    prototype_id: str,
    vad: Vad,
    params: Params = DEFAULT_PARAMS,
    force: bool = False,
) -> list[str]:
    """Нарезать новые Сессии (с force — все) и обновить segments.jsonl. Возвращает нарезанные."""
    index_path = dataset_dir / "segments.jsonl"
    seg_dir = dataset_dir / "segments"
    rows = read_jsonl(index_path)
    done = {r["session"] for r in rows}
    todo = [d for d in find_sessions(recordings_dir, prototype_id) if force or d.name not in done]

    tmp_dir = dataset_dir / ".segments.tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    new_rows = []
    for d in todo:
        for cut in segment_session(d, prototype_id, vad, params):
            new_rows.append(cut.record)
            if cut.audio is not None:
                sf.write(tmp_dir / f"{cut.record['seg_id']}.wav", cut.audio, SAMPLE_RATE, "PCM_16")

    # WAV старой нарезки пересобранных Сессий удаляются, новые переносятся на их место.
    redone = {d.name for d in todo}
    seg_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r["session"] in redone and r["audio"]:
            (dataset_dir / r["audio"]).unlink(missing_ok=True)
    for wav in tmp_dir.iterdir():
        wav.replace(seg_dir / wav.name)
    tmp_dir.rmdir()

    kept = [r for r in rows if r["session"] not in redone]
    write_jsonl(index_path, sorted(kept + new_rows, key=lambda r: r["seg_id"]))
    return sorted(redone)


class SileroVad:
    """Silero VAD с порогами для разговорной речи; паузы короче min_silence_ms не делят речь."""

    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 100, min_speech_ms: int = 100):
        from silero_vad import load_silero_vad

        with warnings.catch_warnings():
            # silero-vad 6.2.3 грузит модель через устаревший torch.jit.load.
            warnings.simplefilter("ignore", FutureWarning)
            self._model = load_silero_vad()
        self._kw = dict(
            threshold=threshold,
            min_silence_duration_ms=min_silence_ms,
            min_speech_duration_ms=min_speech_ms,
            speech_pad_ms=0,
        )

    def __call__(self, audio: np.ndarray, sr: int) -> list[tuple[int, int]]:
        import torch
        from silero_vad import get_speech_timestamps

        with warnings.catch_warnings():
            # 48 кГц silero сам прореживает до 16 кГц и предупреждает об этом.
            warnings.simplefilter("ignore")
            ts = get_speech_timestamps(
                torch.from_numpy(audio), self._model, sampling_rate=sr, **self._kw
            )
        return [(t["start"], t["end"]) for t in ts]
