"""Манифест датасета Голоса → обучающий набор для дообучения Qwen3-TTS."""

import shutil
from pathlib import Path

import numpy as np

from terebot.dataset.jsonl import write_jsonl


def pick_reference(train: list[dict], holdout: list[dict], ref: str | None = None) -> dict:
    """Референс Прототипа — сегмент обучающей выборки.

    По умолчанию — самый длинный, при равенстве — первый по `seg_id`. Сегмент из отложенной
    выборки отвергается: он нужен как эталон похожести и не должен влиять на обучение.
    """
    if ref is None:
        if not train:
            raise ValueError("обучающая выборка пуста")
        return min(train, key=lambda r: (-r["duration"], r["seg_id"]))
    if any(r["seg_id"] == ref for r in holdout):
        raise ValueError(
            f"{ref} — из отложенной выборки: она эталон похожести при оценке, "
            "референсом выберите сегмент обучающей"
        )
    for r in train:
        if r["seg_id"] == ref:
            return r
    raise ValueError(f"{ref} нет в обучающей выборке")


def training_rows(train: list[dict], holdout: list[dict], audio_dir: Path, ref: dict) -> list[dict]:
    """Строки `{audio, text, ref_audio}` по сегментам train, по порядку `seg_id`.

    `audio_dir` — каталог ресемплированных WAV `<seg_id>.wav`; пути абсолютные, как ждёт
    `prepare_data.py`. Пересечение выборок — ошибка манифеста, а не повод молча отфильтровать.
    """
    leaked = {r["seg_id"] for r in train} & {r["seg_id"] for r in holdout}
    if leaked:
        raise ValueError(f"сегменты в обеих выборках: {', '.join(sorted(leaked))}")
    audio_dir = audio_dir.resolve()
    ref_audio = str(audio_dir / f"{ref['seg_id']}.wav")
    return [
        {"audio": str(audio_dir / f"{r['seg_id']}.wav"), "text": r["text"], "ref_audio": ref_audio}
        for r in sorted(train, key=lambda r: r["seg_id"])
    ]


def pad_silence(audio: np.ndarray, sr: int, start_ms: int, end_ms: int) -> np.ndarray:
    """Цифровая тишина по краям сегмента.

    Шлюз режет фрагменты по активности микрофона, и речь в сегментах начинается почти с края.
    Без отступов Голос учится начинать сразу со звука и обрывать затухание — в синтезе съедаются
    первые и последние звуки фраз.
    """
    return np.pad(audio, (start_ms * sr // 1000, end_ms * sr // 1000))


def freeze_holdout(holdout: list[dict], dataset_dir: Path, run_dir: Path) -> list[dict]:
    """Копия отложенной выборки прогона: `holdout/<seg_id>.wav` (исходные, без отступов)
    и `holdout.jsonl` с путями относительно каталога прогона.

    Отложенная выборка меняется вместе с датасетом, а прогон оценивается на своей: по этим
    текстам синтезирует `synth_eval.py`, эти WAV — эталоны `--refs` и живые записи Слепого теста.
    """
    rows = []
    for r in sorted(holdout, key=lambda r: r["seg_id"]):
        dst = run_dir / "holdout" / f"{r['seg_id']}.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset_dir / r["audio"], dst)
        rows.append({**r, "audio": f"holdout/{r['seg_id']}.wav"})
    write_jsonl(run_dir / "holdout.jsonl", rows)
    return rows
