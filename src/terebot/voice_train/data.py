"""Манифест датасета Голоса → обучающий набор для дообучения Qwen3-TTS."""

from pathlib import Path


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
