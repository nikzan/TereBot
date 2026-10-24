import pytest

from terebot.voice_train.curve import ZERO_SHOT, choose, table


def results(sim, cers):
    phrases = [{"category": "holdout", "cer": c} for c in cers]
    return {
        "phrases": phrases,
        "quality": {
            "similarity_median": sim,
            "cer_median": sorted(cers)[len(cers) // 2],
            "cer_median_by_category": {"holdout": sorted(cers)[len(cers) // 2]},
        },
    }


def run(**epochs):
    """epochs: имя → (похожесть, CER фраз)."""
    return {name.replace("_", "-"): results(*v) for name, v in epochs.items()}


GOOD = [0.0, 0.02, 0.05]


def test_table_sorted_by_epoch_zero_shot_first():
    rows = table(
        run(
            checkpoint_epoch_10=(0.6, GOOD),
            zero_shot=(0.5, GOOD),
            checkpoint_epoch_2=(0.55, GOOD),
            checkpoint_epoch_9=(0.6, GOOD),
        )
    )
    assert [r.name for r in rows] == [
        ZERO_SHOT,
        "checkpoint-epoch-2",
        "checkpoint-epoch-9",
        "checkpoint-epoch-10",
    ]
    assert rows[1].cer_mean == pytest.approx(sum(GOOD) / 3)


def test_rule_picks_earliest_plateau_epoch():
    rows = table(
        run(
            zero_shot=(0.48, GOOD),
            checkpoint_epoch_0=(0.52, GOOD),
            checkpoint_epoch_1=(0.57, GOOD),
            checkpoint_epoch_2=(0.60, GOOD),
            checkpoint_epoch_3=(0.605, GOOD),
            checkpoint_epoch_4=(0.61, GOOD),
        )
    )
    assert choose(rows).epoch == 2


def test_rule_drops_epoch_with_grown_cer():
    rows = table(
        run(
            zero_shot=(0.48, GOOD),
            checkpoint_epoch_1=(0.55, GOOD),
            checkpoint_epoch_2=(0.60, GOOD),
            # Похожесть ещё растёт, но разборчивость упала.
            checkpoint_epoch_3=(0.65, [0.05, 0.10, 0.20]),
            checkpoint_epoch_4=(0.655, GOOD),
        )
    )
    choice = choose(rows)
    assert all("эпоха 3" not in r or "CER" in r for r in choice.reasons)
    assert any(r.startswith("эпоха 3: медианный CER") for r in choice.reasons)
    # Без эпохи 3: от 2 к 4 похожесть растёт на 0,0275 за эпоху, плато — на 4 (последняя).
    assert choice.epoch == 4


def test_mean_cer_slack():
    rows = table(
        run(
            zero_shot=(0.48, [0.0, 0.0, 0.0]),
            # Медиана та же, но средний CER выше zero-shot на 0,1.
            checkpoint_epoch_1=(0.60, [0.0, 0.0, 0.3]),
            checkpoint_epoch_2=(0.55, [0.0, 0.0, 0.03]),
        )
    )
    assert choose(rows).epoch == 2


def test_no_allowed_checkpoint():
    rows = table(run(zero_shot=(0.48, GOOD), checkpoint_epoch_1=(0.6, [0.2, 0.3, 0.4])))
    assert choose(rows).epoch is None


def test_unknown_eval_dir_name():
    with pytest.raises(ValueError, match="checkpoint-epoch-N"):
        table(run(final=(0.5, GOOD)))
