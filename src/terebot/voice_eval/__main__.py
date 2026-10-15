import argparse
from pathlib import Path

from terebot.voice_eval.batch import REFERENCE_DIR, evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Метрики качества для каталогов прогона бенчмарка")
    parser.add_argument("runs", nargs="+", type=Path, help="каталоги data/bench/<ts>-<id>")
    parser.add_argument("--reference", type=Path, default=REFERENCE_DIR)
    args = parser.parse_args()
    for run in args.runs:
        q = evaluate_run(run, args.reference)["quality"]
        base = q["baseline_similarity"]
        base_s = f"{base:.3f}" if base is not None else "—"
        print(
            f"{run.name}: похожесть {q['similarity_median']:.3f} (опора {base_s}), "
            f"CER {q['cer_median']:.3f}"
        )


if __name__ == "__main__":
    main()
