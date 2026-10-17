#!/usr/bin/env bash
# Дообучение Голоса на кластере: prepare_data.py → sft_12hz.py из QwenLM/Qwen3-TTS.
#
#   CUDA_VISIBLE_DEVICES=1 scripts/voice_train/train.sh <прогон> [--epochs N] [--limit N]
#                                                     [--batch-size N] [--lr X]
#
# <прогон> — каталог data/voice/<прогон>/ с train.jsonl и run.json от prepare.py.
# Чекпоинты — data/voice/<прогон>/checkpoint-epoch-N, лог — train.log, параметры дописываются
# в run.json. Запуск — из корня репозитория, порядок шагов — scripts/voice_train/README.md.
set -euo pipefail

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; }

EPOCHS=10
BATCH_SIZE=2
LR=2e-5
LIMIT=
RUN=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs) EPOCHS=$2; shift 2 ;;
    --batch-size) BATCH_SIZE=$2; shift 2 ;;
    --lr) LR=$2; shift 2 ;;
    --limit) LIMIT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "неизвестный флаг: $1" >&2; usage >&2; exit 2 ;;
    *) RUN=$1; shift ;;
  esac
done
[[ -n $RUN ]] || { usage >&2; exit 2; }

# Кластер общий: карту выбираем явно, чтобы не занять чужую.
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  echo "задайте CUDA_VISIBLE_DEVICES — номер свободной карты (nvidia-smi)" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
VENDOR=$ROOT/data/vendor/Qwen3-TTS
RUN_DIR=$ROOT/data/voice/$RUN
BASE_REPO=Qwen/Qwen3-TTS-12Hz-1.7B-Base
BASE_DIR=$ROOT/data/models/Qwen3-TTS-12Hz-1.7B-Base
TOKENIZER=Qwen/Qwen3-TTS-Tokenizer-12Hz
SPEAKER=prototype
# Кэш моделей — внутри репозитория, а не в общем ~/.cache.
export HF_HOME=${HF_HOME:-$ROOT/data/hf}

[[ -f $RUN_DIR/train.jsonl ]] || { echo "нет $RUN_DIR/train.jsonl — сначала prepare.py" >&2; exit 1; }
[[ -f $VENDOR/finetuning/sft_12hz.py ]] || { echo "нет $VENDOR — клонируйте QwenLM/Qwen3-TTS" >&2; exit 1; }

[[ -x $ROOT/.venv/bin/python ]] || { echo "нет .venv — сначала uv sync --group train" >&2; exit 1; }

# Интерпретатор venv под обезличенным именем: так процесс выглядит в nvidia-smi и htop.
# Симлинк лежит в .venv/bin рядом с pyvenv.cfg, поэтому venv находится как обычно.
PROC_NAME=${PROC_NAME:-qwen-lora}
ln -sf python "$ROOT/.venv/bin/$PROC_NAME"
PY=("$ROOT/.venv/bin/$PROC_NAME")

# sft_12hz.py копирует каталог init_model_path в каждый чекпоинт, поэтому модель нужна локально.
if [[ ! -f $BASE_DIR/model.safetensors ]]; then
  "$ROOT/.venv/bin/hf" download "$BASE_REPO" --local-dir "$BASE_DIR"
fi

RAW=$RUN_DIR/train.jsonl
if [[ -n $LIMIT ]]; then
  RAW=$RUN_DIR/train.limit.jsonl
  head -n "$LIMIT" "$RUN_DIR/train.jsonl" > "$RAW"
fi
CODES=$RUN_DIR/train.codes.jsonl

export RUN_DIR EPOCHS BATCH_SIZE LR LIMIT VENDOR BASE_REPO TOKENIZER SPEAKER
"${PY[@]}" - <<'EOF'
import json, os, subprocess
from datetime import datetime
from importlib.metadata import version

import torch
from huggingface_hub import HfApi

e = os.environ
path = os.path.join(e["RUN_DIR"], "run.json")
run = json.load(open(path, encoding="utf-8"))
run["train"] = {
    "started": datetime.now().isoformat(timespec="seconds"),
    "base_model": e["BASE_REPO"],
    "base_revision": HfApi().model_info(e["BASE_REPO"]).sha,
    "tokenizer": e["TOKENIZER"],
    "speaker_name": e["SPEAKER"],
    "epochs": int(e["EPOCHS"]),
    "batch_size": int(e["BATCH_SIZE"]),
    "lr": float(e["LR"]),
    "gradient_accumulation_steps": 4,
    "mixed_precision": "bf16",
    "limit": int(e["LIMIT"]) if e["LIMIT"] else None,
    "qwen3_tts_commit": subprocess.check_output(
        ["git", "-C", e["VENDOR"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "versions": {p: version(p) for p in (
        "qwen-tts", "transformers", "accelerate", "torch", "torchaudio", "kernels"
    )},
    "gpu": torch.cuda.get_device_name(0),
}
json.dump(run, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(run["train"], ensure_ascii=False, indent=2))
EOF

cd "$VENDOR/finetuning"
{
  "${PY[@]}" prepare_data.py \
    --device cuda:0 \
    --tokenizer_model_path "$TOKENIZER" \
    --input_jsonl "$RAW" \
    --output_jsonl "$CODES"

  "${PY[@]}" sft_12hz.py \
    --init_model_path "$BASE_DIR" \
    --output_model_path "$RUN_DIR" \
    --train_jsonl "$CODES" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_epochs "$EPOCHS" \
    --speaker_name "$SPEAKER"
} 2>&1 | tee "$RUN_DIR/train.log"

"${PY[@]}" - <<'EOF'
import json, os
from datetime import datetime

path = os.path.join(os.environ["RUN_DIR"], "run.json")
run = json.load(open(path, encoding="utf-8"))
run["train"]["finished"] = datetime.now().isoformat(timespec="seconds")
json.dump(run, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
EOF
echo "готово: $RUN_DIR/checkpoint-epoch-*"
