"""Проверка цели обучения без обучения: CE исходной и исправленной цели sft_12hz.py на Base.

    CUDA_VISIBLE_DEVICES=1 HF_HOME=data/hf .venv/bin/qwen-lora \\
        scripts/voice_train/check_objective.py pilot-2026-10-17 [--clips 16] [--batch-size 2]

Берёт первые N клипов из data/voice/<прогон>/train.codes.jsonl, собирает батчи тем же
TTSDataset.collate_fn из data/vendor/Qwen3-TTS/finetuning/, что и обучение, и в режиме no_grad
считает CE talker (код 0 и EOS) и sub-talker (коды 1–15) двумя способами:
- «исходная» — как в sft_12hz.py на 022e286: текст без text_projection, ручной сдвиг плюс сдвиг
  в loss (talker предсказывает кадр t+2), состояние sub-talker того же кадра, метки sub-talker
  сдвинуты дважды;
- «исправленная» — как после scripts/voice_train/patches/sft_12hz-alignment.patch.
Base умеет синтезировать свои коды, поэтому верная цель должна дать CE заметно ниже ln(2048).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "data" / "vendor" / "Qwen3-TTS" / "finetuning"
BASE_DIR = ROOT / "data" / "models" / "Qwen3-TTS-12Hz-1.7B-Base"
CODEBOOK = 2048


def ce_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    """Сумма CE и число учтённых меток (-100 пропускаются)."""
    logits = logits.reshape(-1, logits.shape[-1]).float()
    labels = labels.reshape(-1)
    n = int((labels != -100).sum())
    return float(F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")), n


@torch.no_grad()
def batch_losses(
    model, batch: dict, speaker: torch.Tensor | None = None,
    objectives: tuple[str, ...] = ("исходная", "исправленная"),
) -> dict[tuple[str, str], tuple[float, int]]:
    """Суммы CE по целям; speaker — готовый эмбеддинг говорящего вместо speaker_encoder
    (у чекпоинта custom_voice это строка его ID в codec_embedding)."""
    dev = model.device
    batch = {k: v.to(dev) for k, v in batch.items()}
    talker = model.talker
    codec_ids = batch["codec_ids"]
    codec_mask = batch["codec_mask"]
    labels = batch["codec_0_labels"]

    if speaker is None:
        speaker = model.speaker_encoder(batch["ref_mels"].to(model.dtype))
    text_ids = batch["input_ids"][:, :, 0]
    codec_in = batch["input_ids"][:, :, 1]
    codec_emb = talker.model.codec_embedding(codec_in) * batch["codec_embedding_mask"]
    codec_emb[:, 6, :] = speaker
    for i in range(1, 16):
        emb = talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        codec_emb = codec_emb + emb * codec_mask.unsqueeze(-1)
    text_raw = talker.model.text_embedding(text_ids)
    text_mask = batch["text_embedding_mask"]
    frames = codec_ids[codec_mask]

    out = {}
    for objective in objectives:
        if objective == "исходная":
            embeds = (text_raw * text_mask + codec_emb)[:, :-1]
            attn = batch["attention_mask"][:, :-1]
            # sft_12hz.py передаёт labels[:, 1:], а ForCausalLMLoss сдвигает ещё раз.
            talker_labels = labels[:, 2:]
        else:
            embeds = talker.text_projection(text_raw) * text_mask + codec_emb
            attn = batch["attention_mask"]
            talker_labels = labels[:, 1:]
        o = talker(inputs_embeds=embeds, attention_mask=attn, output_hidden_states=True)
        logits = o.logits[:, : talker_labels.shape[1]]
        out[(objective, "talker")] = ce_sum(logits, talker_labels)

        hidden = o.hidden_states[0][-1]
        if objective == "исходная":
            states = hidden[codec_mask[:, :-1]]
        else:
            states = hidden[:, :-1][codec_mask[:, 1:]]
        sub_logits, _ = talker.forward_sub_talker_finetune(frames, states)
        if objective == "исходная":
            # forward_finetune передаёт codec_ids[:, 1:] в ForCausalLMLoss, тот сдвигает снова:
            # голова j учится на код j+2, последний код не учится.
            out[(objective, "sub-talker")] = ce_sum(sub_logits[:, :-1], frames[:, 2:])
        else:
            out[(objective, "sub-talker")] = ce_sum(sub_logits, frames[:, 1:])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", help="имя прогона: каталог data/voice/<прогон>/ с train.codes.jsonl")
    ap.add_argument("--clips", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--model", type=Path, default=BASE_DIR)
    args = ap.parse_args()

    codes = ROOT / "data" / "voice" / args.run / "train.codes.jsonl"
    if not codes.is_file():
        raise SystemExit(f"нет {codes} — сначала train.sh этого прогона")
    rows = [json.loads(line) for line in codes.read_text(encoding="utf-8").splitlines()]
    rows = rows[: args.clips]

    sys.path.insert(0, str(VENDOR))
    from dataset import TTSDataset
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from transformers import AutoConfig

    tts = Qwen3TTSModel.from_pretrained(
        str(args.model), dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0"
    )
    model = tts.model.eval()
    dataset = TTSDataset(rows, tts.processor, AutoConfig.from_pretrained(str(args.model)))

    totals: dict[tuple[str, str], list[float]] = {}
    for start in range(0, len(rows), args.batch_size):
        batch = dataset.collate_fn([dataset[i] for i in range(start, min(start + args.batch_size, len(rows)))])
        for key, (s, n) in batch_losses(model, batch).items():
            acc = totals.setdefault(key, [0.0, 0])
            acc[0] += s
            acc[1] += n

    rnd = math.log(CODEBOOK)
    print(f"\nПрогон {args.run}, клипов {len(rows)}, модель {args.model.name}")
    print(f"{'цель':<14}{'talker':>10}{'sub-talker':>12}")
    for objective in ("исходная", "исправленная"):
        t = totals[(objective, "talker")]
        s = totals[(objective, "sub-talker")]
        print(f"{objective:<14}{t[0] / t[1]:>10.3f}{s[0] / s[1]:>12.3f}")
    print(f"{'ln(2048)':<14}{rnd:>10.3f}{rnd:>12.3f}")

    fixed_t = totals[("исправленная", "talker")]
    fixed_s = totals[("исправленная", "sub-talker")]
    orig_t = totals[("исходная", "talker")]
    orig_s = totals[("исходная", "sub-talker")]
    ok = (
        fixed_t[0] / fixed_t[1] < min(rnd, orig_t[0] / orig_t[1])
        and fixed_s[0] / fixed_s[1] < min(rnd, orig_s[0] / orig_s[1])
        and orig_t[0] / orig_t[1] >= rnd
    )
    if ok:
        print("\nисправленная цель ниже случайной и исходной; исходная по talker не лучше случайной —"
              " без исправления не обучать")
    else:
        print("\nкритерий не выполнен: разбор ошибки цели неверен, обучать нельзя до разбора")
        sys.exit(1)


if __name__ == "__main__":
    main()
