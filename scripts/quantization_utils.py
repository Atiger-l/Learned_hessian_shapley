"""Shared helpers for post-training quantization experiments."""
from __future__ import annotations

import json
import os

import torch
from transformers import PreTrainedTokenizerBase

from evaluate import build_global_score_ranking

LETTERS = ["A", "B", "C", "D"]


def _log(msg: str) -> None:
    print(msg, flush=True)


def fake_quant_row(row: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Symmetric per-row fake quant → dequant (simulates INT8 storage error)."""
    if bits <= 0:
        return row
    qmax = float(2 ** (bits - 1) - 1)
    scale = row.abs().max().clamp(min=1e-8) / qmax
    q = (row / scale).round().clamp(-qmax - 1, qmax)
    return (q * scale).to(dtype=row.dtype)


def load_calibration_texts(
    dataset: str,
    n_calib: int,
    *,
    mmlu_subject: str = "high_school_mathematics",
    mmlu_split: str = "test",
    max_chars: int = 50_000,
) -> list[str]:
    """Plain-text calibration strings (same task as Shapley scoring)."""
    dataset = dataset.lower().strip()
    if dataset == "wikitext":
        from datasets import load_dataset

        cap = max(n_calib * 20, 500)
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{cap}]")
        ds = ds.filter(lambda x: len(x.get("text") or "") > 50)
        return [x["text"] for x in ds][:n_calib]

    if dataset == "c4":
        from hf_calibration import load_c4_texts

        texts = load_c4_texts(
            n_calib,
            split="train",
            max_chars=max_chars,
            allow_wikitext_fallback=os.environ.get("C4_NO_FALLBACK", "0") != "1",
        )
        return texts[:n_calib]

    if dataset == "mmlu":
        from hf_calibration import load_mmlu_split

        ds = load_mmlu_split(mmlu_subject, mmlu_split)
        n = min(n_calib, len(ds))
        out: list[str] = []
        for i in range(n):
            row = ds[i]
            lines = [f"Question: {row['question']}"]
            for j, choice in enumerate(row["choices"]):
                lines.append(f"{LETTERS[j]}. {choice}")
            lines.append(f"Answer: {LETTERS[int(row['answer'])]}.")
            out.append("\n".join(lines))
        return out

    raise ValueError(f"Unknown dataset={dataset!r}; use wikitext|c4|mmlu")


def apply_uniform_row_quant(
    model,
    scores: dict[str, torch.Tensor],
    bits: int = 8,
) -> dict[str, float | int]:
    """Fake INT8 on every scored neuron row (100% quant, no Shapley ranking)."""
    total = 0
    quant_rows = 0
    touched = 0
    for name, param in model.named_parameters():
        if name not in scores:
            continue
        touched += 1
        w = param.data
        for i in range(w.shape[0]):
            w[i] = fake_quant_row(w[i], bits=bits)
            quant_rows += 1
            total += 1

    return {
        "total_rows": total,
        "kept_rows_fp16": 0,
        "quantized_rows": quant_rows,
        "quantize_frac": 1.0,
        "bits": bits,
        "matrices_touched": touched,
    }


def apply_shapley_row_quant(
    model,
    scores: dict[str, torch.Tensor],
    quantize_frac: float,
    bits: int = 8,
) -> dict[str, float | int]:
    """
    Shapley-guided mixed precision (simulated):
      - top (1 - quantize_frac) neuron rows by |score| → keep FP16/BF16
      - bottom quantize_frac rows → symmetric INT fake-quant round-trip
    """
    if not (0.0 < quantize_frac < 1.0):
        raise ValueError(f"quantize_frac must be in (0, 1), got {quantize_frac}")

    ranked = build_global_score_ranking(scores)
    total = len(ranked)
    n_keep = max(1, int(total * (1.0 - quantize_frac)))
    top_params = set(ranked[:n_keep])
    n_quant = total - n_keep

    quant_rows = 0
    kept_rows = 0
    touched = 0
    for name, param in model.named_parameters():
        if name not in scores:
            continue
        touched += 1
        w = param.data
        n_rows = w.shape[0]
        for i in range(n_rows):
            if (name, i) in top_params:
                kept_rows += 1
                continue
            w[i] = fake_quant_row(w[i], bits=bits)
            quant_rows += 1

    return {
        "total_rows": total,
        "kept_rows_fp16": kept_rows,
        "quantized_rows": quant_rows,
        "quantize_frac": quantize_frac,
        "bits": bits,
        "matrices_touched": touched,
    }


def save_quant_manifest(output_dir: str, payload: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "quant_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _log(f"Saved {path}")


def save_hf_model(model, tokenizer: PreTrainedTokenizerBase, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _log(f"Saved HF model → {output_dir}")
