"""
Load calibration text from Hugging Face `datasets` into the batch format expected by neural_function:
  input_ids, attention_mask, loss_mask (same shape as input_ids; 0 on padded positions).
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def _ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _encode_one(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    enc = tokenizer(
        text,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    row = {k: v.squeeze(0) for k, v in enc.items()}
    row["loss_mask"] = row["attention_mask"].clone()
    return row


class HFTextCalibrationDataset(Dataset):
    """Plain LM calibration: each row is one text field (e.g. C4, WikiText)."""

    def __init__(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        max_samples: int,
    ):
        _ensure_pad_token(tokenizer)
        self.samples: list[dict[str, torch.Tensor]] = []
        for t in texts:
            if len(self.samples) >= max_samples:
                break
            if not t or not str(t).strip():
                continue
            self.samples.append(_encode_one(tokenizer, str(t).strip(), seq_len))
        if not self.samples:
            raise ValueError("HFTextCalibrationDataset: no non-empty texts after filtering.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self.samples[i]


class MMLUCalibrationDataset(Dataset):
    """Multiple-choice rows formatted as a single string for causal LM loss."""

    def __init__(
        self,
        hf_split,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        max_samples: int,
    ):
        _ensure_pad_token(tokenizer)
        letters = ["A", "B", "C", "D"]
        self.samples: list[dict[str, torch.Tensor]] = []
        n = min(max_samples, len(hf_split))
        for i in range(n):
            row = hf_split[i]
            lines = [f"Question: {row['question']}"]
            for j, choice in enumerate(row["choices"]):
                lines.append(f"{letters[j]}. {choice}")
            lines.append(f"Answer: {letters[int(row['answer'])]}.")
            text = "\n".join(lines)
            self.samples.append(_encode_one(tokenizer, text, seq_len))
        if not self.samples:
            raise ValueError("MMLUCalibrationDataset: empty split.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self.samples[i]


def build_hf_calibration_dataset(
    tokenizer: PreTrainedTokenizerBase,
    *,
    preset: str,
    n_calib: int,
    seq_len: int,
    hf_dataset_path: str | None,
    hf_dataset_config: str | None,
    hf_split: str | None,
    hf_text_column: str,
    mmlu_subject: str,
    c4_streaming_max_chars: int = 50_000,
) -> Dataset:
    """
    preset:
      dummy — caller should use DummyDataset in run_experiment (not used here).
      wikitext — WikiText-103 raw train slice.
      c4 — allenai/c4 English train via streaming (first chunks until ~enough strings).
      mmlu — cais/mmlu multiple-choice formatted as text (subject = mmlu_subject).
      custom — hf_dataset_path required; loads non-streaming split hf_split.
    """
    from datasets import load_dataset

    preset = preset.lower().strip()
    _ensure_pad_token(tokenizer)

    if preset == "wikitext":
        # Enough rows to skip empties and still get n_calib samples
        cap = max(n_calib * 20, 500)
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{cap}]")
        ds = ds.filter(lambda x: len(x.get("text") or "") > 50)
        texts = [x["text"] for x in ds][: n_calib * 2]
        return HFTextCalibrationDataset(texts, tokenizer, seq_len, n_calib)

    if preset == "c4":
        texts: list[str] = []
        stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
        total = 0
        for ex in stream:
            t = (ex.get("text") or "").strip()
            if len(t) < 80:
                continue
            texts.append(t)
            total += len(t)
            if len(texts) >= n_calib * 2 or total >= c4_streaming_max_chars:
                break
        return HFTextCalibrationDataset(texts, tokenizer, seq_len, n_calib)

    if preset == "mmlu":
        subj = mmlu_subject or "abstract_algebra"
        split_name = hf_split or "dev"
        ds = load_dataset("cais/mmlu", subj, split=split_name)
        return MMLUCalibrationDataset(ds, tokenizer, seq_len, n_calib)

    if preset == "custom":
        if not hf_dataset_path:
            raise ValueError("preset=custom requires --hf_dataset_path")
        split = hf_split or "train"
        if hf_dataset_config:
            ds = load_dataset(hf_dataset_path, hf_dataset_config, split=split)
        else:
            ds = load_dataset(hf_dataset_path, split=split)
        col = hf_text_column
        if col not in ds.column_names:
            raise ValueError(f"Column '{col}' not in {ds.column_names}")
        cap = min(max(n_calib * 20, 200), len(ds))
        texts = []
        for i in range(cap):
            val = ds[i][col]
            if val:
                texts.append(str(val))
        return HFTextCalibrationDataset(texts, tokenizer, seq_len, n_calib)

    raise ValueError(f"Unknown dataset preset: {preset!r}")
