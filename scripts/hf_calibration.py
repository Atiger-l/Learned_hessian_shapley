"""
Load calibration text from Hugging Face `datasets` into the batch format expected by neural_function:
  input_ids, attention_mask, loss_mask (same shape as input_ids; 0 on padded positions).
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# Default to hf-mirror on compute nodes (must be set before huggingface_hub / datasets import).
_DEFAULT_MIRROR = "https://hf-mirror.com"
if os.environ.get("USE_OFFICIAL_HF", "0") != "1":
    os.environ.setdefault("HF_ENDPOINT", _DEFAULT_MIRROR)


def _apply_hf_mirror_env() -> None:
    """Align huggingface_hub / datasets with HF_ENDPOINT (hf-mirror on compute nodes)."""
    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        os.environ.setdefault("HUGGINGFACE_HUB_BASE_URL", endpoint)
        os.environ.setdefault("HF_HUB_ENDPOINT", endpoint)
    try:
        import datasets

        if endpoint and hasattr(datasets.config, "HF_ENDPOINT"):
            datasets.config.HF_ENDPOINT = endpoint  # type: ignore[attr-defined]
    except ImportError:
        pass


_apply_hf_mirror_env()

C4_TRAIN_SHARD_COUNT = 1024
MMLU_REPO = "cais/mmlu"
MMLU_SPLITS = ("dev", "test", "validation")


def _mmlu_parquet_name(subject: str, split: str) -> str:
    return f"{subject}/{split}-00000-of-00001.parquet"


def _hf_hub_cache_root() -> str:
    return os.environ.get(
        "HF_HUB_CACHE",
        os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"),
    )


def _find_local_mmlu_subject_dir(subject: str) -> str | None:
    """Return snapshot dir if subject has at least dev parquet (symlink ok)."""
    base = os.path.join(_hf_hub_cache_root(), "datasets--cais--mmlu", "snapshots")
    if not os.path.isdir(base):
        return None
    for rev in sorted(os.listdir(base), reverse=True):
        subj_dir = os.path.join(base, rev, subject)
        dev_pq = os.path.join(subj_dir, "dev-00000-of-00001.parquet")
        if os.path.isfile(dev_pq) or os.path.islink(dev_pq):
            return subj_dir
    return None


def download_mmlu_subject(
    subject: str,
    splits: tuple[str, ...] = MMLU_SPLITS,
) -> dict[str, str]:
    """Download MMLU subject parquet file(s) via HF mirror. Returns split -> local path."""
    from huggingface_hub import hf_hub_download

    _apply_hf_mirror_env()
    endpoint = os.environ.get("HF_ENDPOINT", _DEFAULT_MIRROR).rstrip("/")
    out: dict[str, str] = {}
    for split in splits:
        rel = _mmlu_parquet_name(subject, split)
        print(f"[mmlu] downloading {rel} via {endpoint} ...", flush=True)
        try:
            path = hf_hub_download(
                repo_id=MMLU_REPO,
                repo_type="dataset",
                filename=rel,
                endpoint=endpoint,
            )
        except TypeError:
            path = hf_hub_download(
                repo_id=MMLU_REPO,
                repo_type="dataset",
                filename=rel,
            )
        out[split] = path
        print(f"[mmlu]  {split} -> {path}", flush=True)
    return out


def load_mmlu_split(subject: str, split: str):
    """
    Load one MMLU subject split (dev/test/validation).
    Prefer local cache / mirror download; avoids huggingface.co SSL timeout.
    """
    from datasets import load_dataset

    _apply_hf_mirror_env()
    subject = subject or "abstract_algebra"
    pq_name = f"{split}-00000-of-00001.parquet"

    subj_dir = _find_local_mmlu_subject_dir(subject)
    if subj_dir:
        pq = os.path.join(subj_dir, pq_name)
        if os.path.isfile(pq) or os.path.islink(pq):
            print(f"[mmlu] load {subject}/{split} from cache {pq}", flush=True)
            return load_dataset("parquet", data_files=pq, split="train")

    try:
        paths = download_mmlu_subject(subject, splits=(split,))
        pq = paths[split]
        return load_dataset("parquet", data_files=pq, split="train")
    except Exception as e:
        print(f"[mmlu] mirror download failed: {e}", flush=True)

    print(f"[mmlu] fallback load_dataset(cais/mmlu, {subject}, {split})", flush=True)
    return load_dataset(
        MMLU_REPO,
        subject,
        split=split,
        download_mode="reuse_dataset_if_exists",
    )


def _c4_train_shard_name(shard: int) -> str:
    shard = max(0, min(shard, C4_TRAIN_SHARD_COUNT - 1))
    return f"en/c4-train.{shard:05d}-of-01024.json.gz"


def _resolve_c4_shard_path(shard: int) -> str:
    """Local path: C4_CALIB_DIR / scripts/data/c4_en, else download one shard via HF mirror."""
    fname = _c4_train_shard_name(shard)
    base = os.path.basename(fname)
    here = os.path.dirname(os.path.abspath(__file__))
    search_dirs = []
    if os.environ.get("C4_CALIB_DIR", "").strip():
        search_dirs.append(os.environ["C4_CALIB_DIR"].strip())
    search_dirs.append(os.path.join(here, "data", "c4_en"))

    for calib_dir in search_dirs:
        for candidate in (
            calib_dir,
            os.path.join(calib_dir, base),
            os.path.join(calib_dir, fname),
        ):
            if os.path.isfile(candidate):
                return candidate

    from huggingface_hub import hf_hub_download

    _apply_hf_mirror_env()
    endpoint = os.environ.get("HF_ENDPOINT", _DEFAULT_MIRROR).rstrip("/")
    print(f"[c4] downloading {fname} via {endpoint} ...", flush=True)
    try:
        return hf_hub_download(
            repo_id="allenai/c4",
            repo_type="dataset",
            filename=fname,
            endpoint=endpoint,
        )
    except TypeError:
        # older huggingface_hub: relies on HF_ENDPOINT env only
        return hf_hub_download(
            repo_id="allenai/c4",
            repo_type="dataset",
            filename=fname,
        )


def _load_c4_from_shard(
    n_texts: int,
    *,
    max_chars: int,
    min_len: int,
) -> list[str]:
    """Load calibration texts from one C4 en train json.gz shard (real C4, mirror-friendly)."""
    from datasets import load_dataset

    shard = int(os.environ.get("C4_SHARD", "0"))
    path = _resolve_c4_shard_path(shard)
    print(f"[c4] loading train shard {shard} from {path}", flush=True)

    cap_rows = max(n_texts * 8, 200)
    try:
        ds = load_dataset(
            "json",
            data_files={"train": path},
            split=f"train[:{cap_rows}]",
            streaming=False,
        )
        it = ds
    except Exception:
        ds = load_dataset("json", data_files={"train": path}, split="train", streaming=True)
        it = ds

    texts: list[str] = []
    total = 0
    for ex in it:
        t = (ex.get("text") or "").strip()
        if len(t) < min_len:
            continue
        texts.append(t)
        total += len(t)
        if len(texts) >= n_texts * 2 or total >= max_chars:
            break

    if not texts:
        raise RuntimeError(f"C4 shard produced no texts (path={path})")
    print(f"[c4] loaded {len(texts)} texts from shard (real C4 en train)", flush=True)
    return texts


def load_c4_texts(
    n_texts: int,
    *,
    split: str = "train",
    max_chars: int = 50_000,
    min_len: int = 80,
    allow_wikitext_fallback: bool = True,
) -> list[str]:
    """
    Load real C4 en train text for calibration.

    Order:
      1) One train json.gz shard (hf-mirror / C4_CALIB_DIR) — recommended on compute nodes
      2) Non-streaming train slice via datasets
      3) Streaming train only
      4) WikiText fallback only if allow_wikitext_fallback=True

    Env:
      HF_ENDPOINT=https://hf-mirror.com
      C4_NO_FALLBACK=1  — fail instead of WikiText (recommended for C4 experiments)
      C4_CALIB_DIR=...  — local path to shard .json.gz or directory
      C4_SHARD=0        — which of 1024 train shards (default 0)
    """
    from datasets import load_dataset

    _apply_hf_mirror_env()
    if os.environ.get("C4_NO_FALLBACK", "0") == "1":
        allow_wikitext_fallback = False

    # 0) Single train shard — avoids allenai/c4 builder + validation split issues
    try:
        return _load_c4_from_shard(n_texts, max_chars=max_chars, min_len=min_len)
    except Exception as e:
        print(f"[c4] shard load failed: {e}", flush=True)

    cap = max(n_texts * 4, 400)
    texts: list[str] = []

    def _collect_from_iter(it) -> list[str]:
        out: list[str] = []
        total = 0
        for ex in it:
            t = (ex.get("text") or "").strip()
            if len(t) < min_len:
                continue
            out.append(t)
            total += len(t)
            if len(out) >= n_texts * 2 or total >= max_chars:
                break
        return out

    # 1) Non-streaming slice — works with hf-mirror + local cache (recommended)
    try:
        slice_spec = f"{split}[:{cap}]" if split in ("train", "validation") else split
        ds = load_dataset(
            "allenai/c4",
            "en",
            split=slice_spec,
            download_mode="reuse_dataset_if_exists",
        )
        texts = _collect_from_iter(ds)
        if texts:
            print(f"[c4] loaded {len(texts)} texts via non-streaming {slice_spec}", flush=True)
            return texts
    except Exception as e:
        print(f"[c4] non-streaming load failed: {e}", flush=True)

    # 2) Streaming train only (do not load validation — format inference breaks offline)
    try:
        stream = load_dataset(
            "allenai/c4",
            "en",
            split=split,
            streaming=True,
            data_files={split: _c4_train_shard_name(int(os.environ.get("C4_SHARD", "0")))},
        )
        texts = _collect_from_iter(stream)
        if texts:
            print(f"[c4] loaded {len(texts)} texts via streaming", flush=True)
            return texts
    except Exception as e:
        print(f"[c4] streaming load failed: {e}", flush=True)

    if not allow_wikitext_fallback:
        raise RuntimeError(
            "Could not load real allenai/c4.\n"
            "  bash download_c4_calib.sh\n"
            "  export HF_ENDPOINT=https://hf-mirror.com\n"
            "  export C4_CALIB_DIR=./data/c4_en\n"
            "(WikiText fallback disabled: C4_NO_FALLBACK=1)"
        )

    # 3) Offline fallback: WikiText-103 train (label clearly in logs for PPT)
    print(
        "[c4] WARNING: using WikiText-103 train as C4 calibration fallback "
        "(set C4_NO_FALLBACK=1 to fail instead)",
        flush=True,
    )
    cap_wt = max(n_texts * 20, 500)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{cap_wt}]")
    ds = ds.filter(lambda x: len(x.get("text") or "") > min_len)
    return [x["text"] for x in ds][: n_texts * 2]


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
        texts = load_c4_texts(
            n_calib,
            split="train",
            max_chars=c4_streaming_max_chars,
            allow_wikitext_fallback=os.environ.get("C4_NO_FALLBACK", "0") != "1",
        )
        return HFTextCalibrationDataset(texts, tokenizer, seq_len, n_calib)

    if preset == "mmlu":
        subj = mmlu_subject or "abstract_algebra"
        # MMLU dev 仅 5 条（few-shot），校准需用 test/validation
        split_name = (
            hf_split
            or os.environ.get("MMLU_CALIB_SPLIT", "").strip()
            or "test"
        )
        ds = load_mmlu_split(subj, split_name)
        if len(ds) < n_calib:
            print(
                f"[mmlu] WARNING: {subj}/{split_name} has {len(ds)} rows < n_calib={n_calib}; "
                f"using all available.",
                flush=True,
            )
        print(
            f"[mmlu] calibration: subject={subj} split={split_name} "
            f"rows={len(ds)} n_calib={n_calib}",
            flush=True,
        )
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
