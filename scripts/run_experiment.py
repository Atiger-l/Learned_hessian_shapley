"""
run_experiment.py — end-to-end pipeline
Usage (from scripts/ directory; use python3 if `python` is missing):

  source .../conda.sh && conda activate lhs   # see download_qwen.py header
  python3 run_experiment.py --model ../Qwen2.5-3B-Instruct --dataset dummy
  python3 run_experiment.py --model ../Qwen2.5-3B-Instruct --dataset wikitext --n_calib 128
  python3 run_experiment.py --model ../Qwen2.5-3B-Instruct --dataset mmlu --mmlu_subject abstract_algebra
  python3 run_experiment.py ... --learned_scheme b    # scheme B only (HVPSurrogate; heavier HVP)
  python3 run_experiment.py ... --learned_scheme both # train A + B
  pip install datasets   # Hugging Face datasets (first-time Hub download goes to HF cache)
"""
import argparse
import sys
from pathlib import Path

# Allow `python path/to/run_experiment.py` from repo root
_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent
_DEFAULT_MODEL = str(_REPO_ROOT / "Qwen2.5-3B-Instruct")
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from hf_calibration import build_hf_calibration_dataset
from model_utils import load_tokenizer_and_causal_lm
from neural_function import (
    compute_and_cache_metrics,
    train_surrogate_a,
    train_surrogate_b,
)


# ── Fallback stub when --dataset dummy ─────────────────────────────────────────
class DummyDataset(Dataset):
    def __init__(self, tokenizer, n=64, seq_len=128):
        self.data = [
            tokenizer("The quick brown fox jumps over the lazy dog. " * 4,
                      return_tensors="pt", max_length=seq_len,
                      truncation=True, padding="max_length")
            for _ in range(n)
        ]
        for d in self.data:
            d["loss_mask"] = torch.ones(1, seq_len)

    def __len__(self): return len(self.data)
    def __getitem__(self, i): return {k: v.squeeze(0) for k, v in self.data[i].items()}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def build_loader(tokenizer, args):
    if args.dataset == "dummy":
        ds = DummyDataset(tokenizer, n=args.n_calib, seq_len=args.seq_len)
    else:
        ds = build_hf_calibration_dataset(
            tokenizer,
            preset=args.dataset,
            n_calib=args.n_calib,
            seq_len=args.seq_len,
            hf_dataset_path=args.hf_dataset_path or None,
            hf_dataset_config=args.hf_dataset_config or None,
            hf_split=args.hf_split or None,
            hf_text_column=args.hf_text_column,
            mmlu_subject=args.mmlu_subject,
        )
    return DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)


def _require_cuda_gpu():
    """This codebase uses .cuda() throughout neural_function; fail fast with fix hints."""
    if torch.cuda.is_available():
        return
    tv = getattr(torch.version, "cuda", None)
    extra_cu130 = ""
    if tv is not None and (tv == "13.0" or str(tv).startswith("13")):
        extra_cu130 = (
            "\nNote: Your build reports torch.version.cuda=%s (CUDA 13-series wheel).\n"
            "If nvidia-smi shows CUDA Version 12.x, that combo often breaks — reinstall PyTorch\n"
            "with cu124 or cu121 from https://pytorch.org/get-started/locally/\n"
            "  pip uninstall -y torch torchvision torchaudio\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n\n"
        ) % tv
    print(
        "\nERROR: PyTorch does not see a usable GPU (CUDA).\n"
        + extra_cu130
        + "Other causes: wrong node (CPU-only), or driver/toolkit mismatch.\n\n"
        "Check:\n"
        "  nvidia-smi          # top-right 'CUDA Version' is driver's toolkit compatibility ceiling\n"
        "  python3 -c \"import torch; print(torch.__version__, torch.version.cuda)\"\n\n"
        "Fix: reinstall PyTorch built for CUDA <= what your driver supports, e.g. cu124:\n"
        "  pip uninstall -y torch torchvision torchaudio\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def apply_cuda_memory_fraction(fraction: float | None = None) -> None:
    """
    Cap this process to a fraction of GPU *total* memory (cuda:0 after CUDA_VISIBLE_DEVICES).
    Use when the card is shared, e.g. half VRAM already taken:

      CUDA_MEM_FRACTION=0.48 GPU=5 python3 run_experiment.py ...

    0.48 × 49GiB ≈ 23GiB cap for this job; tune with nvidia-smi (free / total).
    """
    if fraction is None:
        raw = os.environ.get("CUDA_MEM_FRACTION", "").strip()
        if not raw:
            return
        fraction = float(raw)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"CUDA_MEM_FRACTION must be in (0, 1], got {fraction}")
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(
        f"CUDA memory cap: fraction={fraction:.2f} "
        f"(~{fraction * total_gib:.1f} GiB max on logical cuda:0)",
        flush=True,
    )


def _kendall_fisher_vs_learned(scores: dict, learned_key: str) -> tuple[float, float] | None:
    """Flatten Fisher vs learned scores over shared parameter blocks; return (tau, p) or None."""
    if "shapley_fisher" not in scores or learned_key not in scores:
        return None
    fisher_flat, learned_flat = [], []
    for name in scores["shapley_fisher"]:
        if name in scores[learned_key]:
            fisher_flat.append(scores["shapley_fisher"][name].cpu().float())
            learned_flat.append(scores[learned_key][name].cpu().float())
    if not fisher_flat:
        return None
    from scipy.stats import kendalltau

    f = torch.cat(fisher_flat).numpy()
    l = torch.cat(learned_flat).numpy()
    tau, p = kendalltau(f, l)
    return float(tau), float(p)


def main(args):
    _require_cuda_gpu()
    apply_cuda_memory_fraction(getattr(args, "cuda_mem_fraction", None))
    print(f"Loading model: {args.model}")
    tokenizer, model = load_tokenizer_and_causal_lm(args.model)
    model.eval()

    loader = build_loader(tokenizer, args)

    surrogate_a = None
    surrogate_b = None
    if not args.no_learned:
        os.makedirs(args.output_dir, exist_ok=True)
        scheme = args.learned_scheme
        if scheme in ("a", "both"):
            print("\n[Phase 1a] Training curvature surrogate (scheme A)...")
            surrogate_a = train_surrogate_a(
                model, loader,
                n_epochs=args.surrogate_epochs,
                hutchinson_samples=args.hutchinson_samples,
            )
            torch.save(surrogate_a.state_dict(), f"{args.output_dir}/surrogate_a.pt")
            print(f"  Surrogate A saved to {args.output_dir}/surrogate_a.pt")
        if scheme in ("b", "both"):
            print("\n[Phase 1b] Training HVP surrogate (scheme B; uses compute_hvp each epoch)...")
            surrogate_b = train_surrogate_b(
                model, loader,
                n_epochs=args.surrogate_epochs,
                batches_per_epoch=args.surrogate_batches_per_epoch,
            )
            torch.save(surrogate_b.state_dict(), f"{args.output_dir}/surrogate_b.pt")
            print(f"  Surrogate B saved to {args.output_dir}/surrogate_b.pt")

    print("\n[Phase 2] Computing Shapley scores...")
    scores = compute_and_cache_metrics(
        model, loader, surrogate_a=surrogate_a, surrogate_b=surrogate_b
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for metric_name, metric_data in scores.items():
        path = f"{args.output_dir}/{metric_name}.pt"
        torch.save({k: v.cpu() for k, v in metric_data.items()}, path)
        print(f"  Saved {metric_name} → {path}")

    out_a = _kendall_fisher_vs_learned(scores, "shapley_learned_a")
    out_b = _kendall_fisher_vs_learned(scores, "shapley_learned_b")
    if out_a is not None or out_b is not None:
        print("\n[Phase 3] Rank agreement: Fisher Shapley vs learned surrogates (Kendall's τ)...")
        if out_a is not None:
            tau, p = out_a
            print(f"  Fisher vs Learned A: τ={tau:.4f}  (p={p:.4e})")
        if out_b is not None:
            tau_b, p_b = out_b
            print(f"  Fisher vs Learned B: τ={tau_b:.4f}  (p={p_b:.4e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=_DEFAULT_MODEL)
    p.add_argument(
        "--dataset",
        default="dummy",
        choices=("dummy", "wikitext", "c4", "mmlu", "custom"),
        help="Calibration data: dummy=in-memory stub; others use Hugging Face datasets (see hf_calibration.py).",
    )
    p.add_argument("--hf_dataset_path", default="", help="For --dataset custom: e.g. taufiqrahmat/indonesian-news")
    p.add_argument("--hf_dataset_config", default="", help="Subset/config name if required by the dataset")
    p.add_argument("--hf_split", default="", help='Split name, e.g. train or dev[:500]')
    p.add_argument("--hf_text_column", default="text", help="Text field for --dataset custom")
    p.add_argument("--mmlu_subject", default="abstract_algebra", help="MMLU config name under cais/mmlu")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--n_calib", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument(
        "--no_learned",
        action="store_true",
        help="Skip surrogate training (only gradient + Fisher scores).",
    )
    p.add_argument(
        "--learned_scheme",
        choices=("a", "b", "both"),
        default="a",
        help="Which learned surrogate to train: a=diag (scheme A), b=HVP (scheme B), both=A+B. Ignored with --no_learned.",
    )
    p.add_argument("--surrogate_epochs", type=int, default=30)
    p.add_argument("--hutchinson_samples", type=int, default=5)
    p.add_argument(
        "--cuda_mem_fraction",
        type=float,
        default=None,
        help="Max fraction of GPU total VRAM for this process (or env CUDA_MEM_FRACTION). "
        "Use ~0.45–0.50 when sharing a half-full GPU.",
    )
    p.add_argument(
        "--surrogate_batches_per_epoch",
        type=int,
        default=8,
        help="Scheme B only: number of calibration batches of (HVP + features) teacher per epoch "
        "before one Adam step; capped by len(DataLoader). Use 1 to match old single-batch behavior.",
    )
    main(p.parse_args())
