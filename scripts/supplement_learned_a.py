#!/usr/bin/env python3
"""
Train scheme-A surrogate and save ONLY shapley_learned_a.pt (+ surrogate_a.pt).
Does not overwrite gradient / fisher / learned_b / surrogate_b.

  python3 supplement_learned_a.py --dataset c4 --output_dir ./results_c4
  python3 supplement_learned_a.py --dataset mmlu --output_dir ./results_mmlu
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_experiment import (  # noqa: E402
    _DEFAULT_MODEL,
    _require_cuda_gpu,
    apply_cuda_memory_fraction,
    build_loader,
)
from model_utils import load_tokenizer_and_causal_lm  # noqa: E402
from neural_function import compute_and_cache_metrics, train_surrogate_a  # noqa: E402
from run_experiment import _kendall_fisher_vs_learned  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Supplement scheme A scores only.")
    p.add_argument("--model", default=_DEFAULT_MODEL)
    p.add_argument(
        "--dataset",
        required=True,
        choices=("wikitext", "c4", "mmlu"),
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mmlu_subject", default="abstract_algebra")
    p.add_argument("--hf_dataset_path", default="")
    p.add_argument("--hf_dataset_config", default="")
    p.add_argument("--hf_split", default="")
    p.add_argument("--hf_text_column", default="text")
    p.add_argument("--n_calib", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--surrogate_epochs", type=int, default=30)
    p.add_argument("--hutchinson_samples", type=int, default=5)
    args = p.parse_args()

    _require_cuda_gpu()
    apply_cuda_memory_fraction()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer, model = load_tokenizer_and_causal_lm(args.model)
    model.eval()

    loader = build_loader(tokenizer, args)

    print("\n[Phase 1] Training curvature surrogate (scheme A only)...")
    surrogate_a = train_surrogate_a(
        model,
        loader,
        n_epochs=args.surrogate_epochs,
        hutchinson_samples=args.hutchinson_samples,
    )
    path_a = os.path.join(args.output_dir, "surrogate_a.pt")
    torch.save(surrogate_a.state_dict(), path_a)
    print(f"  Saved {path_a}")

    print("\n[Phase 2] Computing Shapley (learned_a only will be written to disk)...")
    scores = compute_and_cache_metrics(model, loader, surrogate_a=surrogate_a, surrogate_b=None)

    learned_path = os.path.join(args.output_dir, "shapley_learned_a.pt")
    if "shapley_learned_a" not in scores:
        raise RuntimeError("shapley_learned_a missing from compute_and_cache_metrics output")
    torch.save(
        {k: v.cpu() for k, v in scores["shapley_learned_a"].items()},
        learned_path,
    )
    print(f"  Saved {learned_path} (other *.pt in {args.output_dir} left unchanged)")

    out_a = _kendall_fisher_vs_learned(scores, "shapley_learned_a")
    fisher_pt = os.path.join(args.output_dir, "shapley_fisher.pt")
    if out_a is not None:
        tau, p = out_a
        print(f"\n[Kendall] Fisher vs Learned A (in-memory Fisher): τ={tau:.4f}  (p={p:.4e})")
    elif os.path.isfile(fisher_pt):
        from evaluate import load_scores

        fisher = load_scores(args.output_dir, "shapley_fisher")
        learned = load_scores(args.output_dir, "shapley_learned_a")
        merged = {"shapley_fisher": fisher, "shapley_learned_a": learned}
        out_disk = _kendall_fisher_vs_learned(merged, "shapley_learned_a")
        if out_disk is not None:
            tau, p = out_disk
            print(f"\n[Kendall] Fisher vs Learned A (on-disk Fisher): τ={tau:.4f}  (p={p:.4e})")


if __name__ == "__main__":
    main()
