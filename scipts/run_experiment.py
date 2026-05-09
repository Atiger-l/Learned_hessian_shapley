"""
run_experiment.py — end-to-end pipeline
Usage:
  python run_experiment.py --model /data/Qwen/Qwen2.5-3B-Instruct --dataset mmlu
"""
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from neural_function import (
    compute_and_cache_metrics,
    train_surrogate,
)


# ── minimal dataset stub (replace with real data loader) ──────────────────────
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


def main(args):
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    dataset = DummyDataset(tokenizer, n=args.n_calib)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

    # ── Phase 1: train surrogate ───────────────────────────────────────────────
    surrogate = None
    if args.use_learned:
        print("\n[Phase 1] Training curvature surrogate...")
        surrogate = train_surrogate(
            model, loader,
            n_epochs=args.surrogate_epochs,
            hutchinson_samples=args.hutchinson_samples,
        )
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(surrogate.state_dict(), f"{args.output_dir}/surrogate.pt")
        print(f"  Surrogate saved to {args.output_dir}/surrogate.pt")

    # ── Phase 2: compute Shapley scores ───────────────────────────────────────
    print("\n[Phase 2] Computing Shapley scores...")
    scores = compute_and_cache_metrics(model, loader, surrogate=surrogate)

    os.makedirs(args.output_dir, exist_ok=True)
    for metric_name, metric_data in scores.items():
        path = f"{args.output_dir}/{metric_name}.pt"
        torch.save({k: v.cpu() for k, v in metric_data.items()}, path)
        print(f"  Saved {metric_name} → {path}")

    # ── Phase 3: quick comparison ─────────────────────────────────────────────
    if "shapley_fisher" in scores and "shapley_learned" in scores:
        print("\n[Phase 3] Comparing Fisher vs Learned Shapley...")
        from scipy.stats import kendalltau
        fisher_flat, learned_flat = [], []
        for name in scores["shapley_fisher"]:
            if name in scores["shapley_learned"]:
                fisher_flat.append(scores["shapley_fisher"][name].cpu().float())
                learned_flat.append(scores["shapley_learned"][name].cpu().float())
        f = torch.cat(fisher_flat).numpy()
        l = torch.cat(learned_flat).numpy()
        tau, p = kendalltau(f, l)
        print(f"  Kendall's τ (Fisher vs Learned): {tau:.4f}  (p={p:.4e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data/Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dataset", default="mmlu")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--n_calib", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--use_learned", action="store_true", default=True)
    p.add_argument("--surrogate_epochs", type=int, default=30)
    p.add_argument("--hutchinson_samples", type=int, default=5)
    main(p.parse_args())
