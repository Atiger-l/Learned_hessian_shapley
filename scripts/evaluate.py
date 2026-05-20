"""
evaluate.py — compare importance metrics via inference-time neuron deactivation.

For each metric, keeps the global top-|score| fraction of neuron rows (same
granularity as Shapley scoring), zeros other rows in the corresponding weight
matrices, runs forward PPL, then restores weights from a snapshot.
"""
import argparse
import os
import torch
from pathlib import Path

from model_utils import load_tokenizer_and_causal_lm
from neural_function import _infer_causal_lm_input_device

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL = str(_REPO_ROOT / "Qwen2.5-3B-Instruct")

METRICS = ("shapley_fisher", "shapley_learned_a", "shapley_learned_b", "gradient")


def load_scores(output_dir: str, metric: str) -> dict:
    path = f"{output_dir}/{metric}.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No scores at {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def build_global_score_ranking(scores: dict) -> list[tuple[str, int]]:
    """(param_name, row_idx) sorted by |score| descending (global pool)."""
    meta: list[tuple[str, int]] = []
    chunks: list[torch.Tensor] = []
    for name, val in scores.items():
        flat = val.float().view(-1).abs()
        n = flat.numel()
        chunks.append(flat)
        meta.extend((name, i) for i in range(n))
    if not chunks:
        return []
    order = torch.argsort(torch.cat(chunks), descending=True).tolist()
    return [meta[i] for i in order]


def get_top_k_params(scores: dict, k: float) -> set:
    """Global top-k% neuron rows by |score| (same pooling as run_experiment)."""
    ranked = build_global_score_ranking(scores)
    n_keep = max(1, int(len(ranked) * k))
    return set(ranked[:n_keep])


def snapshot_scored_weights(model, score_names: set[str]) -> dict[str, torch.Tensor]:
    snap: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name in score_names:
            snap[name] = param.data.detach().clone()
    return snap


def restore_scored_weights(model, snap: dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if name in snap:
            param.data.copy_(snap[name])


def apply_inference_row_mask(
    model,
    scores: dict,
    top_params: set,
) -> tuple[int, int]:
    """
    Deactivate neurons at inference: zero each scored weight row not in top_params.
    Matches Model Shapley inference (keep high-|Shapley| rows, remove the rest).
    """
    kept, total = 0, 0
    for name, param in model.named_parameters():
        if name not in scores:
            continue
        n_rows = param.shape[0]
        total += n_rows
        keep_row = torch.zeros(n_rows, dtype=torch.bool, device=param.device)
        for n, i in top_params:
            if n == name and 0 <= i < n_rows:
                keep_row[i] = True
        kept += int(keep_row.sum().item())
        if keep_row.all():
            continue
        if not keep_row.any():
            param.data.zero_()
            continue
        param.data.mul_(keep_row.unsqueeze(1).to(dtype=param.dtype))
    return kept, total


def evaluate_perplexity(model, tokenizer, texts: list) -> float:
    model.eval()
    device = _infer_causal_lm_input_device(model)
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])
            n = enc["input_ids"].numel()
            total_loss += out.loss.item() * n
            total_tokens += n
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def main(args):
    tokenizer, model = load_tokenizer_and_causal_lm(args.model)

    eval_texts = [
        "The capital of France is Paris. The Eiffel Tower is located there.",
        "Machine learning is a subset of artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
    ] * 10

    # Baseline: full model, no masking
    base_ppl = evaluate_perplexity(model, tokenizer, eval_texts)
    print(f"  {'full_model':20s}  PPL = {base_ppl:.2f}  (no deactivation)")

    results = {"full_model": base_ppl}
    for metric in METRICS:
        try:
            scores = load_scores(args.output_dir, metric)
        except FileNotFoundError:
            print(f"  Skipping {metric} (no scores found)")
            continue

        snap = snapshot_scored_weights(model, set(scores.keys()))
        top_params = get_top_k_params(scores, k=args.top_k)
        kept, total = apply_inference_row_mask(model, scores, top_params)
        ppl = evaluate_perplexity(model, tokenizer, eval_texts)
        restore_scored_weights(model, snap)

        results[metric] = ppl
        frac = kept / max(total, 1)
        print(
            f"  {metric:20s}  PPL = {ppl:.2f}  "
            f"(kept {kept}/{total} rows, {100 * frac:.1f}% active)"
        )

    print("\n── Summary (lower PPL = less damage from keeping only top neurons) ──")
    for m, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {m:20s}  {v:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Inference eval: zero non-top-k neuron rows, then measure PPL."
    )
    p.add_argument("--model", default=_DEFAULT_MODEL)
    p.add_argument("--output_dir", default="./results")
    p.add_argument(
        "--top_k",
        type=float,
        default=0.1,
        help="Fraction of neuron rows to keep active (global top-|score|).",
    )
    main(p.parse_args())
