"""
evaluate.py — compare Fisher vs Learned Shapley on downstream tasks
"""
import argparse
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_scores(output_dir: str, metric: str) -> dict:
    path = f"{output_dir}/{metric}.pt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No scores at {path}")
    return torch.load(path)


def get_top_k_params(scores: dict, k: float = 0.1) -> set:
    """Return names of top-k% neurons by Shapley score."""
    all_scores = []
    for name, val in scores.items():
        for i, s in enumerate(val.float()):
            all_scores.append((name, i, s.item()))
    all_scores.sort(key=lambda x: abs(x[2]), reverse=True)
    n_keep = max(1, int(len(all_scores) * k))
    return {(n, i) for n, i, _ in all_scores[:n_keep]}


def freeze_except_top_k(model, top_params: set):
    """Freeze all params except top-k Shapley neurons."""
    for name, param in model.named_parameters():
        if "model.layers" in name and "mlp" in name and "weight" in name:
            mask = torch.zeros(param.shape[0], dtype=torch.bool)
            for (n, i) in top_params:
                if n == name:
                    mask[i] = True
            param.requires_grad_(False)
            if mask.any():
                param.requires_grad_(True)
        else:
            param.requires_grad_(False)


def evaluate_perplexity(model, tokenizer, texts: list) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.cuda() for k, v in enc.items()}
            out = model(**enc, labels=enc["input_ids"])
            n = enc["input_ids"].numel()
            total_loss += out.loss.item() * n
            total_tokens += n
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # sample eval texts
    eval_texts = [
        "The capital of France is Paris. The Eiffel Tower is located there.",
        "Machine learning is a subset of artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
    ] * 10

    results = {}
    for metric in ["shapley_fisher", "shapley_learned", "gradient"]:
        try:
            scores = load_scores(args.output_dir, metric)
        except FileNotFoundError:
            print(f"  Skipping {metric} (no scores found)")
            continue

        top_params = get_top_k_params(scores, k=args.top_k)
        freeze_except_top_k(model, top_params)

        ppl = evaluate_perplexity(model, tokenizer, eval_texts)
        results[metric] = ppl
        print(f"  {metric:20s}  PPL = {ppl:.2f}")

    print("\n── Summary ──")
    for m, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {m:20s}  {v:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data/Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--top_k", type=float, default=0.1)
    main(p.parse_args())
