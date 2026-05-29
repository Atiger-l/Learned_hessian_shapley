#!/usr/bin/env python3
"""
Downstream eval using existing Shapley scores (no surrogate retraining).

  MMLU multiple-choice accuracy (default: chat generate + parse A/B/C/D):
    python3 evaluate_downstream.py --task mmlu_acc --scores_dir ./results_wikitext \\
      --mmlu_subject abstract_algebra --parallel_gpus 2,3,4,5

  Legacy logprob scoring: add --mmlu_scoring logprob

  C4 language-modeling PPL:
    python3 evaluate_downstream.py --task c4_ppl --scores_dir ./results_wikitext \\
      --parallel_gpus 2,3,4,5

Writes: <output_dir>/eval_<task>.json and eval_<task>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

from evaluate import (
    METRICS,
    apply_inference_row_mask,
    build_global_score_ranking,
    evaluate_perplexity,
    load_scores,
    restore_scored_weights,
    snapshot_scored_weights,
    _DEFAULT_MODEL,
)
from model_utils import load_tokenizer_and_causal_lm

LETTERS = ["A", "B", "C", "D"]


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_fracs(spec: str) -> list[float]:
    return [round(float(x.strip()), 4) for x in spec.split(",") if x.strip()]


def load_c4_eval_texts(n_texts: int, max_chars: int) -> list[str]:
    from hf_calibration import load_c4_texts

    texts = load_c4_texts(
        n_texts,
        split="validation",
        max_chars=max_chars,
        allow_wikitext_fallback=True,
    )
    if not texts:
        raise ValueError("No C4 eval texts loaded")
    return texts[:n_texts] if len(texts) > n_texts else texts


def load_mmlu_test(subject: str, max_items: int | None = None) -> list[dict]:
    from hf_calibration import load_mmlu_split

    ds = load_mmlu_split(subject, "test")
    n = len(ds) if max_items is None else min(max_items, len(ds))
    out = []
    for i in range(n):
        row = ds[i]
        out.append(
            {
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": int(row["answer"]),
            }
        )
    return out


def mmlu_question_block(question: str, choices: list[str]) -> str:
    lines = [f"Question: {question}"]
    for j, choice in enumerate(choices):
        lines.append(f"{LETTERS[j]}. {choice}")
    return "\n".join(lines)


def mmlu_prompt(question: str, choices: list[str]) -> str:
    return mmlu_question_block(question, choices) + "\nAnswer:"


def mmlu_chat_user_content(question: str, choices: list[str]) -> str:
    return (
        "Answer the following multiple-choice question. "
        "Reply with only the letter of the correct option (A, B, C, or D).\n\n"
        + mmlu_question_block(question, choices)
    )


def parse_mmlu_choice(text: str) -> int | None:
    """Map model output to choice index 0..3, or None if unparseable."""
    if not text:
        return None
    s = text.strip().upper()
    if s and s[0] in "ABCD" and (len(s) == 1 or not s[1].isalpha()):
        return LETTERS.index(s[0])
    m = re.search(r"\b([ABCD])\b", s)
    if m:
        return LETTERS.index(m.group(1))
    return None


def choice_nll(model, tokenizer, prompt: str, letter: str) -> float:
    """Token NLL of completing ` {letter}` after prompt (lower = better)."""
    device = next(model.parameters()).device
    prefix = prompt + " "
    full_text = prefix + letter
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    prefix_len = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=512)[
        "input_ids"
    ].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :prefix_len] = -100
    with torch.no_grad():
        out = model(**enc, labels=labels)
    return float(out.loss.item())


def _evaluate_mmlu_logprob(model, tokenizer, items: list[dict]) -> tuple[float, int]:
    model.eval()
    correct = 0
    for item in items:
        prompt = mmlu_prompt(item["question"], item["choices"])
        losses = [choice_nll(model, tokenizer, prompt, LETTERS[j]) for j in range(4)]
        pred = int(min(range(4), key=lambda j: losses[j]))
        if pred == item["answer"]:
            correct += 1
    return correct / max(len(items), 1), len(items)


def _evaluate_mmlu_generate(
    model,
    tokenizer,
    items: list[dict],
    max_new_tokens: int = 16,
) -> tuple[float, int]:
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    for item in items:
        user_content = mmlu_chat_user_content(item["question"], item["choices"])
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = user_content + "\nAnswer:"

        enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        new_tokens = out_ids[0, prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred = parse_mmlu_choice(text)
        if pred is not None and pred == item["answer"]:
            correct += 1
    return correct / max(len(items), 1), len(items)


def evaluate_mmlu_accuracy(
    model,
    tokenizer,
    items: list[dict],
    scoring: str = "generate",
    max_new_tokens: int = 16,
) -> tuple[float, int]:
    if scoring == "logprob":
        return _evaluate_mmlu_logprob(model, tokenizer, items)
    if scoring != "generate":
        raise ValueError(f"Unknown mmlu_scoring={scoring!r}; use 'generate' or 'logprob'")
    return _evaluate_mmlu_generate(model, tokenizer, items, max_new_tokens=max_new_tokens)


def run_metric_rows(
    model,
    tokenizer,
    scores_dir: str,
    metric: str,
    deactivate_fracs: list[float],
    task: str,
    eval_bundle,
    mmlu_scoring: str = "generate",
    mmlu_max_new_tokens: int = 16,
) -> list[dict]:
    scores = load_scores(scores_dir, metric)
    ranked = build_global_score_ranking(scores)
    total = len(ranked)
    snap = snapshot_scored_weights(model, set(scores.keys()))
    rows: list[dict] = []

    for frac in deactivate_fracs:
        top_k = 1.0 - frac
        n_keep = max(1, int(total * top_k))
        top_params = set(ranked[:n_keep])
        apply_inference_row_mask(model, scores, top_params)
        if task == "mmlu_acc":
            items = eval_bundle
            value, n = evaluate_mmlu_accuracy(
                model,
                tokenizer,
                items,
                scoring=mmlu_scoring,
                max_new_tokens=mmlu_max_new_tokens,
            )
            rows.append(
                {
                    "deactivate_frac": frac,
                    "top_k": top_k,
                    "accuracy": value,
                    "n_items": n,
                }
            )
            _log(f"  {metric:20s}  freeze={frac:.2f}  acc={100 * value:.2f}%  (n={n})")
        else:
            texts = eval_bundle
            ppl = evaluate_perplexity(model, tokenizer, texts)
            rows.append({"deactivate_frac": frac, "top_k": top_k, "ppl": ppl})
            _log(f"  {metric:20s}  freeze={frac:.2f}  PPL={ppl:.2f}")
        restore_scored_weights(model, snap)
    return rows


def save_results(
    output_dir: str,
    task: str,
    deactivate_fracs: list[float],
    baseline: float,
    metrics_rows: dict[str, list[dict]],
    extra: dict | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "task": task,
        "deactivate_fracs": deactivate_fracs,
        "baseline": baseline,
        "metrics": metrics_rows,
        **(extra or {}),
    }
    json_path = os.path.join(output_dir, f"eval_{task}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _log(f"Saved {json_path}")

    csv_path = os.path.join(output_dir, f"eval_{task}.csv")
    key = "accuracy" if task == "mmlu_acc" else "ppl"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["deactivate_frac", "top_k", "metric", key, "n_items"],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerow(
            {
                "deactivate_frac": 0.0,
                "top_k": 1.0,
                "metric": "full_model",
                key: baseline,
            }
        )
        for metric, rows in metrics_rows.items():
            for row in rows:
                w.writerow({"metric": metric, **row})
    _log(f"Saved {csv_path}")


def run_single_worker(argv: list[str]) -> None:
    os.execv(sys.executable, [sys.executable, *argv])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("mmlu_acc", "c4_ppl"), required=True)
    p.add_argument("--model", default=_DEFAULT_MODEL)
    p.add_argument(
        "--scores_dir",
        default="./results_wikitext",
        help="Existing *.pt scores (WikiText run — no retrain).",
    )
    p.add_argument("--output_dir", default="")
    p.add_argument(
        "--deactivate_fracs",
        default="0.05,0.06,0.07,0.08,0.09,0.10",
    )
    p.add_argument("--mmlu_subject", default="abstract_algebra")
    p.add_argument(
        "--mmlu_scoring",
        choices=("generate", "logprob"),
        default="generate",
        help="MMLU scoring: generate (chat + parse letter) or logprob (per-option NLL).",
    )
    p.add_argument(
        "--mmlu_max_new_tokens",
        type=int,
        default=16,
        help="Max tokens to generate per MMLU item when --mmlu_scoring=generate.",
    )
    p.add_argument("--mmlu_max_items", type=int, default=0, help="0 = all test items")
    p.add_argument("--n_eval_texts", type=int, default=128)
    p.add_argument("--max_eval_chars", type=int, default=50_000)
    p.add_argument("--parallel_gpus", default="", help="e.g. 2,3,4,5,6,7")
    p.add_argument("--only_metric", default="", help=argparse.SUPPRESS)
    p.add_argument("--baseline_only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--eval_cache", default="", help=argparse.SUPPRESS)
    args = p.parse_args()

    if not args.output_dir:
        if args.task == "mmlu_acc":
            if args.mmlu_subject == "abstract_algebra":
                args.output_dir = "./results_mmlu_acc"
            else:
                args.output_dir = f"./results_mmlu_acc_{args.mmlu_subject}"
        else:
            args.output_dir = "./results_c4_ppl"

    fracs = parse_fracs(args.deactivate_fracs)
    script = str(Path(__file__).resolve())

    if args.parallel_gpus and not args.only_metric and not args.baseline_only:
        gpus = [x.strip() for x in args.parallel_gpus.split(",") if x.strip()]
        metrics = [m for m in METRICS if os.path.exists(os.path.join(args.scores_dir, f"{m}.pt"))]
        if not metrics:
            sys.exit(f"No scores in {args.scores_dir}")

        cache = os.path.join(args.output_dir, f"_eval_cache_{args.task}.json")
        os.makedirs(args.output_dir, exist_ok=True)
        if not os.path.exists(cache):
            _log("Building eval cache …")
            tokenizer, _ = load_tokenizer_and_causal_lm(args.model)
            if args.task == "mmlu_acc":
                items = load_mmlu_test(
                    args.mmlu_subject,
                    None if args.mmlu_max_items <= 0 else args.mmlu_max_items,
                )
                with open(cache, "w", encoding="utf-8") as f:
                    json.dump({"items": items}, f)
            else:
                texts = load_c4_eval_texts(args.n_eval_texts, args.max_eval_chars)
                with open(cache, "w", encoding="utf-8") as f:
                    json.dump({"texts": texts}, f)
            del tokenizer, _

        env_base = os.environ.copy()
        subprocess.run(
            [
                sys.executable,
                script,
                "--task",
                args.task,
                "--scores_dir",
                args.scores_dir,
                "--output_dir",
                args.output_dir,
                "--deactivate_fracs",
                args.deactivate_fracs,
                "--mmlu_subject",
                args.mmlu_subject,
                "--mmlu_scoring",
                args.mmlu_scoring,
                "--mmlu_max_new_tokens",
                str(args.mmlu_max_new_tokens),
                "--mmlu_max_items",
                str(args.mmlu_max_items),
                "--n_eval_texts",
                str(args.n_eval_texts),
                "--eval_cache",
                cache,
                "--baseline_only",
            ],
            env={**env_base, "CUDA_VISIBLE_DEVICES": gpus[0]},
            check=True,
        )

        worker_gpus = gpus[-len(metrics) :] if len(gpus) > len(metrics) else gpus
        procs = []
        for i, metric in enumerate(metrics):
            gpu = worker_gpus[i]
            cmd = [
                script,
                "--task",
                args.task,
                "--scores_dir",
                args.scores_dir,
                "--output_dir",
                args.output_dir,
                "--deactivate_fracs",
                args.deactivate_fracs,
                "--mmlu_subject",
                args.mmlu_subject,
                "--mmlu_scoring",
                args.mmlu_scoring,
                "--mmlu_max_new_tokens",
                str(args.mmlu_max_new_tokens),
                "--eval_cache",
                cache,
                "--only_metric",
                metric,
            ]
            procs.append(
                subprocess.Popen(
                    [sys.executable, *cmd],
                    env={**env_base, "CUDA_VISIBLE_DEVICES": gpu},
                )
            )
        for proc in procs:
            if proc.wait() != 0:
                sys.exit(1)

        partials = {}
        for metric in metrics:
            path = os.path.join(args.output_dir, f"partial_{metric}.json")
            with open(path, encoding="utf-8") as f:
                partials[metric] = json.load(f)["rows"]
        bl_path = os.path.join(args.output_dir, "_baseline.json")
        with open(bl_path, encoding="utf-8") as f:
            bl = json.load(f)
        save_results(
            args.output_dir,
            args.task,
            fracs,
            bl["baseline"],
            partials,
            extra={
                "mmlu_subject": args.mmlu_subject,
                "mmlu_scoring": args.mmlu_scoring,
            }
            if args.task == "mmlu_acc"
            else {},
        )
        return

    # ── single process ──
    tokenizer, model = load_tokenizer_and_causal_lm(args.model)
    if args.eval_cache:
        with open(args.eval_cache, encoding="utf-8") as f:
            cache = json.load(f)
        eval_bundle = cache["items"] if args.task == "mmlu_acc" else cache["texts"]
    elif args.task == "mmlu_acc":
        eval_bundle = load_mmlu_test(
            args.mmlu_subject,
            None if args.mmlu_max_items <= 0 else args.mmlu_max_items,
        )
    else:
        eval_bundle = load_c4_eval_texts(args.n_eval_texts, args.max_eval_chars)

    if args.baseline_only:
        if args.task == "mmlu_acc":
            baseline, n = evaluate_mmlu_accuracy(
                model,
                tokenizer,
                eval_bundle,
                scoring=args.mmlu_scoring,
                max_new_tokens=args.mmlu_max_new_tokens,
            )
            _log(
                f"  full_model  acc={100 * baseline:.2f}%  (n={n}, scoring={args.mmlu_scoring})"
            )
        else:
            baseline = evaluate_perplexity(model, tokenizer, eval_bundle)
            _log(f"  full_model  PPL={baseline:.2f}")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "_baseline.json"), "w") as f:
            json.dump({"baseline": baseline}, f)
        return

    if args.only_metric:
        rows = run_metric_rows(
            model,
            tokenizer,
            args.scores_dir,
            args.only_metric,
            fracs,
            args.task,
            eval_bundle,
            mmlu_scoring=args.mmlu_scoring,
            mmlu_max_new_tokens=args.mmlu_max_new_tokens,
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"partial_{args.only_metric}.json"), "w") as f:
            json.dump({"metric": args.only_metric, "rows": rows}, f)
        return

    _log(f"Task={args.task}  scores={args.scores_dir}")
    if args.task == "mmlu_acc":
        baseline, n = evaluate_mmlu_accuracy(
            model,
            tokenizer,
            eval_bundle,
            scoring=args.mmlu_scoring,
            max_new_tokens=args.mmlu_max_new_tokens,
        )
        _log(
            f"  full_model  acc={100 * baseline:.2f}%  (n={n}, scoring={args.mmlu_scoring})"
        )
    else:
        baseline = evaluate_perplexity(model, tokenizer, eval_bundle)
        _log(f"  full_model  PPL={baseline:.2f}")

    all_rows = {}
    for metric in METRICS:
        p = os.path.join(args.scores_dir, f"{metric}.pt")
        if not os.path.exists(p):
            continue
        all_rows[metric] = run_metric_rows(
            model,
            tokenizer,
            args.scores_dir,
            metric,
            fracs,
            args.task,
            eval_bundle,
            mmlu_scoring=args.mmlu_scoring,
            mmlu_max_new_tokens=args.mmlu_max_new_tokens,
        )
    save_results(
        args.output_dir,
        args.task,
        fracs,
        baseline,
        all_rows,
        extra={
            "mmlu_subject": args.mmlu_subject,
            "mmlu_scoring": args.mmlu_scoring,
        }
        if args.task == "mmlu_acc"
        else {},
    )


if __name__ == "__main__":
    main()
