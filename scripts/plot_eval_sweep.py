#!/usr/bin/env python3
"""
Sweep bottom-neuron deactivation (freeze tail by importance score) and plot PPL.

Deactivation fractions: 0.05, 0.15, 0.25, …, 0.85 (step 0.10) — 9 points total.

Single GPU:
  python3 plot_eval_sweep.py --output_dir ./results_wikitext

Parallel (one metric per GPU; 4 metrics → one batch on 4+ GPUs):
  python3 plot_eval_sweep.py --output_dir ./results_wikitext --parallel_gpus 2,3,4,5,6,7
  bash run_sweep_parallel.sh

Outputs (under --output_dir):
  eval_sweep.csv, eval_sweep.json, eval_sweep_ppl.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEACTIVATE_FRACS_DEFAULT = [0.05] + [round(0.15 + 0.1 * i, 2) for i in range(8)]
# Fine grid: 0.05, 0.06, …, 0.10
DEACTIVATE_FRACS_FINE = [round(0.05 + 0.01 * i, 2) for i in range(6)]

EVAL_TEXTS_CACHE = "eval_texts_cache.json"
BASELINE_CACHE = "eval_sweep_baseline.json"
PARTIAL_PREFIX = "eval_sweep_partial_"


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_deactivate_fracs(spec: str | None) -> list[float]:
    if not spec or not spec.strip():
        return list(DEACTIVATE_FRACS_DEFAULT)
    fracs = [round(float(x.strip()), 4) for x in spec.split(",") if x.strip()]
    if not fracs:
        raise ValueError("empty --deactivate_fracs")
    for f in fracs:
        if not 0.0 < f < 1.0:
            raise ValueError(f"deactivate_frac must be in (0,1), got {f}")
    return fracs


def _cache_path(output_dir: str, name: str) -> str:
    return os.path.join(output_dir, name)


def _partial_path(output_dir: str, metric: str) -> str:
    return _cache_path(output_dir, f"{PARTIAL_PREFIX}{metric}.json")


def load_eval_texts(
    preset: str,
    n_texts: int,
    max_tokens: int,
) -> list[str]:
    if preset == "dummy":
        base = [
            "The capital of France is Paris. The Eiffel Tower is located there.",
            "Machine learning is a subset of artificial intelligence.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        return (base * ((n_texts + 2) // 3))[:n_texts]

    from datasets import load_dataset

    if preset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        texts: list[str] = []
        total_tokens = 0
        for row in ds:
            t = (row.get("text") or "").strip()
            if len(t) < 50:
                continue
            texts.append(t)
            total_tokens += len(t.split())
            if len(texts) >= n_texts or total_tokens >= max_tokens:
                break
        if not texts:
            raise ValueError("wikitext validation: no usable texts")
        return texts

    raise ValueError(f"Unknown eval preset: {preset}")


def ensure_eval_texts_cache(output_dir: str, preset: str, n_texts: int, max_tokens: int) -> list[str]:
    path = _cache_path(output_dir, EVAL_TEXTS_CACHE)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    texts = load_eval_texts(preset, n_texts, max_tokens)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(texts, f)
    return texts


def run_baseline(
    model,
    tokenizer,
    eval_texts: list[str],
    output_dir: str,
    deactivate_fracs: list[float],
) -> float:
    ppl = evaluate_perplexity(model, tokenizer, eval_texts)
    payload = {
        "baseline_ppl": ppl,
        "deactivate_fracs": deactivate_fracs,
        "n_eval_texts": len(eval_texts),
    }
    with open(_cache_path(output_dir, BASELINE_CACHE), "w", encoding="utf-8") as f:
        json.dump(payload, f)
    _log(f"  {'full_model':20s}  PPL = {ppl:.2f}  (no deactivation)")
    return ppl


def run_single_metric(
    model,
    tokenizer,
    eval_texts: list[str],
    scores_dir: str,
    output_dir: str,
    metric: str,
    deactivate_fracs: list[float],
) -> list[dict]:
    path = os.path.join(scores_dir, f"{metric}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    _log(f"  [{metric}] loading scores …")
    scores = load_scores(scores_dir, metric)
    _log(f"  [{metric}] ranking ~1M rows (once) …")
    ranked = build_global_score_ranking(scores)
    total = len(ranked)

    snap = snapshot_scored_weights(model, set(scores.keys()))
    metric_rows: list[dict] = []

    for frac in deactivate_fracs:
        top_k = 1.0 - frac
        n_keep = max(1, int(total * top_k))
        top_params = set(ranked[:n_keep])
        kept, total_rows = apply_inference_row_mask(model, scores, top_params)
        ppl = evaluate_perplexity(model, tokenizer, eval_texts)
        restore_scored_weights(model, snap)

        row = {
            "deactivate_frac": frac,
            "top_k": top_k,
            "ppl": ppl,
            "kept": kept,
            "total": total_rows,
        }
        metric_rows.append(row)
        _log(
            f"  {metric:20s}  freeze={frac:.2f}  top_k={top_k:.2f}  "
            f"PPL={ppl:.2f}  ({kept}/{total_rows} rows)"
        )

    partial = {"metric": metric, "rows": metric_rows}
    with open(_partial_path(output_dir, metric), "w", encoding="utf-8") as f:
        json.dump(partial, f, indent=2)
    _log(f"  [{metric}] wrote {_partial_path(output_dir, metric)}")
    return metric_rows


def run_sweep(
    model,
    tokenizer,
    eval_texts: list[str],
    scores_dir: str,
    output_dir: str,
    deactivate_fracs: list[float],
    metrics: tuple[str, ...] = METRICS,
) -> dict[str, list[dict]]:
    base_ppl = run_baseline(model, tokenizer, eval_texts, output_dir, deactivate_fracs)
    all_rows: dict[str, list[dict]] = {}
    for metric in metrics:
        p = os.path.join(scores_dir, f"{metric}.pt")
        if not os.path.exists(p):
            _log(f"  Skipping {metric} (no {p})")
            continue
        all_rows[metric] = run_single_metric(
            model, tokenizer, eval_texts, scores_dir, output_dir, metric, deactivate_fracs
        )
    all_rows["_baseline_ppl"] = base_ppl
    return all_rows


def load_baseline(output_dir: str) -> float | None:
    path = _cache_path(output_dir, BASELINE_CACHE)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("baseline_ppl")


def merge_partials(output_dir: str, metrics: tuple[str, ...] = METRICS) -> dict:
    baseline = load_baseline(output_dir)
    merged: dict[str, list[dict]] = {}
    for metric in metrics:
        path = _partial_path(output_dir, metric)
        if not os.path.exists(path):
            _log(f"  Missing partial: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            merged[metric] = json.load(f)["rows"]
    if baseline is not None:
        merged["_baseline_ppl"] = baseline
    return merged


def save_table(output_dir: str, deactivate_fracs: list[float], results: dict) -> None:
    csv_path = os.path.join(output_dir, "eval_sweep.csv")
    json_path = os.path.join(output_dir, "eval_sweep.json")

    baseline = results.get("_baseline_ppl", None)
    if isinstance(baseline, list):
        baseline = None
    serializable = {
        "baseline_ppl": baseline,
        "deactivate_fracs": deactivate_fracs,
        "metrics": {k: v for k, v in results.items() if not str(k).startswith("_")},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    fieldnames = ["deactivate_frac", "top_k", "metric", "ppl", "kept", "total"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if baseline is not None:
            w.writerow(
                {
                    "deactivate_frac": 0.0,
                    "top_k": 1.0,
                    "metric": "full_model",
                    "ppl": baseline,
                    "kept": "",
                    "total": "",
                }
            )
        for metric, rows in serializable["metrics"].items():
            for row in rows:
                w.writerow({"metric": metric, **row})

    _log(f"\nSaved {csv_path}")
    _log(f"Saved {json_path}")


COLORS = {
    "shapley_fisher": "#2563eb",
    "gradient": "#16a34a",
    "shapley_learned_a": "#9333ea",
    "shapley_learned_b": "#dc2626",
}
LABELS = {
    "shapley_fisher": "Shapley (Fisher)",
    "gradient": "Gradient",
    "shapley_learned_a": "Shapley (Learned-A)",
    "shapley_learned_b": "Shapley (Learned-B)",
}
# Main PPT figure: compare Fisher vs gradient in the low-damage regime
ZOOM_METRICS = ("shapley_fisher", "gradient")


def _plot_series(ax, results: dict, metrics: tuple[str, ...], deactivate_fracs: list[float]) -> None:
    for metric in metrics:
        rows = results.get(metric)
        if not rows:
            continue
        xs = [r["deactivate_frac"] for r in rows]
        ys = [r["ppl"] for r in rows]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            markersize=6,
            color=COLORS.get(metric),
            label=LABELS.get(metric, metric),
        )


def _save_fig(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _log(f"Saved {path}")


def plot_results(
    output_dir: str,
    deactivate_fracs: list[float],
    results: dict,
    baseline_ppl: float | None,
    log_y: bool,
    zoom_max_frac: float = 0.25,
    ylim_max: float | None = None,
) -> None:
    """
  Writes two PNGs by default:
    eval_sweep_ppl.png      — linear Y, Fisher + Gradient (readable low-PPL band)
    eval_sweep_ppl_log.png  — log Y, all metrics (full dynamic range)
    """
    xlabel = "Bottom neuron freeze fraction (deactivated by |score|)"

    # --- Zoomed linear plot (main figure for PPT) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_series(ax, results, ZOOM_METRICS, deactivate_fracs)
    if baseline_ppl is not None:
        ax.axhline(
            baseline_ppl,
            color="#64748b",
            linestyle="--",
            linewidth=1.5,
            label=f"Full model ({baseline_ppl:.1f})",
        )

    zoom_ppls: list[float] = []
    if baseline_ppl is not None:
        zoom_ppls.append(baseline_ppl)
    for metric in ZOOM_METRICS:
        for row in results.get(metric) or []:
            if row["deactivate_frac"] <= zoom_max_frac:
                zoom_ppls.append(row["ppl"])

    if zoom_ppls:
        y_hi = ylim_max if ylim_max is not None else max(zoom_ppls) * 1.12
        y_lo = max(0.0, min(zoom_ppls) * 0.85)
        ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Perplexity (PPL)")
    ax.set_title(
        f"Inference PPL (linear, freeze ≤ {zoom_max_frac:.2f} sets Y range; Fisher vs Gradient)"
    )
    ax.set_xticks(deactivate_fracs)
    ax.set_xticklabels([f"{x:.2f}" for x in deactivate_fracs], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    _save_fig(fig, os.path.join(output_dir, "eval_sweep_ppl.png"))

    # --- Full log-scale plot (all metrics) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_series(ax, results, METRICS, deactivate_fracs)
    if baseline_ppl is not None:
        ax.axhline(
            baseline_ppl,
            color="#64748b",
            linestyle="--",
            linewidth=1.5,
            label=f"Full model ({baseline_ppl:.1f})",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Perplexity (PPL)")
    ax.set_title("Inference PPL vs bottom-neuron deactivation (all metrics, log scale)")
    ax.set_xticks(deactivate_fracs)
    ax.set_xticklabels([f"{x:.2f}" for x in deactivate_fracs], rotation=45)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    _save_fig(fig, os.path.join(output_dir, "eval_sweep_ppl_log.png"))

    # Optional: single combined figure with user-requested log_y only (legacy flag)
    if log_y:
        return


def _metrics_available(scores_dir: str) -> list[str]:
    return [m for m in METRICS if os.path.exists(os.path.join(scores_dir, f"{m}.pt"))]


def _worker_argv(args: argparse.Namespace, metric: str) -> list[str]:
    fracs_csv = ",".join(str(f) for f in args.deactivate_fracs)
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--output_dir",
        args.output_dir,
        "--scores_dir",
        args.scores_dir,
        "--model",
        args.model,
        "--eval_preset",
        args.eval_preset,
        "--n_eval_texts",
        str(args.n_eval_texts),
        "--max_eval_tokens",
        str(args.max_eval_tokens),
        "--deactivate_fracs",
        fracs_csv,
        "--only_metric",
        metric,
    ]


def run_parallel(args: argparse.Namespace) -> None:
    gpus = [int(x.strip()) for x in args.parallel_gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--parallel_gpus must list at least one GPU id, e.g. 2,3")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics = _metrics_available(args.scores_dir)
    if not metrics:
        raise FileNotFoundError(f"No metric .pt files in {args.scores_dir}")

    _log("Deactivation fractions: " + str(args.deactivate_fracs))
    _log(f"Parallel GPUs: {gpus}  metrics: {metrics}")

    # Cache eval texts once (no model)
    texts = ensure_eval_texts_cache(
        args.output_dir, args.eval_preset, args.n_eval_texts, args.max_eval_tokens
    )
    _log(f"  {len(texts)} eval texts cached\n")

    script = str(Path(__file__).resolve())
    env_base = os.environ.copy()

    # Baseline on first GPU
    gpu0 = str(gpus[0])
    _log(f"[GPU {gpu0}] baseline PPL …")
    env = {**env_base, "CUDA_VISIBLE_DEVICES": gpu0}
    subprocess.run(
        [
            sys.executable,
            script,
            "--output_dir",
            args.output_dir,
            "--model",
            args.model,
            "--baseline_only",
            "--deactivate_fracs",
            ",".join(str(f) for f in args.deactivate_fracs),
        ],
        env=env,
        check=True,
    )

    # At most one 3B model per GPU. Prefer higher-id GPUs for workers (e.g. 4–7)
    # when the list is longer than the batch (baseline uses gpus[0]).
    failed: list[tuple[str, str, int]] = []
    for batch_start in range(0, len(metrics), len(gpus)):
        batch = metrics[batch_start : batch_start + len(gpus)]
        worker_gpus = gpus[-len(batch) :] if len(gpus) > len(batch) else gpus
        procs: list[tuple[str, str, subprocess.Popen]] = []
        for i, metric in enumerate(batch):
            gpu = str(worker_gpus[i])
            _log(f"[GPU {gpu}] starting {metric} …")
            env = {**env_base, "CUDA_VISIBLE_DEVICES": gpu}
            p = subprocess.Popen(_worker_argv(args, metric), env=env)
            procs.append((gpu, metric, p))
        for gpu, metric, p in procs:
            rc = p.wait()
            if rc != 0:
                failed.append((gpu, metric, rc))
                _log(f"[GPU {gpu}] {metric} FAILED (exit {rc})")
            else:
                _log(f"[GPU {gpu}] {metric} done")

    if failed:
        raise RuntimeError(f"Workers failed: {failed}")

    _log("\nMerging partial results …")
    results = merge_partials(args.output_dir)
    baseline = results.pop("_baseline_ppl", None)
    save_table(args.output_dir, args.deactivate_fracs, {"_baseline_ppl": baseline, **results})
    if not args.no_plot:
        _plot_outputs(args, results, baseline)


def _plot_outputs(args: argparse.Namespace, results: dict, baseline: float | None) -> None:
    plot_results(
        args.output_dir,
        args.deactivate_fracs,
        results,
        baseline,
        args.log_y,
        zoom_max_frac=args.zoom_max_frac,
        ylim_max=args.ylim_max,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sweep bottom-neuron freeze fractions and plot PPL curves."
    )
    p.add_argument("--model", default=_DEFAULT_MODEL)
    p.add_argument("--output_dir", default="./results_wikitext")
    p.add_argument(
        "--scores_dir",
        default="",
        help="Directory with *.pt scores (default: same as --output_dir).",
    )
    p.add_argument(
        "--deactivate_fracs",
        type=str,
        default="",
        help=(
            "Comma-separated bottom-freeze fractions, e.g. 0.05,0.06,...,0.10. "
            f"Default: coarse grid {DEACTIVATE_FRACS_DEFAULT}. "
            f"Preset fine: {','.join(str(x) for x in DEACTIVATE_FRACS_FINE)}"
        ),
    )
    p.add_argument(
        "--eval_preset",
        choices=("wikitext", "dummy"),
        default="wikitext",
    )
    p.add_argument("--n_eval_texts", type=int, default=256)
    p.add_argument("--max_eval_tokens", type=int, default=50_000)
    p.add_argument(
        "--log_y",
        action="store_true",
        help="(Legacy) log plot is always written as eval_sweep_ppl_log.png.",
    )
    p.add_argument(
        "--zoom_max_frac",
        type=float,
        default=0.25,
        help="For linear zoom plot: Y max from points with freeze frac <= this (default 0.25).",
    )
    p.add_argument(
        "--ylim_max",
        type=float,
        default=None,
        help="Cap linear zoom Y axis (e.g. 200 to focus on 0.05–0.15).",
    )
    p.add_argument("--no_plot", action="store_true")
    p.add_argument(
        "--parallel_gpus",
        type=str,
        default="",
        help="Comma-separated GPU ids (e.g. 2,3,4,5,6,7). One metric per GPU; 4 metrics fit in one batch.",
    )
    p.add_argument(
        "--only_metric",
        choices=METRICS,
        default="",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--baseline_only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--merge_only",
        action="store_true",
        help="Merge eval_sweep_partial_*.json and plot (no model).",
    )
    args = p.parse_args()
    if not args.scores_dir:
        args.scores_dir = args.output_dir
    args.deactivate_fracs = parse_deactivate_fracs(args.deactivate_fracs)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.merge_only:
        results = merge_partials(args.output_dir)
        baseline = results.pop("_baseline_ppl", None)
        fracs = args.deactivate_fracs
        bl_path = _cache_path(args.output_dir, BASELINE_CACHE)
        if os.path.exists(bl_path):
            with open(bl_path, encoding="utf-8") as f:
                fracs = json.load(f).get("deactivate_fracs", fracs)
        save_table(args.output_dir, fracs, {"_baseline_ppl": baseline, **results})
        if not args.no_plot:
            _plot_outputs(args, results, baseline)
        return

    if args.parallel_gpus:
        run_parallel(args)
        return

    eval_texts = ensure_eval_texts_cache(
        args.output_dir, args.eval_preset, args.n_eval_texts, args.max_eval_tokens
    )

    _log(f"Loading model: {args.model}")
    tokenizer, model = load_tokenizer_and_causal_lm(args.model)
    _log(f"  {len(eval_texts)} eval texts\n")

    if args.baseline_only:
        run_baseline(model, tokenizer, eval_texts, args.output_dir, args.deactivate_fracs)
        return

    if args.only_metric:
        _log(f"Worker metric: {args.only_metric}")
        run_single_metric(
            model,
            tokenizer,
            eval_texts,
            args.scores_dir,
            args.output_dir,
            args.only_metric,
            args.deactivate_fracs,
        )
        return

    _log("Running sweep (single GPU) …")
    results = run_sweep(
        model, tokenizer, eval_texts, args.scores_dir, args.output_dir, args.deactivate_fracs
    )
    baseline = results.pop("_baseline_ppl", None)
    save_table(args.output_dir, args.deactivate_fracs, {"_baseline_ppl": baseline, **results})
    if not args.no_plot:
        _plot_outputs(args, results, baseline)


if __name__ == "__main__":
    main()
