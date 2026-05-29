#!/usr/bin/env python3
"""
Plot PPL curves from eval_sweep.json and/or eval_sweep_partial_*.json (no torch).

  python3 plot_eval_sweep_from_json.py --output_dir ./results_wikitext_fine

Writes (under output_dir):
  eval_sweep_ppl.png       — 4 curves + baseline (linear)
  eval_sweep_ppl_log.png   — log scale
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

METRICS = ("shapley_fisher", "shapley_learned_a", "shapley_learned_b", "gradient")
PARTIAL_PREFIX = "eval_sweep_partial_"
BASELINE_FILE = "eval_sweep_baseline.json"

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


def _log(msg: str) -> None:
    print(msg, flush=True)


def _plot_series(ax, metrics_data: dict, metrics: tuple[str, ...]) -> None:
    for metric in metrics:
        rows = metrics_data.get(metric)
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


def _style_ax(
    ax,
    deactivate_fracs: list[float],
    xlabel: str,
    title: str,
    baseline: float | None,
) -> None:
    if baseline is not None:
        ax.axhline(
            baseline,
            color="#64748b",
            linestyle="--",
            linewidth=1.5,
            label=f"Full model ({baseline:.1f})",
        )
    ax.set_ylabel("Perplexity (PPL)")
    ax.set_title(title)
    ax.set_xticks(deactivate_fracs)
    ax.set_xticklabels([f"{x:.2f}" for x in deactivate_fracs], rotation=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def load_from_partials(output_dir: str) -> dict | None:
    """Build sweep dict from eval_sweep_partial_<metric>.json (fine-run source of truth)."""
    metrics_data: dict[str, list] = {}
    deactivate_fracs: list[float] | None = None
    for metric in METRICS:
        path = os.path.join(output_dir, f"{PARTIAL_PREFIX}{metric}.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            block = json.load(f)
        rows = block["rows"]
        metrics_data[metric] = rows
        fracs = [r["deactivate_frac"] for r in rows]
        if deactivate_fracs is None:
            deactivate_fracs = fracs
        elif fracs != deactivate_fracs:
            _log(f"WARNING: {metric} freeze fracs differ from first metric")

    baseline = None
    bl_path = os.path.join(output_dir, BASELINE_FILE)
    if os.path.exists(bl_path):
        with open(bl_path, encoding="utf-8") as f:
            baseline = json.load(f).get("baseline_ppl")

    return {
        "baseline_ppl": baseline,
        "deactivate_fracs": deactivate_fracs or [],
        "metrics": metrics_data,
        "_source": "partials",
    }


def load_sweep_data(output_dir: str, prefer_partials: bool) -> dict:
    output_dir = os.path.abspath(output_dir)
    partial_data = load_from_partials(output_dir)
    json_path = os.path.join(output_dir, "eval_sweep.json")

    if prefer_partials and partial_data is not None:
        _log(f"Data source: partial JSON files in {output_dir}")
        return partial_data

    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        data["_source"] = json_path
        fracs = data.get("deactivate_fracs", [])
        # If JSON is coarse but fine partials exist, prefer partials
        if partial_data is not None and fracs and max(fracs) > 0.11:
            n_partial = len(partial_data["deactivate_fracs"])
            _log(
                f"WARNING: {json_path} has coarse grid (max freeze={max(fracs):.2f}). "
                f"Using {n_partial} fine points from partials instead."
            )
            return partial_data
        _log(f"Data source: {json_path}")
        return data

    if partial_data is not None:
        _log(f"Data source: partial JSON files (no eval_sweep.json)")
        return partial_data

    print(f"ERROR: no eval_sweep.json or {PARTIAL_PREFIX}*.json in {output_dir}", file=sys.stderr)
    sys.exit(1)


def _frac_range_label(fracs: list[float]) -> str:
    if not fracs:
        return ""
    return f"{min(fracs):.2f}–{max(fracs):.2f}"


def plot_from_json(
    output_dir: str,
    prefer_partials: bool = True,
    ylim_max: float | None = None,
) -> None:
    output_dir = os.path.abspath(output_dir)
    data = load_sweep_data(output_dir, prefer_partials)
    baseline = data.get("baseline_ppl")
    deactivate_fracs = data["deactivate_fracs"]
    metrics_data = data["metrics"]

    _log(f"Freeze fractions ({len(deactivate_fracs)} points): {deactivate_fracs}")

    xlabel = "Bottom neuron freeze fraction (deactivated by |score|)"
    range_lbl = _frac_range_label(deactivate_fracs)

    # --- Linear: 4 curves ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _plot_series(ax, metrics_data, METRICS)
    ppls: list[float] = []
    for metric in METRICS:
        for row in metrics_data.get(metric) or []:
            ppls.append(row["ppl"])
    if baseline is not None:
        ppls.append(baseline)
    if ppls:
        y_hi = ylim_max if ylim_max is not None else max(ppls) * 1.08
        y_lo = max(0.0, min(ppls) * 0.92)
        ax.set_ylim(y_lo, y_hi)

    _style_ax(
        ax,
        deactivate_fracs,
        xlabel,
        f"Inference PPL (freeze {range_lbl})",
        baseline,
    )
    ax.set_xlabel(xlabel)

    out_main = os.path.join(output_dir, "eval_sweep_ppl.png")
    fig.tight_layout()
    fig.savefig(out_main, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved {out_main}")

    # --- Log scale ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _plot_series(ax, metrics_data, METRICS)
    _style_ax(ax, deactivate_fracs, xlabel, f"Inference PPL log scale (freeze {range_lbl})", baseline)
    ax.set_xlabel(xlabel)
    ax.set_yscale("log")
    out_log = os.path.join(output_dir, "eval_sweep_ppl_log.png")
    fig.tight_layout()
    fig.savefig(out_log, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved {out_log}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot PPL curves from saved sweep JSON.")
    p.add_argument(
        "--output_dir",
        default="./results_wikitext_fine",
        help="Directory with eval_sweep.json or eval_sweep_partial_*.json",
    )
    p.add_argument(
        "--json_only",
        action="store_true",
        help="Use eval_sweep.json only (do not prefer partials).",
    )
    p.add_argument("--ylim_max", type=float, default=None)
    args = p.parse_args()
    plot_from_json(args.output_dir, prefer_partials=not args.json_only, ylim_max=args.ylim_max)


if __name__ == "__main__":
    main()
