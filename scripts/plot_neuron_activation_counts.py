#!/usr/bin/env python3
"""
Model Shapley Figure 2: neuron activation counts per layer on two benchmarks.

Fixed threshold (paper-style):
  MMLU Shapley > 5, LM tasks Shapley > 0.5

Global top-p% (fair across metrics — same p, each metric sets its own cutoff):
  per layer: #{neurons with |φ_i| in global top p% for that metric}

Examples:

  # Paper fixed τ
  python3 plot_neuron_activation_counts.py \\
    --panel mmlu:./results_mmlu:5 --panel wikitext:./results_wikitext:0.5 \\
    --metrics shapley_fisher,shapley_learned_a,shapley_learned_b,gradient

  # Top 1% |score|, four metrics comparable
  python3 plot_neuron_activation_counts.py --top_pct 1 \\
    --panel mmlu:./results_mmlu --panel c4:./results_c4 \\
    --metrics shapley_fisher,shapley_learned_a,shapley_learned_b,gradient \\
    --out_dir ./results_fig2
"""
from __future__ import annotations

import argparse
import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))

METRICS = ("shapley_fisher", "shapley_learned_a", "shapley_learned_b", "gradient")

DEFAULT_THRESHOLDS = {"mmlu": 5.0, "gsm8k": 0.5, "wikitext": 0.5, "c4": 0.5}

PANEL_TITLES = {
    "mmlu": "(a) MMLU",
    "gsm8k": "(b) GSM8K",
    "wikitext": "(b) WikiText",
    "c4": "(b) C4",
}

METRIC_LABELS = {
    "shapley_fisher": "Shapley (Fisher)",
    "shapley_learned_a": "Shapley (Learned-A)",
    "shapley_learned_b": "Shapley (Learned-B)",
    "gradient": "Gradient",
}

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_scores(scores_dir: str, metric: str) -> dict:
    path = os.path.join(scores_dir, f"{metric}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No scores at {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def count_active_neurons_per_layer(
    scores: dict,
    threshold: float,
    *,
    use_abs: bool = False,
) -> tuple[np.ndarray, float]:
    """Count neurons with score > threshold (or |score| > threshold)."""
    per_layer: dict[int, int] = {}
    max_layer = -1
    for name, val in scores.items():
        m = LAYER_RE.search(name)
        if not m:
            continue
        layer = int(m.group(1))
        max_layer = max(max_layer, layer)
        t = val.float().view(-1)
        if use_abs:
            n = int((t.abs() > threshold).sum().item())
        else:
            n = int((t > threshold).sum().item())
        per_layer[layer] = per_layer.get(layer, 0) + n

    if max_layer < 0:
        raise ValueError("No model.layers.* parameters found in scores.")

    n_layers = max_layer + 1
    counts = np.zeros(n_layers, dtype=np.int64)
    for layer, n in per_layer.items():
        counts[layer] = n
    return counts, threshold


def count_top_pct_per_layer(
    scores: dict,
    top_pct: float,
    *,
    use_abs: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Global top-p% by |score| (or signed score if use_abs=False).
    Returns (counts per layer, cutoff value used).
    """
    if not 0 < top_pct < 100:
        raise ValueError(f"top_pct must be in (0, 100), got {top_pct}")

    flat: list[float] = []
    layers: list[int] = []
    for name, val in scores.items():
        m = LAYER_RE.search(name)
        if not m:
            continue
        layer = int(m.group(1))
        t = val.float().view(-1)
        if use_abs:
            t = t.abs()
        for v in t.tolist():
            flat.append(float(v))
            layers.append(layer)

    if not flat:
        raise ValueError("No model.layers.* parameters found in scores.")

    # Top p%: scores >= (100-p) percentile
    q = 1.0 - top_pct / 100.0
    cutoff = float(np.quantile(flat, q))
    per_layer: dict[int, int] = {}
    for v, layer in zip(flat, layers):
        if v >= cutoff:
            per_layer[layer] = per_layer.get(layer, 0) + 1

    max_layer = max(per_layer) if per_layer else max(layers)
    n_layers = max_layer + 1
    counts = np.zeros(n_layers, dtype=np.int64)
    for layer, n in per_layer.items():
        counts[layer] = n
    return counts, cutoff


def parse_panel(spec: str, *, top_pct_mode: bool) -> tuple[str, str, float | None]:
    parts = [p.strip() for p in spec.split(":") if p.strip()]
    if len(parts) < 2:
        raise ValueError(
            f"Invalid --panel {spec!r}; use name:scores_dir[:threshold]"
        )
    name = parts[0].lower()
    scores_dir = os.path.abspath(parts[1])
    if top_pct_mode:
        return name, scores_dir, None
    if len(parts) >= 3:
        return name, scores_dir, float(parts[2])
    threshold = DEFAULT_THRESHOLDS.get(name)
    if threshold is None:
        raise ValueError(f"No default threshold for {name!r}; add :threshold")
    return name, scores_dir, threshold


def _series_label(panel_name: str, metric: str, model_tag: str) -> str:
    metric_l = METRIC_LABELS.get(metric, metric)
    if model_tag:
        return f"{model_tag} / {panel_name} ({metric_l})"
    return f"{panel_name} ({metric_l})"


def plot_figure2(
    panels: list[tuple[str, str, float | None]],
    *,
    metrics: list[str],
    top_pct: float | None,
    use_abs: bool,
    log_y: bool,
    model_tag: str,
    out_path: str,
    suptitle: str,
) -> list[dict]:
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.2), squeeze=False)
    axes = axes[0]

    csv_rows: list[dict] = []
    colors = plt.cm.tab10.colors
    rank_by_abs = top_pct is not None or use_abs

    for pidx, (panel_name, scores_dir, fixed_threshold) in enumerate(panels):
        ax = axes[pidx]
        title = PANEL_TITLES.get(panel_name, f"({chr(97 + pidx)}) {panel_name.upper()}")

        if top_pct is not None:
            ylab = f"# neurons in global top {top_pct:g}%"
            ylab += " |score|" if rank_by_abs else " score"
        elif use_abs:
            ylab = f"# neurons with |Shapley| > {fixed_threshold:g}"
        else:
            ylab = f"# neurons with Shapley > {fixed_threshold:g}"

        for midx, metric in enumerate(metrics):
            scores = load_scores(scores_dir, metric)
            if top_pct is not None:
                counts, cutoff = count_top_pct_per_layer(
                    scores, top_pct, use_abs=rank_by_abs
                )
                mode = "top_pct"
                thr_record = top_pct
                cutoff_record = cutoff
            else:
                counts, cutoff = count_active_neurons_per_layer(
                    scores, fixed_threshold, use_abs=use_abs
                )
                mode = "fixed"
                thr_record = fixed_threshold
                cutoff_record = cutoff

            ax.plot(
                np.arange(len(counts)),
                counts,
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=colors[midx % len(colors)],
                label=_series_label(panel_name, metric, model_tag),
            )
            for layer, c in enumerate(counts):
                csv_rows.append(
                    {
                        "panel": panel_name,
                        "scores_dir": scores_dir,
                        "metric": metric,
                        "mode": mode,
                        "top_pct": thr_record if mode == "top_pct" else "",
                        "threshold": thr_record if mode == "fixed" else "",
                        "cutoff": cutoff_record,
                        "layer": layer,
                        "active_neurons": int(c),
                    }
                )
            _log(
                f"{panel_name} {metric}: "
                f"{'top' + str(top_pct) + '% cutoff=' if top_pct else 'τ='}"
                f"{cutoff_record:.6g}, total={int(counts.sum())}, "
                f"peak L{int(counts.argmax())}={int(counts.max())}"
            )

        ax.set_xlabel("Layer index")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        if log_y:
            ax.set_yscale("log")
            if (csv_rows and max(r["active_neurons"] for r in csv_rows if r["panel"] == panel_name) > 0):
                ax.set_ylim(bottom=0.8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(suptitle, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved {out_path}")
    return csv_rows


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    fields = [
        "panel",
        "scores_dir",
        "metric",
        "mode",
        "top_pct",
        "threshold",
        "cutoff",
        "layer",
        "active_neurons",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    _log(f"Saved {path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Neuron activation counts per layer (fixed τ or global top-p%)."
    )
    p.add_argument("--panel", action="append", default=[], metavar="NAME:DIR[:THRESH]")
    p.add_argument("--metric", default="shapley_fisher")
    p.add_argument("--metrics", default="", help="Comma-separated metrics for one figure")
    p.add_argument(
        "--top_pct",
        type=float,
        default=None,
        metavar="P",
        help="Use global top P%% per metric (e.g. 1 or 5); same P for all metrics, "
        "each metric has its own |score| cutoff",
    )
    p.add_argument(
        "--use_abs",
        action="store_true",
        help="Fixed-τ mode: |Shapley| > τ. Top-p%% mode already uses |score| by default.",
    )
    p.add_argument("--no_log_y", action="store_true")
    p.add_argument("--model_tag", default="Qwen2.5-3B-Instruct")
    p.add_argument("--out_dir", default=".")
    args = p.parse_args()

    metrics = (
        [m.strip() for m in args.metrics.split(",") if m.strip()]
        if args.metrics.strip()
        else [args.metric]
    )
    for m in metrics:
        if m not in METRICS:
            raise ValueError(f"Unknown metric {m!r}")

    top_pct_mode = args.top_pct is not None
    if args.panel:
        panels = [parse_panel(s, top_pct_mode=top_pct_mode) for s in args.panel]
    elif top_pct_mode:
        panels = [
            ("mmlu", os.path.join(_SCRIPTS, "results_mmlu"), None),
            ("wikitext", os.path.join(_SCRIPTS, "results_wikitext"), None),
        ]
        _log(f"No --panel; default MMLU + WikiText (top {args.top_pct}%)")
    else:
        panels = [
            ("mmlu", os.path.join(_SCRIPTS, "results_mmlu"), 5.0),
            ("wikitext", os.path.join(_SCRIPTS, "results_wikitext"), 0.5),
        ]
        _log("No --panel; default MMLU (τ=5) + WikiText (τ=0.5)")

    for _, scores_dir, _ in panels:
        if not os.path.isdir(scores_dir):
            raise FileNotFoundError(f"scores_dir not found: {scores_dir}")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if top_pct_mode:
        pct_tag = str(args.top_pct).replace(".", "p")
        out_png = os.path.join(out_dir, f"neuron_activation_counts_top{pct_tag}pct.png")
        out_csv = os.path.join(out_dir, f"neuron_activation_counts_top{pct_tag}pct.csv")
        suptitle = (
            f"Neuron counts per layer (global top {args.top_pct:g}% |score|, per metric)"
        )
    else:
        out_png = os.path.join(out_dir, "neuron_activation_counts.png")
        out_csv = os.path.join(out_dir, "neuron_activation_counts.csv")
        suptitle = "Neuron activation counts per layer (Model Shapley Fig. 2)"

    csv_rows = plot_figure2(
        panels,
        metrics=metrics,
        top_pct=args.top_pct,
        use_abs=args.use_abs,
        log_y=not args.no_log_y,
        model_tag=args.model_tag,
        out_path=out_png,
        suptitle=suptitle,
    )
    write_csv(csv_rows, out_csv)


if __name__ == "__main__":
    main()
