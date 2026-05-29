#!/usr/bin/env python3
"""
Plot downstream eval JSON (C4 PPL / MMLU acc).

  python3 plot_downstream_results.py --input ./results_c4_ppl
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt

METRICS = ("shapley_fisher", "shapley_learned_a", "shapley_learned_b", "gradient")
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


def _find_eval_json(input_dir: str) -> tuple[str, dict]:
    for name in sorted(os.listdir(input_dir)):
        if name.startswith("eval_") and name.endswith(".json"):
            path = os.path.join(input_dir, name)
            with open(path, encoding="utf-8") as f:
                return path, json.load(f)
    raise FileNotFoundError(f"No eval_*.json in {input_dir}")


def _value_key(task: str) -> str:
    return "accuracy" if task == "mmlu_acc" else "ppl"


def _ylabel(task: str) -> str:
    return "Accuracy" if task == "mmlu_acc" else "Perplexity (PPL)"


def _filter_rows(rows: list[dict], min_frac: float | None, max_frac: float | None) -> list[dict]:
    out = rows
    if min_frac is not None:
        out = [r for r in out if r["deactivate_frac"] >= min_frac - 1e-9]
    if max_frac is not None:
        out = [r for r in out if r["deactivate_frac"] <= max_frac + 1e-9]
    return out


def _plot_series(
    ax,
    metrics_data: dict,
    metrics: tuple[str, ...],
    ykey: str,
    min_frac: float | None = None,
    max_frac: float | None = None,
) -> None:
    for metric in metrics:
        rows = metrics_data.get(metric)
        if not rows:
            continue
        rows = _filter_rows(rows, min_frac, max_frac)
        if not rows:
            continue
        xs = [r["deactivate_frac"] for r in rows]
        ys = [r[ykey] for r in rows]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            markersize=6,
            color=COLORS.get(metric),
            label=LABELS.get(metric, metric),
        )


def plot(
    input_dir: str,
    min_frac: float | None = None,
    max_frac: float | None = None,
    out_name: str = "",
) -> None:
    input_dir = os.path.abspath(input_dir)
    eval_path, data = _find_eval_json(input_dir)
    task = data["task"]
    ykey = _value_key(task)
    fracs = _filter_rows(
        [{"deactivate_frac": f} for f in data["deactivate_fracs"]],
        min_frac,
        max_frac,
    )
    fracs = [r["deactivate_frac"] for r in fracs]
    baseline = data["baseline"]
    metrics_data = data["metrics"]
    stem = os.path.splitext(os.path.basename(eval_path))[0]  # eval_c4_ppl

    range_lbl = f"{min(fracs):.2f}–{max(fracs):.2f}" if fracs else ""
    ylabel = _ylabel(task)
    unit = "acc" if task == "mmlu_acc" else "PPL"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    _plot_series(ax, metrics_data, METRICS, ykey, min_frac, max_frac)
    if task == "mmlu_acc":
        ys = [
            r[ykey]
            for rows in metrics_data.values()
            for r in _filter_rows(rows or [], min_frac, max_frac)
        ]
        if baseline is not None:
            ys.append(baseline)
        if ys:
            lo, hi = min(ys), max(ys)
            pad = max(0.03, (hi - lo) * 0.15)
            ax.set_ylim(max(0, lo - pad), min(1.05, hi + pad))
        else:
            ax.set_ylim(0, 1.05)
    else:
        ppls = [
            r["ppl"]
            for rows in metrics_data.values()
            for r in _filter_rows(rows or [], min_frac, max_frac)
        ]
        if baseline is not None:
            ppls.append(baseline)
        if ppls:
            ax.set_ylim(max(0, min(ppls) * 0.92), max(ppls) * 1.08)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Downstream {unit} (freeze {range_lbl})")
    ax.set_xlabel("Bottom neuron freeze fraction (deactivated by |score|)")
    ax.set_xticks(fracs)
    ax.set_xticklabels([f"{x:.2f}" for x in fracs])
    ax.grid(True, alpha=0.3)
    if baseline is not None:
        ax.axhline(
            baseline,
            color="#64748b",
            linestyle="--",
            linewidth=1.5,
            label=f"Full model ({baseline:.2f})",
        )
    ax.legend(loc="best", fontsize=9)
    if out_name:
        out_main = os.path.join(input_dir, out_name)
    elif min_frac is not None or max_frac is not None:
        lo = f"{min(fracs):.2f}" if fracs else "min"
        hi = f"{max(fracs):.2f}" if fracs else "max"
        out_main = os.path.join(input_dir, f"{stem}_f{lo}-{hi}.png")
    else:
        out_main = os.path.join(input_dir, f"{stem}.png")
    fig.tight_layout()
    fig.savefig(out_main, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved {out_main}")

    if task == "c4_ppl" and min_frac is None and max_frac is None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        _plot_series(ax, metrics_data, METRICS, ykey)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Downstream PPL log scale (freeze {range_lbl})")
        ax.set_xlabel("Bottom neuron freeze fraction (deactivated by |score|)")
        ax.set_xticks(fracs)
        ax.set_xticklabels([f"{x:.2f}" for x in fracs])
        ax.grid(True, alpha=0.3)
        if baseline is not None:
            ax.axhline(
                baseline,
                color="#64748b",
                linestyle="--",
                linewidth=1.5,
                label=f"Full model ({baseline:.1f})",
            )
        ax.legend(loc="best", fontsize=9)
        out_log = os.path.join(input_dir, f"{stem}_log.png")
        fig.tight_layout()
        fig.savefig(out_log, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _log(f"Saved {out_log}")

    bar_targets = [f for f in (0.05, 0.10) if fracs and min(fracs) - 1e-9 <= f <= max(fracs) + 1e-9]
    if min_frac is None and max_frac is None:
        bar_targets = [0.05, 0.10]
    for f_target in bar_targets:
        row_by_metric = {}
        for metric in METRICS:
            rows = metrics_data.get(metric) or []
            for r in rows:
                if abs(r["deactivate_frac"] - f_target) < 1e-6:
                    row_by_metric[metric] = r[ykey]
                    break
        if not row_by_metric:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        names = [LABELS.get(m, m) for m in row_by_metric]
        vals = list(row_by_metric.values())
        colors = [COLORS.get(m, "#333") for m in row_by_metric]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"@ freeze {f_target:.2f}")
        if baseline is not None:
            ax.axhline(baseline, color="#64748b", linestyle="--", linewidth=1.2)
        fig.tight_layout()
        bar_path = os.path.join(input_dir, f"{stem}_bar_f{f_target:.2f}.png")
        fig.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _log(f"Saved {bar_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Directory with eval_<task>.json")
    p.add_argument(
        "--min_frac",
        type=float,
        default=None,
        help="Only plot deactivate_frac >= this (e.g. 0.05).",
    )
    p.add_argument(
        "--max_frac",
        type=float,
        default=None,
        help="Only plot deactivate_frac <= this (e.g. 0.08).",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output PNG filename (default: auto from frac range).",
    )
    args = p.parse_args()
    plot(args.input, min_frac=args.min_frac, max_frac=args.max_frac, out_name=args.out)


if __name__ == "__main__":
    main()
