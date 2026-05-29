#!/usr/bin/env python3
"""
Model Shapley–style heatmap: mean |score| per layer × projection.

  # Attention Q/K/V/O (default)
  python3 plot_shapley_heatmap.py --scores_dir ./results_c4 --metrics shapley_fisher,gradient

  # FFN: one PNG per metric (independent color scale; metrics differ in magnitude)
  python3 plot_shapley_heatmap.py --scores_dir ./results_c4 --block ffn --metrics shapley_fisher,gradient

  # Attention combined + FFN per-metric
  python3 plot_shapley_heatmap.py --scores_dir ./results_mmlu --block both --metrics shapley_fisher,...

Writes:
  heatmap_combined.png + heatmap_<metric>.png (optional)     (--block attn)
  heatmap_ffn_<metric>.png only (no 4-in-1 combined)         (--block ffn)
"""
from __future__ import annotations

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

METRICS = ("shapley_fisher", "shapley_learned_a", "shapley_learned_b", "gradient")


def load_scores(scores_dir: str, metric: str) -> dict:
    path = os.path.join(scores_dir, f"{metric}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No scores at {path}")
    return torch.load(path, map_location="cpu", weights_only=True)

ATTN_RE = re.compile(
    r"model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$"
)
FFN_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$"
)

BLOCK_CONFIG = {
    "attn": {
        "regex": ATTN_RE,
        "proj_order": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "proj_labels": ("Q", "K", "V", "O"),
        "xlabel": "Attention projection",
        "axis_short": "Q / K / V / O",
        "title_suffix": "Q/K/V/O",
        "out_prefix": "heatmap",
        "combined_name": "heatmap_combined.png",
    },
    "ffn": {
        "regex": FFN_RE,
        "proj_order": ("gate_proj", "up_proj", "down_proj"),
        "proj_labels": ("Gate", "Up", "Down"),
        "xlabel": "FFN projection",
        "axis_short": "Gate / Up / Down",
        "title_suffix": "Gate/Up/Down",
        "out_prefix": "heatmap_ffn",
        "combined_name": None,  # FFN: per-metric only (scales differ across methods)
    },
}

METRIC_LABELS = {
    "shapley_fisher": "Shapley (Fisher)",
    "shapley_learned_a": "Shapley (Learned-A)",
    "shapley_learned_b": "Shapley (Learned-B)",
    "gradient": "Gradient",
}

TASK_LABELS = {
    "results_wikitext": "WikiText",
    "results_c4": "C4",
    "results_mmlu": "MMLU",
}

DEFAULT_METRICS = ("shapley_fisher", "shapley_learned_a", "shapley_learned_b", "gradient")


def _log(msg: str) -> None:
    print(msg, flush=True)


def aggregate_block_matrix(
    scores: dict,
    block: str,
    *,
    use_abs: bool = True,
) -> tuple[np.ndarray, int]:
    """Build (n_layers, n_proj) matrix: mean score over neuron rows per projection."""
    cfg = BLOCK_CONFIG[block]
    pattern = cfg["regex"]
    proj_order = cfg["proj_order"]
    layer_proj: dict[tuple[int, str], float] = {}
    for name, val in scores.items():
        m = pattern.match(name)
        if not m:
            continue
        layer = int(m.group(1))
        proj = m.group(2)
        t = val.float().view(-1)
        v = t.abs().mean().item() if use_abs else t.mean().item()
        layer_proj[(layer, proj)] = v

    if not layer_proj:
        raise ValueError(f"No {block} projections found in scores.")

    n_layers = max(layer for layer, _ in layer_proj) + 1
    mat = np.full((n_layers, len(proj_order)), np.nan, dtype=np.float64)
    for layer in range(n_layers):
        for j, proj in enumerate(proj_order):
            key = (layer, proj)
            if key in layer_proj:
                mat[layer, j] = layer_proj[key]
    return mat, n_layers


def aggregate_attn_matrix(scores: dict, *, use_abs: bool = True) -> tuple[np.ndarray, int]:
    return aggregate_block_matrix(scores, "attn", use_abs=use_abs)


def _plot_one(
    mat: np.ndarray,
    *,
    title: str,
    out_path: str,
    proj_labels: tuple[str, ...],
    xlabel: str,
    cmap: str = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = False,
) -> None:
    data = mat.copy()
    if log_scale:
        data = np.log1p(np.maximum(data, 0))

    fig, ax = plt.subplots(figsize=(5.5, max(6, 0.18 * mat.shape[0])))
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(range(len(proj_labels)))
    ax.set_xticklabels(proj_labels, fontsize=11)
    ax.set_ylabel("Layer index")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12)
    n = mat.shape[0]
    step = max(1, n // 12)
    yticks = list(range(0, n, step))
    if yticks[-1] != n - 1:
        yticks.append(n - 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(i) for i in yticks])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean |score|" if not log_scale else "log(1 + mean |score|)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved {out_path}")


def plot_metric(
    scores_dir: str,
    metric: str,
    *,
    block: str = "attn",
    use_abs: bool = True,
    cmap: str = "YlOrRd",
    log_scale: bool = False,
    shared_scale: tuple[float, float] | None = None,
) -> np.ndarray:
    cfg = BLOCK_CONFIG[block]
    scores = load_scores(scores_dir, metric)
    mat, _ = aggregate_block_matrix(scores, block, use_abs=use_abs)
    vmin, vmax = shared_scale if shared_scale else (None, None)
    label = METRIC_LABELS.get(metric, metric)
    task = _task_display_name(scores_dir)
    title = f"{label} — {task}\n(mean |score| per layer × {cfg['title_suffix']})"
    out = os.path.join(scores_dir, f"{cfg['out_prefix']}_{metric}.png")
    _plot_one(
        mat,
        title=title,
        out_path=out,
        proj_labels=cfg["proj_labels"],
        xlabel=cfg["xlabel"],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        log_scale=log_scale,
    )
    return mat


def _task_display_name(scores_dir: str) -> str:
    base = os.path.basename(scores_dir.rstrip("/"))
    return TASK_LABELS.get(base, base.replace("results_", "").upper())


def plot_multi(
    scores_dir: str,
    metrics: list[str],
    *,
    block: str = "attn",
    use_abs: bool = True,
    cmap: str = "YlOrRd",
    log_scale: bool = False,
    out_name: str | None = None,
) -> str:
    """One figure per task: 1×N panels (Fisher / Learned-A / Learned-B / Gradient)."""
    cfg = BLOCK_CONFIG[block]
    proj_labels = cfg["proj_labels"]
    if out_name is None:
        out_name = cfg["combined_name"] or "heatmap_combined.png"

    mats: list[np.ndarray] = []
    valid: list[str] = []
    for m in metrics:
        path = os.path.join(scores_dir, f"{m}.pt")
        if not os.path.isfile(path):
            _log(f"Skip {m} (no {path})")
            continue
        scores = load_scores(scores_dir, m)
        mat, _ = aggregate_block_matrix(scores, block, use_abs=use_abs)
        mats.append(mat)
        valid.append(m)

    if not mats:
        raise FileNotFoundError(f"No metric .pt files in {scores_dir}")

    if log_scale:
        mats = [np.log1p(np.maximum(m, 0)) for m in mats]
    vmin = min(np.nanmin(m) for m in mats)
    vmax = max(np.nanmax(m) for m in mats)

    n = len(valid)
    ncols = n
    fig_h = max(5.5, 0.17 * mats[0].shape[0])
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, fig_h))
    axes = np.atleast_1d(axes).flatten()

    task = _task_display_name(scores_dir)
    n_proj = len(proj_labels)
    im = None
    for idx, (metric, mat) in enumerate(zip(valid, mats)):
        ax = axes[idx]
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n_proj))
        ax.set_xticklabels(proj_labels, fontsize=10)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11, pad=8)
        nlay = mat.shape[0]
        step = max(1, nlay // 8)
        yticks = list(range(0, nlay, step))
        if yticks[-1] != nlay - 1:
            yticks.append(nlay - 1)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(i) for i in yticks], fontsize=8)
        if idx == 0:
            ax.set_ylabel("Layer index", fontsize=10)
        ax.set_xlabel(cfg["axis_short"], fontsize=9)

    cbar_label = "mean |score|" if not log_scale else "log(1 + mean |score|)"
    fig.subplots_adjust(right=0.92, wspace=0.28)
    cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax, label=cbar_label)
    fig.suptitle(
        f"{task} — {cfg['title_suffix']} importance "
        f"({mats[0].shape[0]} layers, task-specific calibration)",
        fontsize=13,
        y=0.98,
    )
    out = os.path.join(scores_dir, out_name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved {out}")
    return out


def _plot_ffn_per_metric(
    scores_dir: str,
    metrics: list[str],
    *,
    use_abs: bool,
    cmap: str,
    log_scale: bool,
) -> None:
    """FFN heatmaps: one file per metric, each with its own color scale."""
    n_ok = 0
    for m in metrics:
        path = os.path.join(scores_dir, f"{m}.pt")
        if not os.path.isfile(path):
            _log(f"Skip {m} (no {path})")
            continue
        plot_metric(
            scores_dir,
            m,
            block="ffn",
            use_abs=use_abs,
            cmap=cmap,
            log_scale=log_scale,
        )
        n_ok += 1
    if n_ok == 0:
        raise FileNotFoundError(f"No metric .pt files in {scores_dir}")


def _run_blocks(
    scores_dir: str,
    metrics: list[str],
    blocks: list[str],
    *,
    use_abs: bool,
    cmap: str,
    log_scale: bool,
    also_single: bool,
    ffn_combined: bool,
) -> None:
    for block in blocks:
        _log(f"  block={block}")
        if block == "ffn" and not ffn_combined:
            _plot_ffn_per_metric(
                scores_dir,
                metrics,
                use_abs=use_abs,
                cmap=cmap,
                log_scale=log_scale,
            )
            continue

        out_name = "heatmap_ffn_combined.png" if block == "ffn" else None
        plot_multi(
            scores_dir,
            metrics,
            block=block,
            use_abs=use_abs,
            cmap=cmap,
            log_scale=log_scale,
            out_name=out_name,
        )
        if also_single and block == "attn":
            for m in metrics:
                if os.path.isfile(os.path.join(scores_dir, f"{m}.pt")):
                    plot_metric(
                        scores_dir,
                        m,
                        block=block,
                        use_abs=use_abs,
                        cmap=cmap,
                        log_scale=log_scale,
                    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Layer × projection importance heatmap (attention Q/K/V/O or FFN)."
    )
    p.add_argument("--scores_dir", required=True, help="e.g. ./results_c4 or ./results_wikitext")
    p.add_argument("--metric", default="", help="Single metric (default: shapley_fisher)")
    p.add_argument(
        "--metrics",
        default="",
        help="Comma-separated metrics for one combined figure",
    )
    p.add_argument(
        "--block",
        choices=("attn", "ffn", "both"),
        default="attn",
        help="attn=Q/K/V/O, ffn=gate/up/down, both=generate both (default: attn)",
    )
    p.add_argument("--signed", action="store_true", help="Use mean signed score (default: mean |score|)")
    p.add_argument("--log_scale", action="store_true")
    p.add_argument("--cmap", default="YlOrRd")
    p.add_argument(
        "--ffn_combined",
        action="store_true",
        help="FFN: force 4-in-1 combined figure (shared color scale; not recommended)",
    )
    args = p.parse_args()

    scores_dir = os.path.abspath(args.scores_dir)
    use_abs = not args.signed
    blocks = ["attn", "ffn"] if args.block == "both" else [args.block]
    also_single = os.environ.get("HEATMAP_ALSO_SINGLE", "0") == "1"

    if args.metrics.strip():
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        _run_blocks(
            scores_dir,
            metrics,
            blocks,
            use_abs=use_abs,
            cmap=args.cmap,
            log_scale=args.log_scale,
            also_single=also_single,
            ffn_combined=args.ffn_combined,
        )
    else:
        metric = args.metric or "shapley_fisher"
        for block in blocks:
            plot_metric(
                scores_dir,
                metric,
                block=block,
                use_abs=use_abs,
                cmap=args.cmap,
                log_scale=args.log_scale,
            )


if __name__ == "__main__":
    main()
