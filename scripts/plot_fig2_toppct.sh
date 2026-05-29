#!/usr/bin/env bash
# Figure 2 style: per-layer count of neurons in global top p% |score| (same p for all metrics).
set -euo pipefail
cd "$(dirname "$0")"

TOP_PCT="${TOP_PCT:-1}"
METRICS="${METRICS:-shapley_fisher,shapley_learned_a,shapley_learned_b,gradient}"
OUT_DIR="${OUT_DIR:-./results_fig2}"

python3 plot_neuron_activation_counts.py \
  --top_pct "$TOP_PCT" \
  --panel "mmlu:./results_mmlu" \
  --panel "wikitext:./results_wikitext" \
  --metrics "$METRICS" \
  --out_dir "$OUT_DIR"

echo "Done: ${OUT_DIR}/neuron_activation_counts_top${TOP_PCT}pct.png"
echo "      (use TOP_PCT=5 for 5%% version)"
