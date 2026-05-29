#!/usr/bin/env bash
# Model Shapley Figure 2: neuron activation counts per layer (two benchmarks).
set -euo pipefail
cd "$(dirname "$0")"

# Paper defaults: MMLU τ=5, GSM8K/WikiText/C4 τ=0.5 (signed Shapley > τ)
METRIC="${METRIC:-shapley_fisher}"
OUT_DIR="${OUT_DIR:-./results_fig2}"

python3 plot_neuron_activation_counts.py \
  --panel "mmlu:./results_mmlu:5" \
  --panel "wikitext:./results_wikitext:0.5" \
  --metric "$METRIC" \
  --out_dir "$OUT_DIR"

echo "Done: ${OUT_DIR}/neuron_activation_counts.png"
