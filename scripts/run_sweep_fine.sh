#!/usr/bin/env bash
# Fine sweep: freeze bottom 5%–10% in steps of 1% (0.05, 0.06, …, 0.10).
# Reuses existing Shapley scores; only runs inference PPL (no retrain).
set -euo pipefail
cd "$(dirname "$0")"

SCORES_DIR="${SCORES_DIR:-./results_wikitext}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_wikitext_fine}"
GPUS="${GPUS:-2,3,4,5,6,7}"
FRACS="${FRACS:-0.05,0.06,0.07,0.08,0.09,0.10}"
N_EVAL="${N_EVAL:-256}"
PRESET="${PRESET:-wikitext}"

mkdir -p "$OUTPUT_DIR"

echo "scores=$SCORES_DIR  output=$OUTPUT_DIR"
echo "freeze fracs: $FRACS"
echo "GPUs: $GPUS"

python3 plot_eval_sweep.py \
  --scores_dir "$SCORES_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --deactivate_fracs "$FRACS" \
  --parallel_gpus "$GPUS" \
  --eval_preset "$PRESET" \
  --n_eval_texts "$N_EVAL" \
  "$@"

echo ""
echo "Done. Plot only (no model):"
echo "  python3 plot_eval_sweep_from_json.py --output_dir $OUTPUT_DIR --zoom_max_frac 0.10"
