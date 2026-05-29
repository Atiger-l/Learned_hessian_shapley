#!/usr/bin/env bash
# Parallel eval sweep: one metric per GPU (default 2–7; avoid 0/1 if occupied).
# GPUs 4–7 may have ~8GB used by other jobs; Qwen2.5-3B (~14GB) still fits.
# Kill any old plot_eval_sweep on GPU 0 before starting.
set -euo pipefail
cd "$(dirname "$0")"

SCORES_DIR="${SCORES_DIR:-./results_wikitext}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCORES_DIR}"
GPUS="${GPUS:-2,3,4,5,6,7}"
N_EVAL="${N_EVAL:-256}"
PRESET="${PRESET:-wikitext}"
FRACS="${FRACS:-}"

echo "scores=$SCORES_DIR  output=$OUTPUT_DIR  GPUs=$GPUS"
EXTRA=()
[[ -n "$FRACS" ]] && EXTRA=(--deactivate_fracs "$FRACS")
python3 plot_eval_sweep.py \
  --scores_dir "$SCORES_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --parallel_gpus "$GPUS" \
  "${EXTRA[@]}" \
  --eval_preset "$PRESET" \
  --n_eval_texts "$N_EVAL" \
  "$@"
