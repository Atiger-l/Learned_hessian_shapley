#!/usr/bin/env bash
# Attention: 4 指标横排合并图；FFN: 每种指标单独一张图（各自 color scale）
set -euo pipefail
cd "$(dirname "$0")"

METRICS="${METRICS:-shapley_fisher,shapley_learned_a,shapley_learned_b,gradient}"

for dir in ./results_wikitext ./results_c4 ./results_mmlu; do
  [[ -d "$dir" ]] || continue
  echo "=== $dir ==="
  python3 plot_shapley_heatmap.py --scores_dir "$dir" --metrics "$METRICS" --block attn
  python3 plot_shapley_heatmap.py --scores_dir "$dir" --metrics "$METRICS" --block ffn
done

echo "Done:"
for dir in results_wikitext results_c4 results_mmlu; do
  echo "  ${dir}/heatmap_combined.png"
  echo "  ${dir}/heatmap_ffn_{shapley_fisher,shapley_learned_a,shapley_learned_b,gradient}.png"
done
