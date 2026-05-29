#!/usr/bin/env bash
# MMLU acc only: use existing scores in results_mmlu_high_school_mathematics (no retrain).
#
#   bash run_mmlu_hs_math_acc_only.sh
#   GPUS=4,5,6,7 bash run_mmlu_hs_math_acc_only.sh

set -euo pipefail
cd "$(dirname "$0")"

export MMLU_SUBJECT="${MMLU_SUBJECT:-high_school_mathematics}"
export SCORES_DIR="${SCORES_DIR:-./results_mmlu_${MMLU_SUBJECT}}"
export ACC_DIR="${ACC_DIR:-./results_mmlu_acc_${MMLU_SUBJECT}}"
export FRACS="${FRACS:-0.05,0.06,0.07,0.08,0.09,0.10}"
export GPUS="${GPUS:-4,5,6,7}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_BASE_URL="${HUGGINGFACE_HUB_BASE_URL:-$HF_ENDPOINT}"
export HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-$HF_ENDPOINT}"

if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-lhs}"
fi

_log() { echo "[$(date '+%H:%M:%S')] $*"; }

for f in shapley_fisher.pt shapley_learned_a.pt shapley_learned_b.pt gradient.pt; do
  [[ -f "${SCORES_DIR}/${f}" ]] || { echo "Missing ${SCORES_DIR}/${f}"; exit 1; }
done

_log "MMLU acc: subject=$MMLU_SUBJECT scores=$SCORES_DIR -> $ACC_DIR"
python3 evaluate_downstream.py \
  --task mmlu_acc \
  --scores_dir "$SCORES_DIR" \
  --output_dir "$ACC_DIR" \
  --mmlu_subject "$MMLU_SUBJECT" \
  --deactivate_fracs "$FRACS" \
  --parallel_gpus "$GPUS"

python3 plot_downstream_results.py --input "$ACC_DIR"
_log "Done: $ACC_DIR/eval_mmlu_acc.png  $ACC_DIR/eval_mmlu_acc.json"
