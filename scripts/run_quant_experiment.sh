#!/usr/bin/env bash
# Quantization experiment: GPTQ baseline + Shapley-guided row INT8.
#
# Prerequisites:
#   conda activate lhs
#   # Shapley row quant needs NO extra packages.
#   # GPTQ baseline:
#   pip install gptqmodel
#   # OR: pip install auto-gptq optimum
#   # OR clone only if pip fails:
#   #   git clone https://github.com/AutoGPTQ/AutoGPTQ && cd AutoGPTQ && pip install -e .
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_quant_experiment.sh
#   DATASET=mmlu MMLU_SUBJECT=high_school_mathematics SCORES_DIR=./results_mmlu_high_school_mathematics bash run_quant_experiment.sh
#   SKIP_GPTQ=1 bash run_quant_experiment.sh   # only Shapley row (no gptqmodel)

set -euo pipefail
cd "$(dirname "$0")"

export DATASET="${DATASET:-mmlu}"
export SCORES_DIR="${SCORES_DIR:-./results_mmlu_${MMLU_SUBJECT}}"
export QUANT_FRAC="${QUANT_FRAC:-0.10}"
export WBITS="${WBITS:-8}"
export N_CALIB="${N_CALIB:-64}"
export GPU="${GPU:-2}"
export GPTQ_BACKEND="${GPTQ_BACKEND:-auto_gptq}"
export MMLU_SUBJECT="${MMLU_SUBJECT:-high_school_mathematics}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_BASE_URL="${HUGGINGFACE_HUB_BASE_URL:-$HF_ENDPOINT}"
export HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-$HF_ENDPOINT}"

if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-lhs}"
fi

_log() { echo "[$(date '+%H:%M:%S')] $*"; }

MODEL="../Qwen2.5-3B-Instruct"
OUT_BASE="${OUT_BASE:-${SCORES_DIR}/quant_runs_w${WBITS}_f${QUANT_FRAC}}"

_log "Dataset=$DATASET scores=$SCORES_DIR out_base=$OUT_BASE GPU=$GPU"

run_shapley() {
  local metric="$1"
  local out="${OUT_BASE}/shapley_${metric}"
  _log "Shapley row quant: metric=$metric → $out"
  CUDA_VISIBLE_DEVICES="$GPU" python3 quantize.py \
    --method shapley_row \
    --model "$MODEL" \
    --scores_dir "$SCORES_DIR" \
    --metric "$metric" \
    --quantize_frac "$QUANT_FRAC" \
    --wbits "$WBITS" \
    --dataset "$DATASET" \
    --mmlu_subject "$MMLU_SUBJECT" \
    --output_dir "$out"

  if [[ "$DATASET" == "mmlu" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python3 evaluate_quant.py \
      --model "$out" --task mmlu_acc \
      --mmlu_subject "$MMLU_SUBJECT" --mmlu_scoring generate
  else
    CUDA_VISIBLE_DEVICES="$GPU" python3 evaluate_quant.py \
      --model "$out" --task c4_ppl
  fi
}

if [[ "${SKIP_GPTQ:-0}" != "1" ]]; then
  GPTQ_OUT="${OUT_BASE}/gptq_w${WBITS}"
  _log "GPTQ baseline → $GPTQ_OUT"
  CUDA_VISIBLE_DEVICES="$GPU" python3 quantize.py \
    --method gptq \
    --model "$MODEL" \
    --wbits "$WBITS" \
    --dataset "$DATASET" \
    --mmlu_subject "$MMLU_SUBJECT" \
    --n_calib "$N_CALIB" \
    --gptq_backend "$GPTQ_BACKEND" \
    --output_dir "$GPTQ_OUT"

  if [[ "$DATASET" == "mmlu" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python3 evaluate_quant.py \
      --model "$GPTQ_OUT" --task mmlu_acc \
      --mmlu_subject "$MMLU_SUBJECT" --mmlu_scoring generate
  else
    CUDA_VISIBLE_DEVICES="$GPU" python3 evaluate_quant.py \
      --model "$GPTQ_OUT" --task c4_ppl
  fi
else
  _log "SKIP_GPTQ=1 — skipping GPTQ baseline"
fi

for metric in shapley_fisher shapley_learned_a shapley_learned_b gradient; do
  [[ -f "${SCORES_DIR}/${metric}.pt" ]] || { _log "Skip $metric (no scores)"; continue; }
  run_shapley "$metric"
done

_log "Done. Results under ${OUT_BASE}/"
echo ""
echo "Result dirs:"
echo "  GPTQ:    ${OUT_BASE}/gptq_w${WBITS}/eval_mmlu_acc.json  (or eval_c4_ppl.json)"
echo "  Shapley: ${OUT_BASE}/shapley_<metric>/eval_*.json"
