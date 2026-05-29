#!/usr/bin/env bash
# 剩余实验（任务相关版）：
#   Learned 代理 / Shapley 分数 必须在对应任务的 calibration 上重算；
#   WikiText 已有 results_wikitext/ → 不要重跑 WikiText。
#
# 流程:
#   1) MMLU: run_experiment --dataset mmlu  → results_mmlu/*.pt（含重训 scheme B 代理）
#   2) C4:   run_experiment --dataset c4    → results_c4/*.pt
#   3) 下游: MMLU acc / C4 PPL（各用本任务的 scores_dir）
#   4) ablation.py（独立小实验，单卡）
#
#   bash run_remaining.sh
#
# 若只想做「WikiText 分数迁移到 MMLU/C4」的对比（不重训），见文末 CROSS_TASK=1

set -euo pipefail
cd "$(dirname "$0")"

export GPUS="${GPUS:-2,3,4,5,6,7}"
export FRACS="${FRACS:-0.05,0.06,0.07,0.08,0.09,0.10}"
export MMLU_SUBJECT="${MMLU_SUBJECT:-abstract_algebra}"
export LEARNED_SCHEME="${LEARNED_SCHEME:-b}"
export N_CALIB="${N_CALIB:-64}"
export BATCH="${BATCH:-1}"
export SEQ_LEN="${SEQ_LEN:-48}"
export SURR_EP="${SURR_EP:-30}"
export HUTCH="${HUTCH:-2}"
export SURR_BPE="${SURR_BATCHES_PER_EPOCH:-8}"

GPU0="${GPUS%%,*}"
GPU1="$(echo "$GPUS" | cut -d, -f2)"
GPU1="${GPU1:-$GPU0}"

if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-lhs}"
fi
if [[ "${USE_OFFICIAL_HF:-0}" != "1" ]]; then
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HUGGINGFACE_HUB_BASE_URL="${HUGGINGFACE_HUB_BASE_URL:-$HF_ENDPOINT}"
  export HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-$HF_ENDPOINT}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
echo "HF_ENDPOINT=${HF_ENDPOINT:-<official>}"

MODEL_ARGS=()
[[ -n "${MODEL:-}" ]] && MODEL_ARGS=(--model "$MODEL")
COMMON=(
  "${MODEL_ARGS[@]}"
  --n_calib "$N_CALIB"
  --batch_size "$BATCH"
  --seq_len "$SEQ_LEN"
  --surrogate_epochs "$SURR_EP"
  --hutchinson_samples "$HUTCH"
  --learned_scheme "$LEARNED_SCHEME"
  --surrogate_batches_per_epoch "$SURR_BPE"
)

_log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 交叉任务对照：不重训，WikiText 分数评 MMLU/C4 ─────────────────────────
if [[ "${CROSS_TASK:-0}" == "1" ]]; then
  _log "CROSS_TASK=1: 使用 results_wikitext 分数（不重训代理）"
  SCORES=./results_wikitext
  [[ -d "$SCORES" ]] || { echo "ERROR: 无 $SCORES"; exit 1; }
  python3 evaluate_downstream.py --task mmlu_acc --scores_dir "$SCORES" \
    --output_dir ./results_mmlu_acc_from_wikitext --mmlu_subject "$MMLU_SUBJECT" \
    --deactivate_fracs "$FRACS" --parallel_gpus "$GPUS"
  python3 evaluate_downstream.py --task c4_ppl --scores_dir "$SCORES" \
    --output_dir ./results_c4_ppl_from_wikitext --deactivate_fracs "$FRACS" \
    --parallel_gpus "$GPUS" --n_eval_texts 128
  exit 0
fi

echo "========== 任务相关实验（MMLU/C4 各训各的代理 + 下游）=========="
echo "LEARNED_SCHEME=$LEARNED_SCHEME  GPUS=$GPUS"

# ── Phase A: 算分 + 重训代理（MMLU 与 C4 可并行，各占 1 卡）────────────────
pids=()
names=()

if [[ "${SKIP_MMLU_SCORES:-0}" != "1" ]]; then
  _log "GPU $GPU0: run_experiment mmlu → results_mmlu"
  CUDA_VISIBLE_DEVICES="$GPU0" python3 run_experiment.py \
    "${COMMON[@]}" --dataset mmlu --mmlu_subject "$MMLU_SUBJECT" \
    --output_dir ./results_mmlu &
  pids+=($!); names+=("mmlu:$GPU0")
fi

if [[ "${SKIP_C4_SCORES:-0}" != "1" ]]; then
  _log "GPU $GPU1: run_experiment c4 → results_c4"
  CUDA_VISIBLE_DEVICES="$GPU1" python3 run_experiment.py \
    "${COMMON[@]}" --dataset c4 --output_dir ./results_c4 &
  pids+=($!); names+=("c4:$GPU1")
fi

fail=0
mmlu_ok=0
c4_ok=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    _log "OK ${names[$i]}"
    [[ "${names[$i]}" == mmlu:* ]] && mmlu_ok=1
    [[ "${names[$i]}" == c4:* ]] && c4_ok=1
  else
    _log "FAILED ${names[$i]}"
    fail=1
  fi
done
if [[ "$fail" -ne 0 ]]; then
  _log "Phase A 有失败。若仅 C4 失败，可: SKIP_C4_SCORES=1 SKIP_C4_DOWNSTREAM=1 先跑 MMLU"
  exit 1
fi

# ── Phase B: 下游（串行；内部 4 metric 用满 GPUS）────────────────────────────
if [[ "${SKIP_MMLU_DOWNSTREAM:-0}" != "1" ]]; then
  _log "MMLU accuracy（scores=results_mmlu）"
  python3 evaluate_downstream.py \
    --task mmlu_acc \
    --scores_dir ./results_mmlu \
    --output_dir ./results_mmlu_acc \
    --mmlu_subject "$MMLU_SUBJECT" \
    --deactivate_fracs "$FRACS" \
    --parallel_gpus "$GPUS"
fi

if [[ "${SKIP_C4_DOWNSTREAM:-0}" != "1" ]]; then
  _log "C4 PPL（scores=results_c4）"
  python3 evaluate_downstream.py \
    --task c4_ppl \
    --scores_dir ./results_c4 \
    --output_dir ./results_c4_ppl \
    --deactivate_fracs "$FRACS" \
    --parallel_gpus "$GPUS" \
    --n_eval_texts 128
fi

if [[ "${SKIP_ABLATION:-0}" != "1" ]]; then
  _log "ablation（GPU $GPU0）"
  CUDA_VISIBLE_DEVICES="$GPU0" python3 ablation.py "${MODEL_ARGS[@]}"
fi

echo ""
echo "========== 完成 =========="
echo "  WikiText（已有）: results_wikitext/ + results_wikitext_fine/"
echo "  MMLU 分数+代理:   results_mmlu/"
echo "  MMLU acc:         results_mmlu_acc/"
echo "  C4 分数+代理:     results_c4/"
echo "  C4 PPL:           results_c4_ppl/"
