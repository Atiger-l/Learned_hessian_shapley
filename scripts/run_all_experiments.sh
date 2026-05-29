#!/usr/bin/env bash
# 在 GPU 2–7 上跑完全部实验（能并行的尽量并行）
#
# 用法:
#   cd scripts
#   bash run_all_experiments.sh              # 全流程（含重训代理，慎用）
#
# 已有 WikiText 分数、只跑 MMLU/C4/消融 → 请用:
#   bash run_remaining.sh
#
#   bash run_all_experiments.sh --phase 1
#   SKIP_WIKITEXT=1 bash run_all_experiments.sh   # 仍会重训 MMLU/C4 代理
#
# 环境变量:
#   GPUS=2,3,4,5,6,7          sweep 并行用（默认）
#   LEARNED_SCHEME=b          run_experiment 方案（默认 b）
#   N_CALIB BATCH SEQ_LEN SURR_EP SURR_BATCHES_PER_EPOCH  同 命令行.sh
#   MMLU_SUBJECTS="abstract_algebra high_school_mathematics"  多 subject 并行
#   SKIP_WIKITEXT=1 SKIP_MMLU=1 SKIP_C4=1 SKIP_ABLATION=1     跳过某阶段
#   SKIP_COARSE_SWEEP=1       跳过 0.05–0.85 粗 PPL sweep
#   RUN_WIKITEXT_RETRAIN=0    默认不重训 WikiText（你已完成）

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── GPUs ─────────────────────────────────────────────────────────────────────
export GPUS="${GPUS:-2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
GPU0="${GPU_ARR[0]}"
GPU1="${GPU_ARR[1]:-$GPU0}"
GPU2="${GPU_ARR[2]:-$GPU0}"

# ── conda / HF（与 命令行.sh 一致）────────────────────────────────────────────
if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-lhs}"
fi
if [[ "${USE_OFFICIAL_HF:-0}" != "1" ]]; then
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LEARNED_SCHEME="${LEARNED_SCHEME:-b}"
N_CALIB="${N_CALIB:-64}"
BATCH="${BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-48}"
SURR_EP="${SURR_EP:-30}"
HUTCH="${HUTCH:-2}"
SURR_BPE="${SURR_BATCHES_PER_EPOCH:-8}"
MODEL="${MODEL:-}"
FRACS_FINE="${FRACS_FINE:-0.05,0.06,0.07,0.08,0.09,0.10}"
FRACS_COARSE="${FRACS_COARSE:-0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85}"
MMLU_SUBJECTS="${MMLU_SUBJECTS:-abstract_algebra}"

MODEL_ARGS=()
[[ -n "$MODEL" ]] && MODEL_ARGS=(--model "$MODEL")

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

PHASE="${1:-all}"
[[ "$PHASE" == "--phase" ]] && PHASE="${2:-all}"

_log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }

# 单卡跑 run_experiment
_run_experiment_on_gpu() {
  local gpu="$1"
  shift
  _log "GPU $gpu: python3 run_experiment.py $*"
  CUDA_VISIBLE_DEVICES="$gpu" python3 run_experiment.py "$@"
}

# 多卡 plot_eval_sweep（内部 4 metric 一批）
_run_sweep() {
  local scores_dir="$1"
  local output_dir="$2"
  local fracs="$3"
  local preset="$4"
  _log "PPL sweep: scores=$scores_dir -> $output_dir  fracs=$fracs"
  python3 plot_eval_sweep.py \
    --scores_dir "$scores_dir" \
    --output_dir "$output_dir" \
    --deactivate_fracs "$fracs" \
    --parallel_gpus "$GPUS" \
    --eval_preset "$preset" \
    --n_eval_texts "${N_EVAL:-256}" \
    --no_plot
  python3 plot_eval_sweep_from_json.py --output_dir "$output_dir" --zoom_max_frac 0.10
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: 算 Shapley 分数（每条占 1 张卡，多条并行）
# ═══════════════════════════════════════════════════════════════════════════
phase1() {
  _log "========== Phase 1: run_experiment（并行，每任务 1 GPU）=========="
  local pids=() names=()
  local gi=0

  _next_gpu() {
    echo "${GPU_ARR[$((gi % ${#GPU_ARR[@]}))]}"
    gi=$((gi + 1))
  }

  if [[ "${SKIP_WIKITEXT:-0}" != "1" ]] || [[ "${RUN_WIKITEXT_RETRAIN:-0}" == "1" ]]; then
    g=$(_next_gpu)
    _run_experiment_on_gpu "$g" "${COMMON[@]}" --dataset wikitext --output_dir ./results_wikitext &
    pids+=($!); names+=("wikitext:$g")
  else
    _log "跳过 WikiText run_experiment（SKIP_WIKITEXT=1）"
  fi

  if [[ "${SKIP_MMLU:-0}" != "1" ]]; then
    for subj in $MMLU_SUBJECTS; do
      g=$(_next_gpu)
      out="./results_mmlu_${subj}"
      [[ "$subj" == "abstract_algebra" ]] && out="./results_mmlu"
      _run_experiment_on_gpu "$g" "${COMMON[@]}" \
        --dataset mmlu --mmlu_subject "$subj" --output_dir "$out" &
      pids+=($!); names+=("mmlu_${subj}:$g")
    done
  fi

  if [[ "${SKIP_C4:-0}" != "1" ]]; then
    g=$(_next_gpu)
    _run_experiment_on_gpu "$g" "${COMMON[@]}" --dataset c4 --output_dir ./results_c4 &
    pids+=($!); names+=("c4:$g")
  fi

  local fail=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      _log "FAILED: ${names[$i]}"
      fail=1
    else
      _log "OK: ${names[$i]}"
    fi
  done
  [[ "$fail" -eq 0 ]] || exit 1
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: PPL sweep（每个数据集串行；内部用 GPUS 2–7 并行 4 metrics）
#         避免同时开 3 个 sweep 抢 6 张卡
# ═══════════════════════════════════════════════════════════════════════════
phase2() {
  _log "========== Phase 2: PPL fine sweep 0.05–0.10（每任务占满 $GPUS）=========="

  if [[ "${SKIP_WIKITEXT:-0}" != "1" ]] || [[ -d ./results_wikitext ]] && ls ./results_wikitext/*.pt &>/dev/null; then
    _run_sweep ./results_wikitext ./results_wikitext_fine "$FRACS_FINE" wikitext
  fi

  if [[ "${SKIP_MMLU:-0}" != "1" ]]; then
    for subj in $MMLU_SUBJECTS; do
      scores="./results_mmlu_${subj}"
      out="./results_mmlu_${subj}_fine"
      [[ "$subj" == "abstract_algebra" ]] && scores="./results_mmlu" && out="./results_mmlu_fine"
      [[ -d "$scores" ]] || { _log "跳过 MMLU sweep（无 $scores）"; continue; }
      _run_sweep "$scores" "$out" "$FRACS_FINE" wikitext
    done
  fi

  if [[ "${SKIP_C4:-0}" != "1" ]] && [[ -d ./results_c4 ]]; then
    _run_sweep ./results_c4 ./results_c4_fine "$FRACS_FINE" wikitext
  fi
}

phase2_coarse() {
  [[ "${SKIP_COARSE_SWEEP:-0}" == "1" ]] && return 0
  _log "========== Phase 2b: PPL coarse sweep 0.05–0.85 =========="

  [[ -d ./results_wikitext ]] && _run_sweep ./results_wikitext ./results_wikitext_sweep "$FRACS_COARSE" wikitext
  [[ -d ./results_mmlu ]] && _run_sweep ./results_mmlu ./results_mmlu_sweep "$FRACS_COARSE" wikitext
  [[ -d ./results_c4 ]] && _run_sweep ./results_c4 ./results_c4_sweep "$FRACS_COARSE" wikitext
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: 消融（单卡）
# ═══════════════════════════════════════════════════════════════════════════
phase3() {
  [[ "${SKIP_ABLATION:-0}" == "1" ]] && return 0
  _log "========== Phase 3: ablation.py（GPU $GPU0）=========="
  CUDA_VISIBLE_DEVICES="$GPU0" python3 ablation.py "${MODEL_ARGS[@]}"
}

# ═══════════════════════════════════════════════════════════════════════════
case "$PHASE" in
  all)
    phase1
    phase2
    phase2_coarse
    phase3
    ;;
  1) phase1 ;;
  2) phase2 ;;
  2b|coarse) phase2_coarse ;;
  3) phase3 ;;
  *)
    echo "用法: $0 [all|1|2|2b|3]   或  $0 --phase 1"
    exit 1
    ;;
esac

_log "========== 全部完成 =========="
_log "主图: results_wikitext_fine/eval_sweep_ppl.png"
_log "MMLU: results_mmlu_fine/eval_sweep_ppl.png（若已跑 MMLU）"
