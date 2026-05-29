#!/usr/bin/env bash
# MMLU 重跑：high_school_mathematics（比 abstract_algebra 更易出 acc 曲线）
#
#   bash run_mmlu_hs_math.sh
#   GPU=5 bash run_mmlu_hs_math.sh

set -euo pipefail
cd "$(dirname "$0")"

export MMLU_SUBJECT="${MMLU_SUBJECT:-high_school_mathematics}"
# 整卡空闲时用 GPU=1 等；勿与陌生进程同卡（CUDA_MEM_FRACTION 挡不住别人占显存）
export GPU="${GPU:-1}"
export CUDA_MEM_FRACTION="${CUDA_MEM_FRACTION:-}"
export MMLU_CALIB_SPLIT="${MMLU_CALIB_SPLIT:-test}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_BASE_URL="${HUGGINGFACE_HUB_BASE_URL:-$HF_ENDPOINT}"
export HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-$HF_ENDPOINT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HUTCH="${HUTCH:-2}"
export SEQ_LEN="${SEQ_LEN:-48}"
export N_CALIB="${N_CALIB:-64}"
export LEARNED_SCHEME="${LEARNED_SCHEME:-both}"
export FRACS="${FRACS:-0.05,0.06,0.07,0.08,0.09,0.10}"
export GPUS="${GPUS:-4,5,6,7}"

# 半卡仍 OOM 时：OOM_SAFE=1（降 seq_len / hutch / 先只训 A）
if [[ "${OOM_SAFE:-0}" == "1" ]]; then
  export SEQ_LEN="${SEQ_LEN:-32}"
  export HUTCH="${HUTCH:-1}"
  export N_CALIB="${N_CALIB:-32}"
  export LEARNED_SCHEME="${LEARNED_SCHEME:-a}"
  unset CUDA_MEM_FRACTION
fi

SCORES_DIR="./results_mmlu_${MMLU_SUBJECT}"
ACC_DIR="./results_mmlu_acc_${MMLU_SUBJECT}"

if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-lhs}"
fi

_log() { echo "[$(date '+%H:%M:%S')] $*"; }

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  _log "1/3 Prefetch MMLU ${MMLU_SUBJECT} (hf-mirror) ..."
  python3 download_mmlu_subject.py "$MMLU_SUBJECT"
else
  _log "1/3 Skip download (SKIP_DOWNLOAD=1)"
fi

_cap="${CUDA_MEM_FRACTION:-<unset>}"
_log "2/3 run_experiment GPU=$GPU VRAM_cap=$_cap split=$MMLU_CALIB_SPLIT scheme=$LEARNED_SCHEME seq=$SEQ_LEN hutch=$HUTCH -> $SCORES_DIR"
_nvidia_warn() {
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv 2>/dev/null | head -8 || true
}
_nvidia_warn
_extra=()
[[ -n "${CUDA_MEM_FRACTION:-}" ]] && _extra+=(CUDA_MEM_FRACTION="$CUDA_MEM_FRACTION")
CUDA_VISIBLE_DEVICES="$GPU" "${_extra[@]}" python3 run_experiment.py \
  --dataset mmlu \
  --mmlu_subject "$MMLU_SUBJECT" \
  --hf_split "$MMLU_CALIB_SPLIT" \
  --output_dir "$SCORES_DIR" \
  --learned_scheme "$LEARNED_SCHEME" \
  --n_calib "$N_CALIB" \
  --batch_size 1 \
  --seq_len "$SEQ_LEN" \
  --hutchinson_samples "$HUTCH"

_log "3/3 downstream acc -> $ACC_DIR"
python3 evaluate_downstream.py \
  --task mmlu_acc \
  --scores_dir "$SCORES_DIR" \
  --output_dir "$ACC_DIR" \
  --mmlu_subject "$MMLU_SUBJECT" \
  --deactivate_fracs "$FRACS" \
  --parallel_gpus "$GPUS"

python3 plot_downstream_results.py --input "$ACC_DIR"
python3 plot_shapley_heatmap.py --scores_dir "$SCORES_DIR" \
  --metrics shapley_fisher,shapley_learned_a,shapley_learned_b,gradient

_log "Done: $ACC_DIR/eval_mmlu_acc.png  $SCORES_DIR/heatmap_combined.png"
