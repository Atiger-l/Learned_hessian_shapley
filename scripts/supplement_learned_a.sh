#!/usr/bin/env bash
# 仅补 Scheme A：训练 surrogate_a + shapley_learned_a.pt，不重跑 B/Fisher/Gradient 下游全集。
#
# 串行（默认，一张卡依次跑 c4 → mmlu）:
#   GPUS=3 bash supplement_learned_a.sh
#
# 并行（C4 与 MMLU 各用一张卡同时训分 + 下游）:
#   PARALLEL=1 GPU_C4=3 GPU_MMLU=4 bash supplement_learned_a.sh
#   PARALLEL=1 GPUS=3,4 bash supplement_learned_a.sh   # 同上，GPUS 前两个分别为 C4/MMLU
#
# 只补一个:
#   DATASETS=c4 GPU_C4=3 bash supplement_learned_a.sh
#   DATASETS=mmlu GPU_MMLU=4 bash supplement_learned_a.sh

set -euo pipefail
cd "$(dirname "$0")"

export GPUS="${GPUS:-3,4}"
export FRACS="${FRACS:-0.05,0.06,0.07,0.08,0.09,0.10}"
export MMLU_SUBJECT="${MMLU_SUBJECT:-abstract_algebra}"
export N_CALIB="${N_CALIB:-64}"
export BATCH="${BATCH:-1}"
export SEQ_LEN="${SEQ_LEN:-48}"
export SURR_EP="${SURR_EP:-30}"
export HUTCH="${HUTCH:-2}"
export DATASETS="${DATASETS:-c4,mmlu}"
export PARALLEL="${PARALLEL:-0}"

# 并行时各任务专用 GPU（也可用 GPUS=3,4 自动取前两个）
GPU_C4="${GPU_C4:-${GPUS%%,*}}"
GPU_MMLU="${GPU_MMLU:-$(echo "$GPUS" | cut -d, -f2)}"
GPU_MMLU="${GPU_MMLU:-$GPU_C4}"
GPU_SERIAL="${GPUS%%,*}"

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
# 必须用真 C4；拉不到就报错，禁止悄悄退回 WikiText
export C4_NO_FALLBACK="${C4_NO_FALLBACK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_ARGS=()
[[ -n "${MODEL:-}" ]] && MODEL_ARGS=(--model "$MODEL")
SUPP_COMMON=(
  "${MODEL_ARGS[@]}"
  --n_calib "$N_CALIB"
  --batch_size "$BATCH"
  --seq_len "$SEQ_LEN"
  --surrogate_epochs "$SURR_EP"
  --hutchinson_samples "$HUTCH"
)

_log() { echo "[$(date '+%H:%M:%S')] $*"; }

_train_a() {
  local ds="$1" gpu="$2" scores="$3"
  shift 3
  local -a extra=("$@")
  if [[ -f "$scores/shapley_learned_a.pt" && "${FORCE_RETRAIN_A:-0}" != "1" ]]; then
    _log "Skip $ds scores (已有 $scores/shapley_learned_a.pt)"
    return 0
  fi
  _log "GPU $gpu: supplement_learned_a $ds → $scores"
  CUDA_VISIBLE_DEVICES="$gpu" python3 supplement_learned_a.py \
    "${SUPP_COMMON[@]}" "${extra[@]}" --output_dir "$scores"
}

_downstream_a() {
  local ds="$1" gpu="$2" scores="$3" down="$4" eval_task="$5" cache="$6"
  _log "GPU $gpu: downstream Learned-A $ds ($eval_task) → $down"
  CUDA_VISIBLE_DEVICES="$gpu" python3 evaluate_downstream.py \
    --task "$eval_task" \
    --scores_dir "$scores" \
    --output_dir "$down" \
    --deactivate_fracs "$FRACS" \
    --only_metric shapley_learned_a \
    --eval_cache "$cache" \
    --mmlu_subject "$MMLU_SUBJECT"
  python3 merge_downstream_metric.py \
    --output_dir "$down" \
    --task "$eval_task" \
    --metric shapley_learned_a
  python3 plot_downstream_results.py --input "$down"
}

_run_c4() {
  local gpu="$1"
  if [[ "${C4_NO_FALLBACK:-1}" == "1" ]] && [[ ! -d "${C4_CALIB_DIR:-./data/c4_en}" ]] \
      && ! ls "${C4_CALIB_DIR:-./data/c4_en}"/*.json.gz 1>/dev/null 2>&1; then
    _log "Tip: run 'bash download_c4_calib.sh' once if shard download fails on this node"
  fi
  _train_a c4 "$gpu" ./results_c4 --dataset c4
  _downstream_a c4 "$gpu" ./results_c4 ./results_c4_ppl c4_ppl ./results_c4_ppl/_eval_cache_c4_ppl.json
}

_run_mmlu() {
  local gpu="$1"
  [[ "${SKIP_MMLU:-0}" == "1" ]] && { _log "Skip mmlu (SKIP_MMLU=1)"; return 0; }
  _train_a mmlu "$gpu" ./results_mmlu --dataset mmlu --mmlu_subject "$MMLU_SUBJECT"
  _downstream_a mmlu "$gpu" ./results_mmlu ./results_mmlu_acc mmlu_acc ./results_mmlu_acc/_eval_cache_mmlu_acc.json
}

_wants() {
  local target="$1"
  local d
  IFS=',' read -r -a DS_ARR <<< "$DATASETS"
  for d in "${DS_ARR[@]}"; do
    d="$(echo "$d" | xargs)"
    [[ "$d" == "$target" ]] && return 0
  done
  return 1
}

if [[ "$PARALLEL" == "1" ]]; then
  _log "PARALLEL=1  GPU_C4=$GPU_C4  GPU_MMLU=$GPU_MMLU"
  pids=()
  names=()
  if _wants c4; then
    _run_c4 "$GPU_C4" &
    pids+=($!)
    names+=("c4:$GPU_C4")
  fi
  if _wants mmlu; then
    _run_mmlu "$GPU_MMLU" &
    pids+=($!)
    names+=("mmlu:$GPU_MMLU")
  fi
  fail=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      _log "OK ${names[$i]}"
    else
      _log "FAILED ${names[$i]}"
      fail=1
    fi
  done
  [[ "$fail" -eq 0 ]] || exit 1
else
  IFS=',' read -r -a DS_ARR <<< "$DATASETS"
  for ds in "${DS_ARR[@]}"; do
    ds="$(echo "$ds" | xargs)"
    case "$ds" in
      c4)   _run_c4 "${GPU_C4:-$GPU_SERIAL}" ;;
      mmlu) _run_mmlu "${GPU_MMLU:-$GPU_SERIAL}" ;;
      wikitext)
        if [[ -f ./results_wikitext/shapley_learned_a.pt ]]; then
          _log "Skip wikitext (already has shapley_learned_a.pt)"
        else
          _train_a wikitext "$GPU_SERIAL" ./results_wikitext --dataset wikitext
        fi
        ;;
      *) echo "Unknown dataset: $ds" >&2; exit 1 ;;
    esac
  done
fi

_log "Done."
