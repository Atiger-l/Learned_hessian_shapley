#!/usr/bin/env bash
# 全流程：训练 scheme A 代理 + 真实校准数据 + Fisher/Learned Shapley（任意一步失败即中止）
#
# 用法（在任意目录）：
#   bash /path/to/Learned_hessian_shapley/scripts/命令行.sh
# 或先 cd 到 scripts 再：
#   bash 命令行.sh
#
# 常用环境变量（可选）：
#   PIPELINE=wikitext   # wikitext | mmlu | c4 | all（默认 wikitext；all 会依次跑三种，耗时很长）
#   LEARNED_SCHEME=b    # a | b | both — scheme B 为 HVP 代理；OOM 可试 SEQ_LEN=32 N_CALIB=32
#   SURR_BATCHES_PER_EPOCH=16  # 可选：方案 B 每 epoch 用几条 calibration batch 堆 teacher（不设则 Python 默认 8）
#   CUDA_VISIBLE_DEVICES=3    # 默认只用物理 2 号卡；需多卡时再设为 2,3 等
#   USE_OFFICIAL_HF=1                  # 直连官方 huggingface.co（默认改用镜像，见脚本内）
#   HF_ENDPOINT=https://xxxx           # 自定义镜像（不设则默认 hf-mirror）
#   CONDA_ENV=lhs                       # 若使用 conda，要激活的环境名
#   SKIP_CONDA=1                        # 已手动 conda activate 时可设为 1 跳过脚本内激活

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 可选：自动激活 conda 环境 ─────────────────────────────────────────────
if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-lhs}"
fi

# ── GPU / Hugging Face ────────────────────────────────────────────────────
# 默认只用物理 GPU 2（进程内为 cuda:0）。多卡示例: CUDA_VISIBLE_DEVICES=2,3 且 LHS_DEVICE_MAP=auto
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
echo "CUDA: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (单卡时进程内即 cuda:0)"
# 计算节点常出现 [Errno 101] 无法访问 huggingface.co；默认走 hf-mirror（与 huggingface_hub 兼容）
# 需要官方源时：USE_OFFICIAL_HF=1 bash 命令行.sh
if [[ "${USE_OFFICIAL_HF:-0}" == "1" ]]; then
  unset HF_ENDPOINT
  echo "HF: 使用官方 Hub（未设置 HF_ENDPOINT）"
else
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  echo "HF: HF_ENDPOINT=$HF_ENDPOINT"
fi

# ── 预检：Python 与 CUDA ───────────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || { echo "ERROR: 未找到 python3"; exit 1; }
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用，请先修好 PyTorch/驱动'; print('OK:', torch.__version__, torch.version.cuda)"

# ── 共用超参（Hutchinson 二阶梯度极耗显存：默认保守；加大请逐步试 SEQ_LEN / BATCH）──────────────
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
N_CALIB="${N_CALIB:-64}"
BATCH="${BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-48}"
SURR_EP="${SURR_EP:-30}"
HUTCH="${HUTCH:-2}"
SURR_BATCHES_PER_EPOCH="${SURR_BATCHES_PER_EPOCH:-}"
LEARNED_SCHEME="${LEARNED_SCHEME:-a}"
MODEL="${MODEL:-}"  # 空则使用 run_experiment.py 默认（仓库根下 Qwen2.5-3B-Instruct）

_BPE_SHOW="${SURR_BATCHES_PER_EPOCH:-8}"
echo "运行超参: BATCH=$BATCH SEQ_LEN=$SEQ_LEN N_CALIB=$N_CALIB SURR_EP=$SURR_EP HUTCH=$HUTCH LEARNED_SCHEME=$LEARNED_SCHEME SURR_BATCHES_PER_EPOCH=$_BPE_SHOW PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

MODEL_ARGS=()
if [[ -n "$MODEL" ]]; then
  MODEL_ARGS=(--model "$MODEL")
fi

SURR_BPE_ARGS=()
if [[ -n "$SURR_BATCHES_PER_EPOCH" ]]; then
  SURR_BPE_ARGS=(--surrogate_batches_per_epoch "$SURR_BATCHES_PER_EPOCH")
fi

COMMON_ARGS=(
  "${MODEL_ARGS[@]}"
  --n_calib "$N_CALIB"
  --batch_size "$BATCH"
  --seq_len "$SEQ_LEN"
  --surrogate_epochs "$SURR_EP"
  --hutchinson_samples "$HUTCH"
  --learned_scheme "$LEARNED_SCHEME"
  "${SURR_BPE_ARGS[@]}"
)

run_wikitext() {
  echo "========== [WikiText-103 全流程] =========="
  python3 run_experiment.py \
    "${COMMON_ARGS[@]}" \
    --dataset wikitext \
    --output_dir ./results_wikitext
}

run_mmlu() {
  echo "========== [MMLU 全流程] =========="
  python3 run_experiment.py \
    "${COMMON_ARGS[@]}" \
    --dataset mmlu \
    --mmlu_subject "${MMLU_SUBJECT:-abstract_algebra}" \
    --output_dir ./results_mmlu
}

run_c4() {
  echo "========== [C4 全流程] =========="
  python3 run_experiment.py \
    "${COMMON_ARGS[@]}" \
    --dataset c4 \
    --output_dir ./results_c4
}

# ── 自定义 HF 数据集示例（默认不执行）：设置 RUN_CUSTOM=1 并填写路径 ───────
run_custom() {
  [[ "${RUN_CUSTOM:-0}" == "1" ]] || return 0
  : "${HF_DATASET_PATH:?请设置 HF_DATASET_PATH，例如 export HF_DATASET_PATH=org/name}"
  echo "========== [custom 全流程] =========="
  python3 run_experiment.py \
    "${COMMON_ARGS[@]}" \
    --dataset custom \
    --hf_dataset_path "$HF_DATASET_PATH" \
    --hf_dataset_config "${HF_DATASET_CONFIG:-}" \
    --hf_split "${HF_SPLIT:-train[:1000]}" \
    --hf_text_column "${HF_TEXT_COLUMN:-text}" \
    --output_dir ./results_custom
}

PIPELINE="${PIPELINE:-wikitext}"

case "$PIPELINE" in
  wikitext)
    run_wikitext
    ;;
  mmlu)
    run_mmlu
    ;;
  c4)
    run_c4
    ;;
  all)
    run_wikitext
    run_mmlu
    run_c4
    ;;
  *)
    echo "ERROR: 未知 PIPELINE=$PIPELINE（应为 wikitext | mmlu | c4 | all）"
    exit 1
    ;;
esac

run_custom

echo "========== 全部完成 =========="
