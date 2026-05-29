#!/usr/bin/env bash
# 预下载 C4 校准用单个 train shard（约 300MB），供离线 / 补 A 使用。
#
#   bash download_c4_calib.sh
#   C4_SHARD=0 bash download_c4_calib.sh
#
# 完成后运行（禁止 WikiText fallback）:
#   C4_NO_FALLBACK=1 DATASETS=c4 GPUS=6 bash supplement_learned_a.sh

set -euo pipefail
cd "$(dirname "$0")"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_BASE_URL="${HUGGINGFACE_HUB_BASE_URL:-$HF_ENDPOINT}"
export HF_HUB_ENDPOINT="${HF_HUB_ENDPOINT:-$HF_ENDPOINT}"
export C4_SHARD="${C4_SHARD:-0}"

DEST="${C4_CALIB_DIR:-$(pwd)/data/c4_en}"
mkdir -p "$DEST"

if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-lhs}"
fi

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "Downloading C4 shard $C4_SHARD → $DEST"

python3 download_c4_calib.py

echo ""
echo "Test load:"
C4_NO_FALLBACK=1 python3 -c "
from hf_calibration import load_c4_texts
t = load_c4_texts(8, max_chars=20000)
print('sample:', len(t), 'texts; first 80 chars:', (t[0][:80] if t else 'EMPTY'))
"

echo ""
echo "Next: C4_NO_FALLBACK=1 DATASETS=c4 GPUS=<id> bash supplement_learned_a.sh"
