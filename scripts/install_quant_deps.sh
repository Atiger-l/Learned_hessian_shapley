#!/usr/bin/env bash
# Install GPTQ backend for scripts/quantize.py --method gptq
#
#   bash install_quant_deps.sh          # auto-gptq (recommended, lighter)
#   bash install_quant_deps.sh gptqmodel

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-lhs}"

BACKEND="${1:-auto_gptq}"

if [[ "$BACKEND" == "auto_gptq" ]]; then
  pip install -U auto-gptq optimum peft
  python3 -c "from gptq_compat import patch_transformers_for_gptq; patch_transformers_for_gptq(); from auto_gptq import AutoGPTQForCausalLM; print('auto-gptq OK')"
elif [[ "$BACKEND" == "gptqmodel" ]]; then
  pip install -U gptqmodel logbar threadpoolctl tokenicer device_smi random_word
  python3 -c "from gptq_compat import patch_transformers_for_gptq; patch_transformers_for_gptq(); from gptqmodel import GPTQModel; print('gptqmodel OK')"
else
  echo "Usage: bash install_quant_deps.sh [auto_gptq|gptqmodel]"
  exit 1
fi
