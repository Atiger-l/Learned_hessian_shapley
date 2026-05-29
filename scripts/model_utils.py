"""Shared Transformers load logic (single-GPU default; Hessian / Hutchinson need consistent placement)."""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer_and_causal_lm(model_path: str):
    """
    Default: single GPU ``device_map=None`` then ``.to(\"cuda\")`` — matches ``neural_function``
    second-order autograd and avoids CPU-reported embed weights with ``device_map=\"auto\"``.

    Opt-in to accelerate sharding: ``LHS_DEVICE_MAP=auto``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Eager attention: fused SDPA / flash lack double-backward (Hutchinson / HVP).
    common = dict(trust_remote_code=True, attn_implementation="eager")

    if torch.cuda.is_available():
        common["torch_dtype"] = torch.bfloat16
        use_auto = os.environ.get("LHS_DEVICE_MAP", "").strip().lower() == "auto"
        if use_auto:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map="auto", **common
                )
            except ValueError as e:
                err = str(e).lower()
                if "accelerate" not in err and "device_map" not in err:
                    raise
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map=None, **common
                ).to("cuda")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map=None, **common
            ).to("cuda")
    else:
        print(
            "WARNING: CUDA not available — this pipeline expects GPU (.cuda() in neural_function). "
            "Fix driver/PyTorch CUDA mismatch or run on a GPU node."
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=None, torch_dtype=torch.float32, **common
        )

    return tokenizer, model
