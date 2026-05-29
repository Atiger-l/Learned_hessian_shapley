#!/usr/bin/env python3
"""
Evaluate a quantized / Shapley-mixed model: C4 PPL and/or MMLU accuracy (generate).

  python3 evaluate_quant.py --model ./results_c4/quant_gptq_w8 --task c4_ppl
  python3 evaluate_quant.py --model ./results_c4/quant_shapley_a_w8_f0.10 --task mmlu_acc \\
    --mmlu_subject high_school_mathematics
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from evaluate import _DEFAULT_MODEL, evaluate_perplexity
from evaluate_downstream import evaluate_mmlu_accuracy, load_mmlu_test
from hf_calibration import load_c4_texts
from model_utils import load_tokenizer_and_causal_lm


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_quant_model(model_path: str):
    """Load HF FP16/BF16 or GPTQ checkpoint. Returns (tokenizer, model)."""
    from transformers import AutoTokenizer

    manifest_path = os.path.join(model_path, "quant_manifest.json")
    method = "hf"
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        method = manifest.get("method", "hf")

    is_gptq = method == "gptq" or os.path.isfile(os.path.join(model_path, "quantize_config.json"))
    if is_gptq:
        from gptq_compat import patch_transformers_for_gptq

        patch_transformers_for_gptq()
        try:
            from gptqmodel import GPTQModel

            _log(f"Loading GPTQ model (gptqmodel) from {model_path}")
            model = GPTQModel.from_quantized(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            return tokenizer, model
        except Exception as e1:
            _log(f"gptqmodel load failed ({e1}), trying auto-gptq …")
            try:
                from auto_gptq import AutoGPTQForCausalLM

                model = AutoGPTQForCausalLM.from_quantized(
                    model_path,
                    device="cuda:0",
                    use_triton=False,
                    trust_remote_code=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                return tokenizer, model
            except Exception as e2:
                sys.exit(
                    f"Could not load GPTQ model at {model_path}.\n"
                    f"  gptqmodel: {e1}\n  auto-gptq: {e2}\n"
                    "Install: pip install gptqmodel  OR  pip install auto-gptq"
                )

    _log(f"Loading HF weights from {model_path}")
    return load_tokenizer_and_causal_lm(model_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Quantized model dir (HF or GPTQ).")
    p.add_argument("--task", choices=("c4_ppl", "mmlu_acc"), required=True)
    p.add_argument("--output_dir", default="")
    p.add_argument("--mmlu_subject", default="high_school_mathematics")
    p.add_argument("--mmlu_max_items", type=int, default=0, help="0 = all test items.")
    p.add_argument("--n_eval_texts", type=int, default=128)
    p.add_argument("--max_eval_chars", type=int, default=50_000)
    p.add_argument("--mmlu_scoring", default="generate", choices=("generate", "logprob"))
    args = p.parse_args()

    tokenizer, model = load_quant_model(args.model)

    result: dict = {"model": os.path.abspath(args.model), "task": args.task}

    if args.task == "c4_ppl":
        texts = load_c4_texts(
            args.n_eval_texts,
            split="validation",
            max_chars=args.max_eval_chars,
            allow_wikitext_fallback=True,
        )
        ppl = evaluate_perplexity(model, tokenizer, texts[: args.n_eval_texts])
        result["ppl"] = ppl
        result["n_texts"] = min(len(texts), args.n_eval_texts)
        _log(f"  PPL={ppl:.2f}  (n={result['n_texts']})")
    else:
        items = load_mmlu_test(
            args.mmlu_subject,
            None if args.mmlu_max_items <= 0 else args.mmlu_max_items,
        )
        acc, n = evaluate_mmlu_accuracy(
            model,
            tokenizer,
            items,
            scoring=args.mmlu_scoring,
        )
        result["accuracy"] = acc
        result["n_items"] = n
        result["mmlu_subject"] = args.mmlu_subject
        result["mmlu_scoring"] = args.mmlu_scoring
        _log(f"  acc={100 * acc:.2f}%  (n={n}, scoring={args.mmlu_scoring})")

    out_dir = args.output_dir or args.model
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eval_{args.task}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    _log(f"Saved {out_path}")


if __name__ == "__main__":
    main()
