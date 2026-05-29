#!/usr/bin/env python3
"""
Post-training quantization for Qwen2.5-3B-Instruct.

Methods
-------
  shapley_row   Shapley-guided row mixed INT (no GPTQ dependency).
                Bottom `quantize_frac` neuron rows (by |score|) get fake INT8;
                top rows stay FP16/BF16. Saved as standard HF weights.

  gptq          Full-model GPTQ baseline via pip package `gptqmodel`
                (recommended) or `auto-gptq`. **No need to clone GPTQ source**
                if pip install succeeds.

Install (pick one)
------------------
  pip install gptqmodel          # recommended for Qwen2.5
  # or
  pip install auto-gptq optimum

If pip fails on compute node:
  pip install git+https://github.com/ModelCloud/GPTQModel.git
  # legacy fallback:
  git clone https://github.com/AutoGPTQ/AutoGPTQ && cd AutoGPTQ && pip install -e .

Examples
--------
  # Shapley-Learned-A: quantize bottom 10% rows (scores from C4 run)
  python3 quantize.py --method shapley_row --scores_dir ./results_c4 \\
    --metric shapley_learned_a --quantize_frac 0.10 --dataset c4 \\
    --output_dir ./results_c4/quant_shapley_a_w8_f0.10

  # GPTQ INT8 baseline (same C4 calibration)
  python3 quantize.py --method gptq --wbits 8 --dataset c4 \\
    --output_dir ./results_c4/quant_gptq_w8
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from evaluate import _DEFAULT_MODEL, load_scores
from model_utils import load_tokenizer_and_causal_lm
from quantization_utils import (
    apply_shapley_row_quant,
    apply_uniform_row_quant,
    load_calibration_texts,
    save_hf_model,
    save_quant_manifest,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_shapley_row(args) -> None:
    if not args.scores_dir:
        sys.exit("--scores_dir required for shapley_row")
    scores = load_scores(args.scores_dir, args.metric)
    tokenizer, model = load_tokenizer_and_causal_lm(args.model)

    stats = apply_shapley_row_quant(
        model,
        scores,
        quantize_frac=args.quantize_frac,
        bits=args.wbits,
    )
    manifest = {
        "method": "shapley_row",
        "model": args.model,
        "scores_dir": os.path.abspath(args.scores_dir),
        "metric": args.metric,
        "dataset": args.dataset,
        "mmlu_subject": args.mmlu_subject,
        **stats,
    }
    save_hf_model(model, tokenizer, args.output_dir)
    save_quant_manifest(args.output_dir, manifest)
    _log(
        f"Shapley row quant: kept {stats['kept_rows_fp16']} FP16 rows, "
        f"fake-quant {stats['quantized_rows']} rows @ {args.wbits}-bit"
    )


def run_uniform_row(args) -> None:
    if not args.scores_dir:
        sys.exit("--scores_dir required for uniform_row")
    scores = load_scores(args.scores_dir, args.metric)
    tokenizer, model = load_tokenizer_and_causal_lm(args.model)

    stats = apply_uniform_row_quant(model, scores, bits=args.wbits)
    manifest = {
        "method": "uniform_row",
        "model": args.model,
        "scores_dir": os.path.abspath(args.scores_dir),
        "metric": args.metric,
        "dataset": args.dataset,
        "mmlu_subject": args.mmlu_subject,
        **stats,
    }
    save_hf_model(model, tokenizer, args.output_dir)
    save_quant_manifest(args.output_dir, manifest)
    _log(
        f"Uniform row quant: fake-quant all {stats['quantized_rows']} rows @ {args.wbits}-bit"
    )


def _run_gptq_gptqmodel(model_path: str, texts: list[str], output_dir: str, wbits: int, group_size: int, batch_size: int):
    from gptq_compat import patch_transformers_for_gptq

    patch_transformers_for_gptq()
    try:
        from gptqmodel import GPTQModel
        from gptqmodel.quantization import QuantizeConfig
    except ImportError as e:
        raise ImportError(
            "gptqmodel not installed or missing deps. Run:\n"
            "  pip install gptqmodel logbar threadpoolctl tokenicer device_smi\n"
            "or use: --gptq_backend auto_gptq\n"
            f"({e})"
        ) from e

    quant_config = QuantizeConfig(
        bits=wbits,
        group_size=group_size,
        desc_act=False,
    )
    _log(f"[gptqmodel] loading {model_path} …")
    model = GPTQModel.load(model_path, quant_config, trust_remote_code=True)
    _log(f"[gptqmodel] quantizing with {len(texts)} calib texts …")
    model.quantize(texts, batch_size=batch_size)
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    _log(f"[gptqmodel] saved → {output_dir}")


def _run_gptq_auto_gptq(model_path: str, texts: list[str], output_dir: str, wbits: int, group_size: int):
    from gptq_compat import patch_transformers_for_gptq

    patch_transformers_for_gptq()
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "auto-gptq not installed. Run:\n"
            "  pip install auto-gptq optimum\n"
            "or clone:\n"
            "  git clone https://github.com/AutoGPTQ/AutoGPTQ && cd AutoGPTQ && pip install -e .\n"
            f"({e})"
        ) from e

    quant_config = BaseQuantizeConfig(
        bits=wbits,
        group_size=group_size,
        desc_act=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _log(f"[auto-gptq] loading {model_path} …")
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quant_config,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device
    examples = []
    for t in texts:
        batch = tokenizer(t, return_tensors="pt")
        examples.append({k: v.to(device) for k, v in batch.items()})
    _log(f"[auto-gptq] quantizing with {len(examples)} calib texts on {device} …")
    model.quantize(examples, cache_examples_on_gpu=True)
    os.makedirs(output_dir, exist_ok=True)
    model.save_quantized(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)
    _log(f"[auto-gptq] saved → {output_dir}")


def run_gptq(args) -> None:
    texts = load_calibration_texts(
        args.dataset,
        args.n_calib,
        mmlu_subject=args.mmlu_subject,
        mmlu_split=args.mmlu_split,
    )
    _log(f"Calibration: dataset={args.dataset} n={len(texts)}")

    backend = args.gptq_backend
    if backend == "auto":
        try:
            from auto_gptq import AutoGPTQForCausalLM  # noqa: F401

            backend = "auto_gptq"
        except ImportError:
            backend = "gptqmodel"

    if backend == "gptqmodel":
        _run_gptq_gptqmodel(
            args.model,
            texts,
            args.output_dir,
            wbits=args.wbits,
            group_size=args.group_size,
            batch_size=args.batch_size,
        )
    else:
        _run_gptq_auto_gptq(
            args.model,
            texts,
            args.output_dir,
            wbits=args.wbits,
            group_size=args.group_size,
        )

    manifest = {
        "method": "gptq",
        "gptq_backend": backend,
        "model": args.model,
        "dataset": args.dataset,
        "mmlu_subject": args.mmlu_subject,
        "n_calib": len(texts),
        "wbits": args.wbits,
        "group_size": args.group_size,
    }
    save_quant_manifest(args.output_dir, manifest)


def main() -> None:
    p = argparse.ArgumentParser(description="Quantize Qwen2.5 with GPTQ or Shapley-guided row INT.")
    p.add_argument(
        "--method",
        choices=("shapley_row", "uniform_row", "gptq"),
        required=True,
        help="shapley_row: partial Shapley-guided mixed INT; uniform_row: fake INT8 all rows; gptq: full GPTQ.",
    )
    p.add_argument("--model", default=_DEFAULT_MODEL)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--scores_dir", default="", help="Required for shapley_row (existing *.pt scores).")
    p.add_argument(
        "--metric",
        default="shapley_fisher",
        choices=("shapley_fisher", "shapley_learned_a", "shapley_learned_b", "gradient"),
    )
    p.add_argument(
        "--quantize_frac",
        type=float,
        default=0.10,
        help="For shapley_row: fraction of lowest-|score| rows to fake-quant (default 0.10).",
    )
    p.add_argument("--wbits", type=int, default=8, choices=(4, 8))
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--dataset", default="c4", choices=("wikitext", "c4", "mmlu"))
    p.add_argument("--mmlu_subject", default="high_school_mathematics")
    p.add_argument("--mmlu_split", default="test")
    p.add_argument("--n_calib", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=1, help="GPTQ calib batch size.")
    p.add_argument(
        "--gptq_backend",
        choices=("auto", "gptqmodel", "auto_gptq"),
        default="auto",
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.method == "shapley_row":
        run_shapley_row(args)
    elif args.method == "uniform_row":
        run_uniform_row(args)
    else:
        run_gptq(args)


if __name__ == "__main__":
    main()
