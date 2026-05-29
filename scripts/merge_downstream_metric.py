#!/usr/bin/env python3
"""Merge partial_<metric>.json into eval_<task>.json and refresh CSV."""
from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from evaluate_downstream import save_results  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, help="e.g. ./results_c4_ppl")
    p.add_argument("--task", choices=("c4_ppl", "mmlu_acc"), required=True)
    p.add_argument("--metric", required=True, help="e.g. shapley_learned_a")
    args = p.parse_args()

    eval_path = os.path.join(args.output_dir, f"eval_{args.task}.json")
    partial_path = os.path.join(args.output_dir, f"partial_{args.metric}.json")
    bl_path = os.path.join(args.output_dir, "_baseline.json")

    for path in (eval_path, partial_path, bl_path):
        if not os.path.isfile(path):
            print(f"ERROR: missing {path}", file=sys.stderr)
            sys.exit(1)

    with open(eval_path, encoding="utf-8") as f:
        payload = json.load(f)
    with open(partial_path, encoding="utf-8") as f:
        partial = json.load(f)
    with open(bl_path, encoding="utf-8") as f:
        bl = json.load(f)

    payload.setdefault("metrics", {})[args.metric] = partial["rows"]
    extra = {}
    if args.task == "mmlu_acc" and "mmlu_subject" in payload:
        extra["mmlu_subject"] = payload["mmlu_subject"]

    save_results(
        args.output_dir,
        args.task,
        payload["deactivate_fracs"],
        bl["baseline"],
        payload["metrics"],
        extra=extra or None,
    )
    print(f"Merged {args.metric} into {eval_path}")


if __name__ == "__main__":
    main()
