#!/usr/bin/env python3
"""Prefetch MMLU subject parquet via hf-mirror (dev + test + validation)."""
from __future__ import annotations

import argparse
import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from hf_calibration import MMLU_SPLITS, download_mmlu_subject  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("subject", default="high_school_mathematics", nargs="?")
    p.add_argument("--splits", default=",".join(MMLU_SPLITS))
    args = p.parse_args()
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    paths = download_mmlu_subject(args.subject, splits=splits)
    print("OK:", paths)


if __name__ == "__main__":
    main()
