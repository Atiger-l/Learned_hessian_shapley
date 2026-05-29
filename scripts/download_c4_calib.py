#!/usr/bin/env python3
"""Download one C4 en train shard for calibration (mirror-friendly)."""
from __future__ import annotations

import os
import shutil
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from hf_calibration import _apply_hf_mirror_env, _c4_train_shard_name  # noqa: E402


def main() -> None:
    _apply_hf_mirror_env()
    from huggingface_hub import hf_hub_download

    shard = int(os.environ.get("C4_SHARD", "0"))
    rel = _c4_train_shard_name(shard)
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    dest_dir = os.environ.get(
        "C4_CALIB_DIR",
        os.path.join(_SCRIPTS, "data", "c4_en"),
    )
    os.makedirs(dest_dir, exist_ok=True)

    print(f"HF_ENDPOINT={endpoint}")
    print(f"Downloading {rel} ...")
    try:
        path = hf_hub_download(
            repo_id="allenai/c4",
            repo_type="dataset",
            filename=rel,
            endpoint=endpoint,
        )
    except TypeError:
        path = hf_hub_download(
            repo_id="allenai/c4",
            repo_type="dataset",
            filename=rel,
        )

    out = os.path.join(dest_dir, os.path.basename(rel))
    if os.path.abspath(path) != os.path.abspath(out):
        shutil.copy2(path, out)
    print(f"OK: {out}")


if __name__ == "__main__":
    main()
