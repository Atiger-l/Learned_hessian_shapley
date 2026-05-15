#!/usr/bin/env python3
"""Download Qwen weights into Learned_hessian_shapley/Qwen2.5-3B-Instruct/ (repo root).

Requires outbound HTTPS to Hugging Face (or a mirror).

Usage (Debian/Ubuntu often has no `python` — use python3):

  # One-time: enable `conda activate` in this shell (pick the path that exists)
  source /opt/miniconda3/etc/profile.d/conda.sh   # or: ~/miniconda3/... or ~/anaconda3/...
  conda activate lhs                              # env name is lhs, not lhse

  pip install -U huggingface_hub
  python3 scripts/download_qwen.py --dest "$(pwd)/Qwen2.5-3B-Instruct"

Through HTTP proxy (if your lab provides one):
  export HTTPS_PROXY=http://proxy.example.com:8080
  export HTTP_PROXY=http://proxy.example.com:8080
  python3 scripts/download_qwen.py --dest "$(pwd)/Qwen2.5-3B-Instruct"
  # equivalent: python3 scripts/download_qwen.py --proxy http://proxy.example.com:8080 ...

China mirror (example):
  export HF_ENDPOINT=https://hf-mirror.com
  python3 scripts/download_qwen.py

CLI equivalent (new Hub CLI — use ``hf``, not ``huggingface-cli``):
  hf download Qwen/Qwen2.5-3B-Instruct \\
    --local-dir Qwen2.5-3B-Instruct

If you see [Errno 101] Network is unreachable:
  Many clusters block outbound internet on compute nodes (hostname often like "slave").
  Download on a machine WITH internet (login node / laptop), then copy the folder:

  Laptop / PC (any folder):
    pip install -U huggingface_hub
    hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./Qwen2.5-3B-Instruct

  Upload to server (pick one):
    scp -r ./Qwen2.5-3B-Instruct chengjialin_intern@slave:~/work/work/Learned_hessian_shapley/
    rsync -avz --progress ./Qwen2.5-3B-Instruct/ \\
      chengjialin_intern@slave:~/work/work/Learned_hessian_shapley/Qwen2.5-3B-Instruct/

  On server, run with:
    --model ~/work/work/Learned_hessian_shapley/Qwen2.5-3B-Instruct

  Or use your site's HTTP proxy (must be reachable FROM this node):
    export HTTPS_PROXY=http://internal-proxy.company.com:8080
    export HTTP_PROXY=http://internal-proxy.company.com:8080
    python3 scripts/download_qwen.py ...
    # or: python3 scripts/download_qwen.py --proxy http://internal-proxy:8080 ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub import snapshot_download


_OFFLINE_HINT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
下载失败：当前机器访问不了外网 (Network is unreachable / errno 101)。
常见于：在集群「计算节点」上跑脚本，而出站 HTTPS 被禁用。

可行做法（任选）：
  1) 在有外网的机器上下载同一目录，再 rsync/scp 到本仓库根目录下 Qwen2.5-3B-Instruct/
  2) 在集群「登录节点」(login) 上执行本脚本（若登录节点可访问 Hugging Face）
  3) HTTP 代理（集群常见）：export HTTPS_PROXY=... 且 HTTP_PROXY=...，或 --proxy URL；
     前提：本机能连到代理；代理能访问 huggingface.co（或 HF_ENDPOINT 镜像）。
  4) 国内可试镜像（仍需能访问镜像站）：export HF_ENDPOINT=https://hf-mirror.com
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument(
        "--dest",
        default=str(root / "Qwen2.5-3B-Instruct"),
        help="Directory to materialize files (default: <repo_root>/Qwen2.5-3B-Instruct)",
    )
    p.add_argument(
        "--proxy",
        default="",
        metavar="URL",
        help="HTTP(S) proxy for this run, e.g. http://127.0.0.1:7890 or http://user:pass@internal-proxy:8080 "
        "(sets HTTPS_PROXY and HTTP_PROXY unless already set)",
    )
    args = p.parse_args()
    if args.proxy:
        os.environ.setdefault("HTTPS_PROXY", args.proxy)
        os.environ.setdefault("HTTP_PROXY", args.proxy)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id=args.repo, local_dir=str(dest))
    except LocalEntryNotFoundError as e:
        err = str(e)
        if (
            "101" in err
            or "unreachable" in err.lower()
            or "ConnectError" in err
            or "Connection refused" in err
        ):
            print(_OFFLINE_HINT, file=sys.stderr)
        raise SystemExit(1) from e
    except OSError as e:
        if getattr(e, "errno", None) == 101:
            print(_OFFLINE_HINT, file=sys.stderr)
            raise SystemExit(1) from e
        raise
    print(f"Downloaded {args.repo} -> {dest.resolve()}")


if __name__ == "__main__":
    main()
