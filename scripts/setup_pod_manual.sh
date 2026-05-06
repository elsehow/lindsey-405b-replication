#!/usr/bin/env bash
# Manual pod-setup script for when Nick provisions a RunPod pod by hand
# (auto_sweep.py handles the auto-poll + auto-deploy case).
#
# Usage: ssh into the pod, then run this script. Or scp + execute remotely:
#   scp -P <port> -i ~/.ssh/id_ed25519 setup_pod_manual.sh root@<ip>:/workspace/
#   ssh -p <port> -i ~/.ssh/id_ed25519 root@<ip> 'bash /workspace/setup_pod_manual.sh'
#
# What it does (in order):
#   1. Pre-flight: probe libcuda for driver version; abort if < 12081.
#   2. Install uv (single static binary).
#   3. Create venv at /workspace/venv with Python 3.12 (uv pulls 3.12 if needed —
#      e.g. RunPod images shipping Python 3.11 only).
#   4. Install the pinned stack: vllm 0.20.0, vllm-lens 1.1.0, anthropic, pydantic,
#      huggingface_hub[cli,hf_transfer].
#   5. Verify imports work end-to-end.
#
# After this, run scripts manually:
#   scp lindsey_full_sweep.py judge_lindsey_batch.py to /workspace/
#   /workspace/venv/bin/hf download RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic ...
#   VLLM_WORKER_MULTIPROC_METHOD=spawn /workspace/venv/bin/python lindsey_full_sweep.py

set -euo pipefail

MIN_DRIVER=12081  # PyPI torch 2.8 cu128 wheel needs CUDA driver ≥ 12.8.1

echo "=== 1. Driver pre-flight ==="
DRIVER=$(python3 -c "import ctypes; lib=ctypes.CDLL('libcuda.so'); v=ctypes.c_int(); lib.cuDriverGetVersion(ctypes.byref(v)); print(v.value)")
echo "  CUDA driver version: $DRIVER"
if [ "$DRIVER" -lt "$MIN_DRIVER" ]; then
    echo "  FAIL: driver $DRIVER < required $MIN_DRIVER. PyPI torch 2.8 cu128 wheel will crash workers."
    echo "  See https://github.com/elsehow/lindsey-405b-replication/scripts/auto_sweep.py for context."
    exit 1
fi
echo "  OK"

echo "=== 2. Install uv ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "=== 3. Create venv (Python 3.12) ==="
# --seed bundles pip; --python 3.12 forces 3.12 even if system only has 3.11
# (uv downloads its own 3.12 if needed).
uv venv /workspace/venv --python 3.12 --seed

echo "=== 4. Install pinned stack ==="
# vllm 0.20.0: pinned exactly. 0.20.1 (released 2026-05-04) added a hard
#   deep_gemm dep that breaks vllm-lens 1.1.0 paths.
# vllm-lens 1.1.0: requires Python 3.12 (won't install on 3.11).
# huggingface_hub[cli,hf_transfer]: provides the `hf` CLI + Rust accelerator.
uv pip install --python /workspace/venv/bin/python \
    "vllm==0.20.0" \
    "vllm-lens==1.1.0" \
    "anthropic>=0.40" \
    "pydantic>=2" \
    "huggingface_hub[cli,hf_transfer]"

echo "=== 5. Verify ==="
/workspace/venv/bin/python -c "
import torch, vllm, vllm_lens, anthropic
print('torch       :', torch.__version__, '| cuda:', torch.cuda.is_available(), '| devices:', torch.cuda.device_count())
print('vllm        :', vllm.__version__)
print('vllm_lens   :', vllm_lens.__version__)
print('anthropic   :', anthropic.__version__)
"
echo "  OK"
echo
echo "Setup complete. Next steps:"
echo "  scp -P <port> -i ~/.ssh/id_ed25519 lindsey_full_sweep.py judge_lindsey_batch.py root@<ip>:/workspace/"
echo "  Then download weights + run sweep (see runpod-vllm-fp8.md)."
