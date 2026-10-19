#!/usr/bin/env bash
# Install the SWELLS training stack on a fresh GCP Deep Learning VM.
set -euxo pipefail

WORK=${WORK:-/opt/swells}
sudo mkdir -p "$WORK"
sudo chown -R "$(id -u):$(id -g)" "$WORK"
cd "$WORK"

# The DLVM image installs the driver on first boot.
for _ in $(seq 1 60); do nvidia-smi && break || sleep 20; done
nvidia-smi

sudo apt-get update -qq
sudo apt-get install -y -qq p7zip-full git tmux jq

python3 -m pip install --upgrade pip wheel

# vllm pins a torch build, so install it first and let everything else follow.
pip install "vllm==0.10.1.1"
pip install "trl==0.21.0" "peft>=0.14" "transformers>=4.55" "accelerate>=1.4" \
            "datasets>=3.0" sacrebleu pandas pyarrow

# Optional: flash-attn is a large source build on some images; the training
# scripts fall back to sdpa when it is missing.
pip install flash-attn --no-build-isolation || echo "flash-attn skipped"

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i} {p.name} {p.total_memory/1e9:.0f} GB")
PY

echo "setup complete"
