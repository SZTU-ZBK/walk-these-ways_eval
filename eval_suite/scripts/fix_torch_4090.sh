#!/usr/bin/env bash
# RTX 4090（sm_89）需要 torch >= 1.13；torch 1.10+cu113 会报 nvrtc arch 错误。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate_eval_env.sh"

echo "[fix_torch_4090] 升级 PyTorch 1.13 + cu117（兼容 RTX 4090 sm_89）..."
pip install "torch==1.13.1+cu117" "torchvision==0.14.1+cu117" "torchaudio==0.13.1+cu117" \
  --extra-index-url https://download.pytorch.org/whl/cu117

python - <<'PY'
import torch
cap = torch.cuda.get_device_capability()
print(f"torch {torch.__version__}, GPU capability {cap}")
assert cap[0] >= 8, "未检测到 CUDA GPU"
PY

python -c "import isaacgym; from isaacgym import gymapi; print('[fix_torch_4090] isaacgym 导入 OK')"
echo "[fix_torch_4090] 完成。请运行: python scripts/test.py"
