#!/usr/bin/env bash
# 为 walk-these-ways / Isaac Gym Preview 4 创建独立 conda 环境。
# 不要与 isaaclab（Isaac Sim / Python 3.10）混用。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
CONDA_ENV="${CONDA_ENV:-${AUTODL_ROOT}/conda_envs/isaacgym}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
[ -f /etc/network_turbo ] && source /etc/network_turbo
[ -f "${CONDA_SH}" ] || { echo "找不到 conda: ${CONDA_SH}"; exit 1; }
source "${CONDA_SH}"

# pycurl（ml_logger 依赖）需要 libcurl 开发头文件
if ! command -v curl-config >/dev/null 2>&1; then
  echo "[setup] 安装 libcurl 开发包（pycurl 编译依赖）..."
  apt-get update -qq && apt-get install -y -qq libcurl4-openssl-dev
fi

mkdir -p "${AUTODL_ROOT}/conda_envs"

if [[ ! -d "${CONDA_ENV}/bin" ]]; then
  echo "[setup] 创建 conda 环境: ${CONDA_ENV} (python=${PYTHON_VERSION})"
  conda create -y -p "${CONDA_ENV}" "python=${PYTHON_VERSION}"
else
  echo "[setup] 环境已存在: ${CONDA_ENV}"
fi

conda activate "${CONDA_ENV}"
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libgomp.so.1}"

echo "[setup] 安装 PyTorch ..."
pip install --upgrade pip

# RTX 4090（sm_89）与 torch 1.10+cu113 不兼容，需 1.13+cu117
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
if [[ "${GPU_NAME}" == *"4090"* ]] || [[ "${GPU_NAME}" == *"4080"* ]] || [[ "${GPU_NAME}" == *"L40"* ]]; then
  echo "[setup] 检测到 Ada 架构 GPU（${GPU_NAME}），使用 torch 1.13 + cu117"
  pip install "torch==1.13.1+cu117" "torchvision==0.14.1+cu117" "torchaudio==0.13.1+cu117" \
    --extra-index-url https://download.pytorch.org/whl/cu117
else
  echo "[setup] 使用 torch 1.10 + cu113"
  pip install "torch==1.10.0+cu113" "torchvision==0.11.1+cu113" "torchaudio==0.10.0+cu113" \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
fi

echo "[setup] 安装 go1_gym 依赖 ..."
pip install "numpy==1.23.5" tqdm matplotlib pyyaml \
  "ml_logger==0.8.117" "ml_dash==0.3.20" "jaynes>=0.9.2" "params-proto==2.10.5" "gym>=0.14.0"

echo "[setup] 安装本仓库 ..."
pip install -e "${REPO_ROOT}"

echo "[setup] conda 环境就绪。下一步："
echo "  1) bash eval_suite/scripts/install_isaacgym_hint.sh"
echo "  2) （RTX 4090 若仍报 nvrtc 错误）bash eval_suite/scripts/fix_torch_4090.sh"
echo "  3) source eval_suite/scripts/activate_eval_env.sh && python scripts/test.py"
