#!/usr/bin/env bash
# 激活 walk-these-ways 形态学评估专用环境（独立 isaacgym conda，非 isaaclab）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
ISAACGYM_ENV="${ISAACGYM_ENV:-${AUTODL_ROOT}/conda_envs/isaacgym}"
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
[ -f /etc/network_turbo ] && source /etc/network_turbo
[ -f "${CONDA_SH}" ] || { echo "找不到 conda: ${CONDA_SH}"; exit 1; }
source "${CONDA_SH}"

if [[ ! -d "${ISAACGYM_ENV}/bin" ]]; then
  echo "[activate_eval_env] 未找到 ${ISAACGYM_ENV}"
  echo "  请先运行: bash eval_suite/scripts/setup_isaacgym_conda.sh"
  exit 1
fi

conda activate "${ISAACGYM_ENV}"
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libgomp.so.1}"
# Isaac Gym 的 gym_38.so 需要找到 conda 里的 libpython3.8.so
export LD_LIBRARY_PATH="${ISAACGYM_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

echo "[activate_eval_env] python: $(which python) ($(python -V 2>&1))"
echo "[activate_eval_env] 仓库:   ${REPO_ROOT}"

if ! python -c "import isaacgym" 2>/dev/null; then
  echo "[activate_eval_env] 提示：isaacgym 尚未安装，请运行 install_isaacgym_hint.sh"
fi
