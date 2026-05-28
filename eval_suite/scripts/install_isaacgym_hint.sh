#!/usr/bin/env bash
# 若已有 Isaac Gym Preview 4 压缩包，则自动解压并安装。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
ISAACGYM_ENV="${ISAACGYM_ENV:-${AUTODL_ROOT}/conda_envs/isaacgym}"

if [[ ! -d "${ISAACGYM_ENV}/bin" ]]; then
  echo "[install_isaacgym] 未找到 conda 环境 ${ISAACGYM_ENV}"
  echo "  请先运行: bash eval_suite/scripts/setup_isaacgym_conda.sh"
  exit 1
fi

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/activate_eval_env.sh"

find_isaacgym_python() {
  local base
  for base in \
    "${AUTODL_ROOT}/isaacgym/python" \
    "${REPO_ROOT}/isaacgym/python" \
    "${AUTODL_ROOT}/IsaacGym_Preview_4_Package/isaacgym/python" \
    "${AUTODL_ROOT}/isaacgym/isaacgym/python"; do
    if [[ -f "${base}/setup.py" ]]; then
      echo "${base}"
      return 0
    fi
  done
  return 1
}

CANDIDATES=(
  "${REPO_ROOT}/IsaacGym_Preview_4_Package.tar.gz"
  "${AUTODL_ROOT}/IsaacGym_Preview_4_Package.tar.gz"
  "${AUTODL_ROOT}/isaacgym/IsaacGym_Preview_4_Package.tar.gz"
  "/root/IsaacGym_Preview_4_Package.tar.gz"
  "${AUTODL_ROOT}/IsaacGym_Preview_4.tar.gz"
)

TAR_PATH=""
for tar in "${CANDIDATES[@]}"; do
  if [[ -f "$tar" ]]; then
    TAR_PATH="$tar"
    break
  fi
done

ISAACGYM_DIR=""
if ISAACGYM_DIR="$(find_isaacgym_python)"; then
  echo "[install_isaacgym] 已存在解压目录: ${ISAACGYM_DIR}"
elif [[ -n "${TAR_PATH}" ]]; then
  echo "[install_isaacgym] 找到压缩包: ${TAR_PATH}"
  echo "[install_isaacgym] 解压到 ${AUTODL_ROOT} ..."
  tar -xzf "${TAR_PATH}" -C "${AUTODL_ROOT}"
  ISAACGYM_DIR="$(find_isaacgym_python)" || true
else
  echo "[install_isaacgym] 未找到 Isaac Gym 压缩包。"
  echo "  请将 IsaacGym_Preview_4_Package.tar.gz 放到以下任一位置："
  echo "    ${REPO_ROOT}/"
  echo "    ${AUTODL_ROOT}/"
  exit 1
fi

if [[ -z "${ISAACGYM_DIR}" || ! -f "${ISAACGYM_DIR}/setup.py" ]]; then
  echo "[install_isaacgym] 解压后未找到 isaacgym/python/setup.py"
  echo "  请检查压缩包是否完整。"
  exit 1
fi

echo "[install_isaacgym] 安装: ${ISAACGYM_DIR}"
pip install -e "${ISAACGYM_DIR}"
python -c "import isaacgym; print('isaacgym 安装成功')"
echo "[install_isaacgym] 完成。"
