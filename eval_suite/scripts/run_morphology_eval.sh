#!/usr/bin/env bash
# 形态学评估一键脚本。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
ISAACGYM_ENV="${AUTODL_ROOT}/conda_envs/isaacgym"

if [[ ! -d "${ISAACGYM_ENV}/bin" ]]; then
  echo "=== [0/4] 首次运行：创建 isaacgym conda 环境 ==="
  bash "${REPO_ROOT}/eval_suite/scripts/setup_isaacgym_conda.sh"
fi

# shellcheck source=/dev/null
source "${REPO_ROOT}/eval_suite/scripts/activate_eval_env.sh"

SYMM_POOL="${SYMM_POOL:-eval_suite/assets/symmetric_v64}"
ASYM_POOL="${ASYM_POOL:-eval_suite/assets/full_asym_v64}"
OUT_DIR="${OUT_DIR:-eval_suite/results/$(date +%Y%m%d_%H%M%S)}"
LOGDIR_LABEL="${LOGDIR_LABEL:-gait-conditioned-agility/pretrain-v0/train}"

echo "=== [1/4] 检查运行前提 ==="
if ! python -c "import isaacgym" 2>/dev/null; then
  echo "未检测到 isaacgym，尝试安装..."
  bash "${REPO_ROOT}/eval_suite/scripts/install_isaacgym_hint.sh"
fi
python -m eval_suite.scripts.check_prerequisites

echo "=== [2/4] 生成变体池（已有 manifest 则跳过）==="
if [[ ! -f "$SYMM_POOL/manifest.json" ]]; then
  python -m eval_suite.morphology.build_variant_pool \
    --mode symmetric --num_variants 64 --out_dir "$SYMM_POOL" \
    --seed 42 --delta_scale 0.5 --sort_manifest
fi
if [[ ! -f "$ASYM_POOL/manifest.json" ]]; then
  python -m eval_suite.morphology.build_variant_pool \
    --mode full_asym --num_variants 64 --out_dir "$ASYM_POOL" \
    --seed 42 --delta_scale 0.5 --sort_manifest
fi

echo "=== [3/4] 离线 smoke test ==="
python -m eval_suite.scripts.smoke_test_offline

echo "=== [4/4] 运行三种工况评估 ==="
python -m eval_suite.runners.run_all \
  --logdir "$LOGDIR_LABEL" \
  --sym_pool "$SYMM_POOL" \
  --asym_pool "$ASYM_POOL" \
  --out_dir "$OUT_DIR" \
  --headless

echo "完成。结果目录: $OUT_DIR"
