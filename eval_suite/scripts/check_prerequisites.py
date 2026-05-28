"""仿真评估运行前的环境检查。"""

from __future__ import annotations

import importlib
from pathlib import Path


def require_isaac_gym() -> None:
    try:
        importlib.import_module("isaacgym")
    except ImportError as exc:
        raise SystemExit(
            "未安装 Isaac Gym。请先完成 conda 环境配置：\n"
            "  bash eval_suite/scripts/setup_isaacgym_conda.sh\n"
            "  source eval_suite/scripts/activate_eval_env.sh\n"
            "  bash eval_suite/scripts/install_isaacgym_hint.sh\n"
            "详见 eval_suite/docs/ENV_SETUP.md"
        ) from exc


def require_go1_gym(repo_root: Path | None = None) -> None:
    try:
        importlib.import_module("go1_gym")
    except ImportError as exc:
        root = repo_root or Path(__file__).resolve().parents[2]
        raise SystemExit(
            f"无法导入 go1_gym。请在 {root} 下运行: pip install -e . --no-deps"
        ) from exc


def check_all(repo_root: Path | None = None) -> None:
    require_isaac_gym()
    require_go1_gym(repo_root)


if __name__ == "__main__":
    check_all()
    print("环境检查通过")
