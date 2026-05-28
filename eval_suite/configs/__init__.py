"""Load EvalConfig from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from eval_suite.configs.eval_config import EvalConfig


def load_eval_config(path: Path | None = None) -> EvalConfig:
    if path is None:
        path = Path(__file__).resolve().parent / "eval_defaults.yaml"
    if not path.is_file():
        return EvalConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = EvalConfig()
    for key, value in data.items():
        if hasattr(cfg, key):
            if key.endswith("_pool") or key.endswith("_urdf"):
                setattr(cfg, key, Path(value))
            else:
                setattr(cfg, key, value)
    return cfg
