#!/usr/bin/env python3
"""Batch-generate Go1 variant URDF pools and manifest.json for morphology eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval_suite.morphology.go1_variant_urdf import write_variant_urdf
from eval_suite.morphology.quadruped_variant_sampling import (
    sample_full_asymmetric_variant,
    sample_quadruped_variant,
)
from eval_suite.morphology.variant_manifest import sort_manifest_entries_by_severity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["symmetric", "full_asym"], required=True)
    parser.add_argument("--num_variants", type=int, default=64)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta_scale", type=float, default=0.5)
    parser.add_argument("--sort_manifest", action="store_true")
    args = parser.parse_args()

    import numpy as np

    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir.resolve()
    urdf_dir = out_dir / "urdf"
    urdf_dir.mkdir(parents=True, exist_ok=True)

    sampler = sample_quadruped_variant if args.mode == "symmetric" else sample_full_asymmetric_variant
    manifest: list[dict] = []

    for i in range(args.num_variants):
        sample = sampler(rng, delta_scale=args.delta_scale)
        rel = f"urdf/variant_{i:03d}.urdf"
        urdf_path = out_dir / rel
        write_variant_urdf(urdf_path, sample)
        manifest.append(sample.to_manifest_dict(variant_index=i, urdf_relpath=rel, usd_relpath=None))
        print(f"[build_variant_pool] wrote {urdf_path}")

    if args.sort_manifest:
        manifest = sort_manifest_entries_by_severity(manifest)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[build_variant_pool] manifest -> {manifest_path} ({len(manifest)} variants)")


if __name__ == "__main__":
    main()
