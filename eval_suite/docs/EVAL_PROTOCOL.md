# 形态学评估协议

分支：`eval/morphology-suite`  
代码目录：`eval_suite/`（与 `go1_gym/` 训练代码隔离）

> 环境配置请先阅读 **[ENV_SETUP.md](ENV_SETUP.md)**（独立 `isaacgym` conda，勿复用 `isaaclab`）。

## 依赖摘要

- 独立 conda：`/root/autodl-tmp/conda_envs/isaacgym`（Python 3.8 + torch 1.10 + Isaac Gym Preview 4）
- 预训练 checkpoint：`runs/gait-conditioned-agility/pretrain-v0/train/<timestamp>/`
  - `checkpoints/body_latest.jit`
  - `checkpoints/adaptation_module_latest.jit`
  - `parameters.pkl`

## 三种测试工况

| 工况 ID | 说明 | URDF 来源 | 变体数 |
|---------|------|-----------|--------|
| `baseline` | 标准 Go1 URDF | `resources/robots/go1/urdf/go1.urdf` | 1 |
| `symmetric` | 等比例随机化缩放 | manifest 池前 N 个（默认 N=16） | 16 |
| `full_asym` | 四腿独立非对称缩放 | manifest 池前 N 个（默认 N=16） | 16 |

> 变体池可预生成 64 个 URDF，评估时通过 `eval_defaults.yaml` 的 `num_variants: 16` 只取前 16 个。全量 64 时改为 `num_variants: 64`。

全局几何缩放 `[0.8, 1.2]`，质量按 `scale³`。评估时关闭域随机化，仅改 URDF 几何。

## 四项指标

### 1. 最大极限速度 `v_max`（鲁棒性）

默认对 `vx ∈ [0, 4.0] m/s` **二分搜索，粒度 0.2 m/s**（约 4–5 次 trial）。成功条件：无提前终止 + 稳态段内 `mean(|v_meas_x - v_cmd_x|) < 0.3 m/s`。

可选 `--speed_search_coarse_only` 改为线性扫描（更慢，仅调试）。

### 2. 偏航误差（准确程度 — 直线行走）

命令：`vx=1.5 m/s, vy=0, wz=0`（与稳定性 episode 相同，走直线）。

从机体四元数提取 yaw 角，在稳态段（去 warmup 15 步后的 steady 段）计算：

- **yaw_offset_mean**：相对 warmup 末时刻航向的偏移，\(\text{mean}(|\Delta\text{yaw}|)\)（rad）
- **yaw_variance**：稳态段 yaw 角方差 \(\text{Var}(\text{yaw})\)（rad²）

时序保存在 `straight_line.npz`（含 `yaw`, `yaw_drift`）。

### 3. Base 方差（稳定性）

`vx=1.5` 直线行走，稳态段（后 3 s）的 `var_vx/vy/roll/pitch` 等权均值 → `base_variance_scalar`。

### 4. 输出功率（能量效率）

同上 episode，`P(t)=Σ|τ·q̇|` 稳态均值 → `mean_power`；辅助指标 `cot = P/(mgv)`。

## 常用命令

```bash
# 环境（首次）
bash eval_suite/scripts/setup_isaacgym_conda.sh
source eval_suite/scripts/activate_eval_env.sh
bash eval_suite/scripts/install_isaacgym_hint.sh

# 生成变体池（若尚未生成）
python -m eval_suite.morphology.build_variant_pool --mode symmetric --num_variants 64 \
  --out_dir eval_suite/assets/symmetric_v64 --seed 42 --delta_scale 0.5 --sort_manifest
python -m eval_suite.morphology.build_variant_pool --mode full_asym --num_variants 64 \
  --out_dir eval_suite/assets/full_asym_v64 --seed 42 --delta_scale 0.5 --sort_manifest

# 离线检查（无需 Isaac Gym）
python -m eval_suite.scripts.smoke_test_offline

# 单工况 / 全量
python -m eval_suite.runners.run_condition --condition baseline --out_dir eval_suite/results/test_baseline
bash eval_suite/scripts/run_morphology_eval.sh

# 快速模式（粗粒度速度扫描）
python -m eval_suite.runners.run_all --speed_search_coarse_only --headless
```

## 输出结构

```
eval_suite/results/<timestamp>/
  baseline/   symmetric/   full_asym/
    manifest.json  config.yaml  summary.csv  summary.json  REPORT.md
    episodes/variant_XXX/
      speed_limit.json  yaw_tracking.npz  stability.npz  power.npz
  summary_all.csv  summary_all.json  COMBINED_REPORT.md
```

### summary.csv 主要字段

| 列名 | 含义 |
|------|------|
| `condition` | baseline / symmetric / full_asym |
| `variant_index` | 变体编号 |
| `v_max` | 最大成功命令速度 (m/s) |
| `yaw_offset_mean` | 直线行走 yaw 偏移均值 (rad) |
| `yaw_variance` | 直线行走 yaw 角方差 (rad²) |
| `base_variance_scalar` | 综合 base 方差 |
| `mean_power` | 平均功率 (W) |

### NPZ / JSON

- **straight_line.npz / stability.npz**：速度、姿态、yaw、力矩、功率时序
- **speed_limit.json**：`[{vx, success, terminated_early}, ...]`

## 日志示例

```
[baseline][variant_000] v_max=2.10 yaw_off=0.0521 yaw_var=0.001234 base_var=0.001234 power=45.67
```

## 预计耗时

默认 **快速配置**（0.2 m/s 速度粒度、60 步 episode、变体内复用 env）：

| 操作 | 耗时 |
|------|------|
| 离线 smoke test | <1 秒 |
| baseline 单变体 | ~15–20 秒 |
| **全量 33 变体**（1+16+16，默认） | **~10–15 分钟** |
| 全量 129 变体（num_variants=64） | ~40–60 分钟 |

主要时间花在每个 URDF 变体重建 Isaac Gym 环境（~5–8 秒/次）。变体内复用 env，速度搜索不再重复 load。

若需更细速度分辨率，在 `eval_defaults.yaml` 中改 `speed_precision`（如 0.5）并加长 `speed_episode_steps`。

## 实现说明

- 每个 URDF 变体建 1 次 Isaac Gym env，变体内速度/偏航/稳定性指标复用同一 env
- 形态采样移植自 `gen_loco_isaac_lab`，URDF 已适配 Go1 关节名（`*_thigh_joint`, `*_calf_joint`, `*_foot`）
