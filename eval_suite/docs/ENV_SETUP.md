# 环境配置指南

walk-these-ways 基于 **Isaac Gym Preview 4**（旧版 GPU 仿真），与 **Isaac Lab / Isaac Sim**（本机 `isaaclab` conda）是两套独立栈，**不建议复用 `isaaclab` 环境**。

## 为什么要单独建环境？

| 项目 | `isaaclab` 环境 | walk-these-ways 需要 |
|------|-----------------|----------------------|
| 仿真后端 | Isaac Sim 4.5 / Isaac Lab | Isaac Gym Preview 4 |
| Python | 3.10 | **3.8**（Preview 4 官方推荐） |
| PyTorch | 较新版本 | **1.10 + cu113** |
| numpy | 1.26.x | **1.23.5**（`setup.py` 锁定） |
| 混装风险 | 在 isaaclab 里装 isaacgym 易冲突、难排查 | 独立 env 一次配好 |

结论：**请为 Isaac Gym 新建 `conda_envs/isaacgym`，与 `conda_envs/isaaclab` 并存。**

## 一键安装（推荐）

压缩包上传完成后，在仓库根目录执行：

```bash
cd /root/autodl-tmp/walk-these-ways_eval

# 1) 创建独立 conda 环境 + PyTorch + go1_gym（首次约 2–5 分钟）
bash eval_suite/scripts/setup_isaacgym_conda.sh

# 2) 激活环境
source eval_suite/scripts/activate_eval_env.sh

# 3) 解压并安装 Isaac Gym（tar 放在 /root/autodl-tmp/）
bash eval_suite/scripts/install_isaacgym_hint.sh

# 4) 验证
python scripts/test.py
python -m eval_suite.scripts.smoke_test_offline
python -m eval_suite.scripts.check_prerequisites
```

> **顺序说明：** 必须先建 conda 环境（Python 3.8），再在**该环境内** `pip install -e isaacgym/python`。Isaac Gym 官方支持 Python 3.6–3.8，本仓库选用 3.8 + torch 1.10。

## 压缩包放置位置

脚本会按顺序查找：

- `walk-these-ways_eval/IsaacGym_Preview_4_Package.tar.gz`（仓库根目录，推荐）
- `/root/autodl-tmp/IsaacGym_Preview_4_Package.tar.gz`
- `/root/autodl-tmp/isaacgym/IsaacGym_Preview_4_Package.tar.gz`
- `/root/IsaacGym_Preview_4_Package.tar.gz`

解压后应存在：`/root/autodl-tmp/isaacgym/python/setup.py`。

## 环境路径

| 用途 | 路径 | 激活方式 |
|------|------|----------|
| Isaac Lab 训练/仿真 | `/root/autodl-tmp/conda_envs/isaaclab` | `source /root/autodl-tmp/activate_isaaclab.sh` |
| walk-these-ways 评估 | `/root/autodl-tmp/conda_envs/isaacgym` | `source eval_suite/scripts/activate_eval_env.sh` |

## 运行形态学评估

```bash
source eval_suite/scripts/activate_eval_env.sh
bash eval_suite/scripts/run_morphology_eval.sh
```

## 常见问题

**Q: 能否在 isaaclab 环境里 `pip install isaacgym`？**  
A: 不推荐。Python 版本、numpy、PyTorch 均不匹配，之前已出现 `numpy==1.23.5` 冲突。

**Q: 运行时报 `libpython3.8.so.1.0: cannot open shared object file`？**  
A: 已通过 `activate_eval_env.sh` 自动设置 `LD_LIBRARY_PATH`。请确保先 `source eval_suite/scripts/activate_eval_env.sh` 再运行 Python。

**Q: RTX 4090 上报 `nvrtc: error: invalid value for --gpu-architecture`？**  
A: torch 1.10 不支持 sm_89。在已激活环境中运行：
```bash
bash eval_suite/scripts/fix_torch_4090.sh
```

**Q: 两个环境会抢 GPU 吗？**  
A: 不会同时占用；评估前 `conda activate isaacgym`，做 Isaac Lab 实验时再切回 `isaaclab` 即可。

**Q: checkpoint 在哪？**  
A: `runs/gait-conditioned-agility/pretrain-v0/train/<timestamp>/checkpoints/`（需含 `body_latest.jit`、`adaptation_module_latest.jit`）。

更多评估协议见 [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md)。
