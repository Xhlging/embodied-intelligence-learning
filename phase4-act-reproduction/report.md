# Phase 4: ACT 复现报告

> 任务: 阅读 ACT 论文 + 了解 ALOHA 设备 + 跑通仿真环境  
> 论文: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (arXiv:2304.13705)](https://arxiv.org/abs/2304.13705)  
> 复现链路: LeRobot 0.6.1 (ACTPolicy) + gym-aloha (MuJoCo) + 官方演示数据集

## 实验环境

| 项目 | 配置 |
|------|------|
| 仿真环境 | **gym-aloha** 0.1.4 (HuggingFace) — `AlohaInsertion-v0` 双手插销任务 |
| 物理引擎 | MuJoCo 3.11.0 + dm_control 1.0.44 |
| 框架 | **LeRobot 0.6.1**（内置 ACTPolicy）+ PyTorch 2.11 + CUDA 13.0 |
| 硬件 | RTX 4060 8GB (WSL2)，训练 GPU 利用率 98% |
| 数据集 | `lerobot/aloha_sim_insertion_human` — 50 episodes, 25000 帧, 人类遥操作演示 (88MB) |
| Python | 3.13.12 |

## 复现步骤

```bash
# 1. 环境安装（关键版本组合，踩坑后锁定）
pip install --no-deps gym-aloha dm-control        # 跳过 labmaze（Python 3.13 无 wheel 且需 bazel）
pip install "mujoco>=3.11" gymnasium imageio imageio-ffmpeg dm-env dm-tree
pip install "lerobot[dataset,training]" "transformers>=5.4,<5.6" "huggingface-hub>=1.6,<2.0" opencv-python

# 2. 下载演示数据集
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download lerobot/aloha_sim_insertion_human \
  --repo-type dataset --local-dir data/aloha_sim_insertion_human

# 3. 训练（包装脚本解决 3 个兼容问题，见下）
python3 train_aloha.py \
  --env.type aloha --env.task AlohaInsertion-v0 \
  --policy.type act --policy.push_to_hub false \
  --dataset.repo_id lerobot/aloha_sim_insertion_human --dataset.root ./data \
  --output_dir ./outputs/act_aloha_insertion \
  --steps 10000 --batch_size 8 --num_workers 4 \
  --save_freq 2000 --env_eval_freq 10000 \
  --eval.use_async_envs false --eval.batch_size 2 --eval.n_episodes 10

# 4. 续训（从 checkpoint 恢复）
python3 train_aloha.py \
  --config_path ./outputs/act_aloha_insertion/checkpoints/last/pretrained_model/train_config.json \
  --resume true --steps 20000
```

**包装脚本 `train_aloha.py` 解决的兼容问题**：
1. gym-aloha 只在 `import gym_aloha` 时注册 gymnasium 环境——训练进程提前 import
2. lerobot eval 调用 `env.call("task_description")`——给 AlohaEnv monkey-patch 该属性
3. WSL2 无 X server——`--eval.use_async_envs false` 用 SyncVectorEnv 避免子进程渲染崩溃

## 实验结果

| 指标 | 结果 |
|------|------|
| 训练 | 20000 步（约 50 分钟 @ 6.9 step/s） |
| 评估 | 10 episodes @ step 20000 |
| **成功率** | **1/10 = 10%** |
| 评估视频 | `output/eval_episode_0~9.mp4`（**10 个完整**，独立评估生成） |
| Checkpoints | 10 个（2000 步间隔）+ last |

> 说明：训练过程中的评估（`--env_eval_freq`）只渲染前 4 个视频（LeRobot 内部硬编码 `max_episodes_rendered=4`，省磁盘）；完整的 10 个视频用独立评估生成：
>
> ```bash
> python3 eval_aloha.py \
>   --policy.path ./outputs/act_aloha_insertion/checkpoints/last/pretrained_model \
>   --env.type aloha --env.task AlohaInsertion-v0 \
>   --eval.n_episodes 10 --eval.use_async_envs false --eval.batch_size 1
> ```
>
> `eval_episode_5.mp4` 为成功案例（max_reward=4）。

### 训练曲线（两个阶段的对比）

| 训练步数 | 成功率 | 高奖励 episodes | 分析 |
|----------|:--:|:--:|------|
| 10000 | 1/10 = 10% | 6/10 (200~461) | 大部分 episode 接近成功（插销几乎到位） |
| 20000 | 1/10 = 10% | 4/10 (316~386) | **续训无明显提升**——模型已收敛 |

### 结果分析

1. **成功率偏低但符合预期**：LeRobot 官方参考（500-episode 统计）中 gym-aloha AlohaInsertion 的 ACT 成功率 ≈ 20%。10 个样本的 10% 在统计噪声范围内，且多个 episode 获得高奖励（接近成功但插销未完全插入）
2. **20000 步未见提升**：ACT 在 10000 步左右已收敛——续训收益低，说明该配置（batch 8 / lr 1e-5 / kl_weight 10 / chunk 100）已到模型能力上限
3. **任务难度**：双手插销（peg-in-hole）是接触密集的精细操作任务，对端到端模仿学习是公认的难点

### 瓶颈分析

- **训练阶段**：RTX 4060 算力满载（GPU 98%, 功耗 79.6W/82W），显存仅用 2.9/8GB → 瓶颈是 GPU 绝对算力，可通过 bf16 混合精度 ~2× 加速
- **评估阶段**：单环境串行渲染（GPU 3%），每 episode ~70 秒 → 瓶颈是 MuJoCo 渲染 + 单样本推理串行

## 遇到的问题与解决方案

| # | 问题 | 解决方案 |
|---|------|---------|
| 1 | `labmaze` 在 Python 3.13 无预编译 wheel，需 bazel 编译 | `pip install --no-deps` 跳过（主包导入不需要） |
| 2 | gym-aloha 0.1.4 要求 mujoco<3.9 与 dm_control 1.0.44 要求 mujoco≥3.11 冲突 | `--no-deps` 装 gym-aloha 后手动升级 mujoco 3.11.0 |
| 3 | lerobot 0.6.1 需要 hub≥1.6，transformers 4.x 需要 hub<1.0 | 锁定 `transformers>=5.4,<5.6` + `hub>=1.6,<2.0` |
| 4 | `--eval_freq` 参数不存在 | 用 `--env_eval_freq`（0.6.1 命名） |
| 5 | 训练要求 `--policy.push_to_hub`（默认 True 需 repo_id） | 显式 `--policy.push_to_hub false` |
| 6 | eval 报 `Namespace gym_aloha not found` | 包装脚本提前 `import gym_aloha` |
| 7 | eval 报 `task_description` 属性缺失 | monkey-patch `AlohaEnv.task_description` |
| 8 | WSL2 无 X server，AsyncVectorEnv 子进程崩溃 | `--eval.use_async_envs false` |
| 9 | 每次评估 35 分钟拖死训练 | `--env_eval_freq 10000`（只在最终评估）+ `--eval.n_episodes 10` |
| 10 | 续训报"目录已存在" | `--config_path <checkpoint>/train_config.json --resume true` 双参数 |

## 超参数观察（阶段目标）

| 超参数 | 作用 | 本实验观察 |
|--------|------|-----------|
| `chunk_size=100` | 一次预测 2 秒动作（50Hz×100 步） | 产生平滑的连续动作轨迹 |
| `kl_weight=10` | CVAE 风格变量的 KL 正则强度 | 训练稳定性的关键（官方推荐） |
| `--steps` | 训练总步数 | 10000→20000 无提升，说明已收敛 |
| `--eval.use_async_envs` | 评估并行方式 | 无头环境必须 false |

## 总结

1. **任务达成**：ACT 在 gym-aloha 仿真环境完整跑通——数据准备 → 训练 → 评估 → 视频输出全链路
2. **成功率 10%**（1/10）：与 LeRobot 官方参考（~20%）同量级，处于统计噪声范围；多个 episode 接近成功，说明模型学到了插销任务的运动模式但精度不足
3. **核心收获**：
   - ACT 输出**关节空间**动作（14 维），配合 Action Chunking 一次预测 100 步，是"高频控制 × 低频推理"矛盾的工程解法
   - CVAE 风格变量（推理时置零）是模仿学习处理演示多样性的关键设计
   - 仿真环境调试的最大成本在**环境兼容**（版本矩阵），而非算法本身
4. **可改进方向**：增加演示数据量（50→200 episodes）、调大 batch、bf16 混合精度提速、temporal ensembling 调参

## 产出文件

```
phase4-act-reproduction/
├── report.md                              # 本报告
├── notes/
│   ├── act-paper-reading.md               # ACT 论文精读笔记
│   └── aloha-guide.md                     # ALOHA 硬件/操作指南（斯坦福+松灵）
└── code/
    ├── train_aloha.py                     # 训练包装脚本（3 个兼容补丁）
    ├── data/aloha_sim_insertion_human/    # 演示数据集 (88MB)
    └── outputs/act_aloha_insertion/
        ├── checkpoints/                   # 10 个 checkpoint + last (4.7GB)
        └── eval/videos_step_020000/       # 10 个评估推理视频
```
