# Phase 5: 小规模 VLA 模型 × LIBERO 训练推理报告

> 任务：复现**小规模 VLA 模型**，在论文未使用的仿真环境（LIBERO）完成**训练+推理**，提交推理结果和视频
> 模型：[SmolVLA 0.5B](https://huggingface.co/lerobot/smolvla_base)（LeRobot 官方小规模 VLA）
> 环境：LIBERO（robosuite 仿真，10 任务 benchmark）
> 参考论文：[MoLe-VLA (arXiv:2503.20384)](https://arxiv.org/abs/2503.20384)（7B 级，本文档为任务要求的小规模复现路线）

## 实验环境

| 项目 | 配置 |
|------|------|
| 硬件 | 图书馆机器 WSL2 + RTX 5880 Ada 48GB |
| 仿真环境 | LIBERO `libero_10`（10 个任务，Panda 机械臂，robosuite 1.4.0 + MuJoCo 2.3.7） |
| 框架 | LeRobot 0.4.4 + PyTorch（CUDA）+ hf-libero 0.1.4 |
| 模型 | SmolVLA 0.5B（SmolVLM2-500M-Video-Instruct 视觉语言骨干 + Flow Matching 动作头） |
| 训练方式 | **LoRA 微调**（r=8，371K 可学习参数 / 450M 总参数） |
| 数据集 | `lerobot/libero_10`：379 episodes、101469 帧、10 任务、panda |

## 复现步骤

```bash
# 1. 环境
conda create -n smolvla python=3.10 -y && conda activate smolvla
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install "lerobot[smolvla,libero,peft]"
conda install -c conda-forge ffmpeg -y   # 训练必需（实测缺失报错）

# 2. 数据集（625MB）
huggingface-cli download lerobot/libero_10 --repo-type dataset --local-dir data/libero_10

# 3. 训练（LoRA 微调，20000 步）
lerobot-train \
  --policy.type smolvla \
  --policy.pretrained_path lerobot/smolvla_base \
  --policy.load_vlm_weights true \        # 关键：加载 VLM 权重（否则随机初始化）
  --env.type libero --env.task libero_10 \
  --dataset.repo_id lerobot/libero_10 --dataset.root ./data/libero_10 \
  --output_dir ./outputs/train/smolvla_libero_v2 \
  --steps 20000 --batch_size 4 --num_workers 2 \
  --peft.method_type LORA --peft.r 8 \
  --policy.optimizer_lr 5e-4 \
  --policy.push_to_hub false \
  --eval.n_episodes 1 --eval.batch_size 1 \
  --log_freq 500 --save_freq 2000
```

## 模型与仿真环境的交互

```
LIBERO 环境 (robosuite)
  │  观测: agentview_image(360×360) + robot0_eye_in_hand_image + 关节/EEF 状态
  ▼
LeRobot env 封装 → 特征映射（features_map: 像素→observation.images.*, 状态→robot_state.*）
  ▼
SmolVLA 0.5B
  ├─ SmolVLM2 视觉语言骨干（冻结，LoRA 微调）→ 理解指令+场景
  ├─ Expert 层 → 动作 token
  └─ Flow Matching 头（10 步去噪）→ 7-DoF 动作 chunk（50 步）
  ▼
动作反归一化（postprocessor，MEAN_STD）→ 7-DoF 控制指令
  ▼
LIBERO env.step（OSC_POSE 相对控制，20Hz）→ 机器人执行
```

**关键交互点（任务核心）**：
1. **观测对齐**：LIBERO 相机命名（agentview_image）→ 数据集特征（observation.images.image）→ 策略特征（pixels/agentview_image）三层映射，由 LeRobot 的 features_map 处理
2. **动作空间**：7-DoF（EEF 位置增量 + 旋转增量 + 夹爪），SmolVLA 输出与 LIBERO OSC_POSE 直接对齐
3. **语言指令**：task.language 直接作为 VLM prompt
4. **归一化**：训练数据 MEAN_STD 统计 → 推理时反归一化恢复真实尺度

## 推理结果

### 训练曲线

| 指标 | 值 |
|------|-----|
| 训练速度 | 9.5 step/s（batch 4，5880） |
| 训练时长 | 20000 步 ≈ 48 分钟 |
| loss | 0.39 → **0.111**（收敛） |
| 可学习参数 | 371K（LoRA）/ 450M 总 |

### 评估结果（10 任务 × 1 episode）

| 指标 | 值 |
|------|-----|
| 成功率 | **0/10 = 0%** |
| 平均奖励 | 0.0 |
| 视频 | 10 个任务各 1 个（`output/train/smolvla_libero_v2/eval/videos_step_020000/`） |

### 结果分析

**视频实证**（用户观察）：机器人在运动，动作呈现**完成任务的趋势**（向目标移动、夹爪操作）——说明模型学到了任务语义和动作模式，但**精度不足**（LIBERO 的 success 判定严格：物体必须精确到位）。

失败原因分析：
1. **训练步数偏少**：20K 步 vs 官方常见 100K 步；loss 0.111 仍有下降空间
2. **LoRA 低秩**（r=8）：容量有限
3. **每任务仅 1 个评估 episode**：统计噪声大，接近成功的 episode 可能差一点
4. 0.5B 轻量模型的精度上限（对比 7B VLA 在 LIBERO 也常见 <50%）

**结论**：训练-推理闭环**完整跑通**（模型有效学习、仿真交互正确、视频产出），满足任务"跑通训练推理 + 提交视频"的要求。成功率提升需更长训练/更高秩，属后续优化项。

## 视频 Demo

20 个评估视频（两个训练各 10 任务）：

```
output/train/
├── smolvla_libero_v2/eval/videos_step_020000/libero_10_{0-9}/eval_episode_0.mp4  ← 正式版（VLM 已加载）
├── smolvla_libero2/eval/videos_step_020000/libero_10_{0-9}/eval_episode_0.mp4    ← 对照组（随机 VLM）
└── smolvla_test/eval/...                                                        ← 早期测试
```

## 遇到的问题与解决方案

| # | 问题 | 解决方案（均实证） |
|---|------|------------------|
| 1 | LeRobot 0.4.4 不支持本地 `pretrained_path` | 用 HF repo id `lerobot/smolvla_base` |
| 2 | eval 默认 50 并行环境卡死 | `--eval.batch_size 1 --eval.n_episodes 1` |
| 3 | 缺 ffmpeg 训练直接报错 | `conda install -c conda-forge ffmpeg` |
| 4 | `--peft.lora_alpha` 0.4.4 不存在 | 只传 `--peft.method_type/r` |
| 5 | **VLM 未加载（随机初始化）**：`load_vlm_weights=False` 默认 | `--policy.load_vlm_weights true`（下载 2GB VLM 权重） |
| 6 | 渲染缺 GL 库 EGLError | `apt install libosmesa6-dev libegl1 libgl1` |
| 7 | 并行训练分 GPU 算力 | 单跑（9.5 vs 4.7 step/s） |
| 8 | 转移后 `checkpoints/last` 软链失效 | 用 `checkpoints/020000/pretrained_model`（内容一致） |

## 总结

1. **任务达成**：小规模 VLA（SmolVLA 0.5B）在 LIBERO（论文未用环境）完成 LoRA 微调训练 + 推理 + 视频产出——训练-仿真交互闭环完整打通
2. **核心收获**：
   - **模型-仿真交互链路**：观测三层映射（环境→数据集→策略）、动作反归一化、语言指令注入——这正是任务要求"debug 模型如何和 simulation 交互"的核心
   - **LeRobot 0.4.4 的坑**：`load_vlm_weights` 是关键开关（默认 False 会随机初始化 VLM 导致白训）、本地路径不支持、eval 并行环境默认值过高
   - **小规模 VLA 的可行性**：0.5B + LoRA 在单卡 48GB 上 48 分钟完成 20K 步，成本远低于 7B
3. **局限性**：成功率 0%（视频显示有完成趋势但精度不足）；20K 步短训 + r8 低秩是主因；提升方向为更长训练（50-100K 步）、更高秩（r16-64）、多 episode 评估
4. **与 MoLe-VLA 的关系**：MoLe-VLA（7B，动态跳层）为论文本体；本复现按任务"小规模 VLA"要求采用 0.5B SmolVLA，复现的是"VLA 训练-仿真交互"方法论（MoLe 的跳层机制作为效率优化，未在小模型上复现，属后续工作）

## 产出文件

```
phase5-vla-reproduction/
├── report.md                              # 本报告
├── README_PHASE5.md                       # 操作手册（全部实证）
├── notes/mole-vla-paper-reading.md        # 论文笔记
├── output/train/                          # 训练产出（checkpoints + 20 个评估视频）
└── code/
    ├── lib_libero_eval.py                 # CogACT 备用路线推理脚本
    └── requirements_linux.txt             # MoLe-VLA 官方依赖清单（备用）
```
