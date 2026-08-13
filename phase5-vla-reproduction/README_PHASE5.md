# Phase 5 操作手册（路线 A：SmolVLA 小规模 VLA × LIBERO 训练推理）

> 任务要求：复现**小规模 VLA 模型**，在论文未使用的仿真环境（LIBERO）完成**训练+推理**，产出视频
> 方案：**SmolVLA 0.5B**（LeRobot 官方小规模 VLA）+ **LIBERO**（robosuite 仿真）+ **LoRA 微调**
> 硬件：图书馆机器 WSL2 + RTX 5880 Ada 48GB（本地实测同版本环境可跑）
> 时间预算：3 小时

## 为什么是 SmolVLA

| 任务要求 | SmolVLA 满足 |
|---------|:--:|
| 小规模 VLA | ✅ 0.5B（单卡可训练） |
| 训练+推理 | ✅ LoRA 微调 + 评估 |
| 论文未用环境（LIBERO） | ✅ 论文（MoLe-VLA）只用 RLBench |
| 视频交付 | ✅ eval 自动录制 |

> MoLe-VLA 本体是 7B（论文架构），本路线以"小规模 VLA 的训练-仿真交互闭环"为目标，与小组任务字面要求对齐。

## ⏱️ 执行顺序

| 步骤 | 内容 | 耗时 |
|------|------|:--:|
| 0 | WSL2 GPU 验证 | 1 分钟 |
| 1 | clone + Miniconda | 10 分钟 |
| 2 | 环境：LeRobot 0.6.1 + extras | 20 分钟 |
| 3 | 下载数据集 | 10 分钟 |
| 4 | **测速 + 训练（LoRA）** | 1.5-2 小时 |
| 5 | 评估 + 视频 | 20 分钟 |
| 6 | 回传 GitHub | 5 分钟 |

## 第 0 步：WSL2 GPU 验证

```bash
nvidia-smi   # 应显示 RTX 5880 Ada
```

## 第 1 步：clone + Miniconda

```bash
cd ~
git clone https://github.com/Xhlging/embodied-intelligence-learning.git
cd embodied-intelligence-learning

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
```

## 第 2 步：环境（LeRobot + smolvla/libero/peft）

```bash
# 独立环境
conda create -n smolvla python=3.10 -y
conda activate smolvla

# ① GPU torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# ② LeRobot + SmolVLA/LIBERO/LoRA 支持（自动装 hf-libero）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "lerobot[smolvla,libero,peft]"

# ③ ffmpeg（训练必需！实测缺失时 lerobot-train 直接报错，不只是评估）
conda install -c conda-forge ffmpeg -y
```

✅ 验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # True
python -c "from lerobot.envs.configs import EnvConfig; EnvConfig.get_choice_class('libero'); print('libero env OK')"
python -c "from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig; print('smolvla OK')"
```

## 第 3 步：下载数据集

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download lerobot/libero_10 --repo-type dataset --local-dir data/libero_10
```

> `lerobot/libero_10`：379 episodes、10 个任务、panda（实测元数据）。
> 也可用更大的 `HuggingFaceVLA/libero`（1693 episodes，含多子集）。

✅ 验证：`ls data/libero_10/meta/info.json`

## 第 4 步：训练（LoRA 微调 SmolVLA）

```bash
# ① 先测速：跑 500 步，看 step/s，估算总时长
lerobot-train \
  --policy.path lerobot/smolvla_base \
  --env.type libero --env.task libero_10 \
  --dataset.repo_id lerobot/libero_10 --dataset.root ./data \
  --output_dir ./outputs/train/smolvla_libero \
  --steps 500 --batch_size 4 --num_workers 4 \
  --peft.method_type LORA --peft.r 64 --peft.lora_alpha 64 \
  --policy.optimizer_lr 1e-3 \
  --policy.push_to_hub false \
  --save_freq 1000

# ② 看输出里的 step/s，按剩余时间定 steps（例：2 step/s → 1.5 小时 ≈ 10000 步）
#    正式训练（改 steps 后重跑；output_dir 换新名或加 --resume true）
```

✅ 验证：`outputs/train/smolvla_libero/checkpoints/` 出现 checkpoint

## 第 5 步：评估 + 录制视频

```bash
lerobot-eval \
  --policy.path ./outputs/train/smolvla_libero/checkpoints/last/pretrained_model \
  --env.type libero --env.task libero_10 \
  --eval.n_episodes 10 \
  --eval.use_async_envs false \
  --eval.recording true
```

✅ 验证：`outputs/eval/.../videos/` 出现 mp4

## 第 6 步：回传

```bash
cd ~/embodied-intelligence-learning
git add phase5-vla-reproduction/output/
git commit -m "Phase 5: SmolVLA × LIBERO 推理视频"
git push
```

## 常见问题

| 问题 | 解决 |
|------|------|
| `import hf_libero` 或 libero 报错 | 确认装了 `lerobot[libero]`（含 hf-libero 0.1.4） |
| 训练 OOM | batch_size 4 → 2；或 `--peft.r 32` |
| 评估时环境创建失败 | WSL2 需 `sudo apt install -y libosmesa6-dev libegl1 libgl1`（本地实证：缺库渲染崩） |
| 首次运行 LIBERO 交互提问 | 输 n（默认路径）；或预建 `~/.libero/config.yaml` |
| 训练太慢 | 减小 steps；用 libero_10 单任务；`--num_workers 8` |
| 成功率低 | 正常（短训），任务要求是跑通训练-推理链路 + 视频 |

## 备用路线（CogACT 7B 推理，已实证 API）

`code/lib_libero_eval.py` 保留（CogACT 7B 零样本推理，权重 30GB 需下载）。仅当 SmolVLA 训练失败时启用。
