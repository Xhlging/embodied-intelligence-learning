# Phase 5 操作手册（WSL2 + RTX 5880 Ada 48GB）

> 目标：CogACT（MoLe-VLA backbone）在 LIBERO 仿真上推理 + 产出视频
> 总耗时预算：3 小时（权重下载 30GB 是最大项，**到馆先启动**）

## ⏱️ 执行顺序总览

| 步骤 | 内容 | 耗时 | 验证 |
|------|------|:--:|------|
| 0 | WSL2 GPU 验证 | 1 分钟 | `nvidia-smi` |
| 1 | 拉代码 + Miniconda | 10 分钟 | `conda --version` |
| 2 | 系统依赖（OSMesa 渲染） | 5 分钟 | `dpkg -l libosmesa6` |
| 3 | 官方环境 + 依赖 | 20 分钟 | `from vla import load_vla` |
| 4 | LIBERO 安装 | 10 分钟 | `from libero.libero import benchmark` |
| **5** | **权重下载 30GB（后台）** | **40-60 分钟** | `ls ~/models/CogACT-Base/checkpoints/` |
| 6 | 推理 + 录视频 | 15 分钟 | `output/` 出现 mp4 |
| 7 | 回传 GitHub | 5 分钟 | git push |

## 第 0 步：WSL2 GPU 验证

```bash
nvidia-smi
# 应显示 RTX 5880 Ada
```

## 第 1 步：拉代码 + Miniconda

```bash
cd ~
git clone https://github.com/Xhlging/embodied-intelligence-learning.git
cd embodied-intelligence-learning

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda --version   # ✅ 验证
```

## 第 2 步：系统依赖（LIBERO 渲染必需，本地实证过）

```bash
sudo apt update
sudo apt install -y libosmesa6-dev libegl1 libgl1 libgl1-mesa-dev libglib2.0-0
```

> LIBERO 的 OffScreenRenderEnv 用 mujoco 离屏渲染，**EGL 和 OSMesa 库都必需**（缺 libegl1 会报 `EGLError`）。本地实证确认：缺库时环境创建/reset 正常，但渲染崩溃。

## 第 3 步：官方环境 + 依赖

```bash
cd phase5-vla-reproduction/code/MoLe-VLA-Pytorch

# ① prismatic 复制（代码 import prismatic.*，官方结构）
cp -r prismatic_new prismatic

# ② 创建环境
conda create -n cogact python=3.10 -y
conda activate cogact

# ③ 官方依赖清单（Linux 完整版，含 tensorflow/tfds）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r ../../requirements_linux.txt

# ④ dlimp（不在 PyPI）
pip install "git+https://ghfast.top/https://github.com/kvablack/dlimp.git"

# ⑤ torch GPU 版（官方 pin 2.5.1 + cu121）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

✅ 验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.5.1+cu121 True

python -c "from vla import load_vla; print('VLA import OK')"
```

## 第 4 步：LIBERO 安装

```bash
cd ~/embodied-intelligence-learning/phase5-vla-reproduction/code
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

✅ 验证：

```bash
python -c "from libero.libero import benchmark; from libero.libero.envs import OffScreenRenderEnv, RobosuiteEnv; print('LIBERO OK')"
# 首次运行会问数据集路径 → 输入 n（用默认）
```

## 第 5 步：权重下载（⚠️ 到馆第一件事，后台跑）

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download CogACT/CogACT-Base --local-dir ~/models/CogACT-Base
```

✅ 验证（目录结构必须是这样）：

```bash
ls ~/models/CogACT-Base/
# 必须包含: config.json  dataset_statistics.json  checkpoints/
ls ~/models/CogACT-Base/checkpoints/
# 必须包含: CogACT-Base.pt   ← 推理脚本 --ckpt 指向这个文件
```

> ⚠️ **`--ckpt` 传的是 .pt 文件路径**（不是目录）——`load_vla` 要求父目录叫 `checkpoints`，同级有 `config.json` 和 `dataset_statistics.json`

## 第 6 步：推理 + 录视频

```bash
cd ~/embodied-intelligence-learning/phase5-vla-reproduction/code
python lib_libero_eval.py --ckpt ~/models/CogACT-Base/checkpoints/CogACT-Base.pt \
  --action-model DiT-B --benchmark libero_spatial --task-idx 0 --num-episodes 5
```

✅ 验证：`phase5-vla-reproduction/output/` 出现 `libero_spatial_task0_ep*.mp4`

### 动作尺度调参（机器人不动/乱动时）

| 现象 | 调整 |
|------|------|
| 完全不动（动作太小） | `--action-scale 1.0` 或 `2.0` |
| 乱飞/抖动（动作太大） | `--action-scale 0.2` 或 `0.1` |
| 换策略配置 | `--policy-setup widowx_bridge` |

> 零样本成功率预期 10-30%（CogACT 没训过 LIBERO，动作尺度/频率不匹配）——**视频能出即跑通**，报告如实说明

## 第 7 步：回传结果

```bash
cd ~/embodied-intelligence-learning
git add phase5-vla-reproduction/output/
git commit -m "Phase 5: LIBERO 推理视频"
git push
```

## 常见问题

| 问题 | 解决 |
|------|------|
| `import vla` 报错 | 确认第 3 步 ①② 执行（prismatic 复制 + 官方依赖） |
| LIBERO 渲染黑屏/失败 | 第 2 步 OSMesa 未装；或换 `RobosuiteEnv`（脚本自动 fallback） |
| 显存不足 | 48GB 足够 7B bf16；仍 OOM 换 `--action-model DiT-S` |
| 权重下载慢 | 已用 hf-mirror；再慢用 `aria2c -x16` 多线程 |
| 推理很慢（>5s/步） | 正常（7B VLA），每 episode 约 2 分钟 |
| 报 `Missing config.json` | `--ckpt` 必须指向 `checkpoints/xxx.pt`，且 config.json 在同级 |
