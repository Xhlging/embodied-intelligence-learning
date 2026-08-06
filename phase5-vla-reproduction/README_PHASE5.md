# Phase 5 操作手册（WSL2 + RTX 5880 Ada 48GB）

> 目标：MoLe-VLA / CogACT 在 LIBERO 仿真上完成推理，产出视频
> 环境：图书馆机器 WSL2（Linux）+ 官方 environment.yml 完整依赖
> 时间预算：3 小时

## 第 0 步：确认 WSL2 GPU（1 分钟）

```bash
nvidia-smi
# 应显示 RTX 5880 Ada（WSL2 GPU 直通）
```

## 第 1 步：拉代码 + 装 Miniconda（10 分钟）

```bash
cd ~
git clone https://github.com/Xhlging/embodied-intelligence-learning.git
cd embodied-intelligence-learning

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
```

## 第 2 步：创建官方环境（15 分钟）

```bash
cd phase5-vla-reproduction/code/MoLe-VLA-Pytorch

# ① prismatic 复制（官方结构，代码 import prismatic.*）
cp -r prismatic_new prismatic

# ② 创建 python 3.10 环境
conda create -n cogact python=3.10 -y
conda activate cogact

# ③ 官方依赖清单（Linux 完整版）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r ../../requirements_linux.txt

# ④ dlimp（不在 PyPI，GitHub 安装）
pip install "git+https://ghfast.top/https://github.com/kvablack/dlimp.git"

# ⑤ torch GPU 版（官方 pin 2.5.1 + cu121）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## 第 3 步：验证（2 分钟）

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.5.1+cu121 True

python -c "from vla import load_vla; print('VLA import OK')"
# 期望: 无报错
```

## 第 4 步：下载权重（30-60 分钟，后台跑）

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download CogACT/CogACT-Base --local-dir ~/models/CogACT-Base
# 验证: ls ~/models/CogACT-Base/checkpoints/
```

## 第 5 步：生成 LIBERO 演示数据（可选）

```bash
cd ~/embodied-intelligence-learning/phase5-vla-reproduction/code/LIBERO  # 若已克隆
python scripts/collect_demonstrations.py --task-name libero_spatial --num-episodes 10
```

## 第 6 步：推理 + 录视频（30 分钟）

```bash
cd ~/embodied-intelligence-learning/phase5-vla-reproduction/code
python lib_libero_eval.py --ckpt ~/models/CogACT-Base --action-model DiT-B \
  --benchmark libero_spatial --task-idx 0 --num-episodes 5
```

✅ 验证：`output/` 目录出现 `libero_spatial_task0_ep*.mp4`

### 动作尺度调参（机器人不动/乱动时）

| 现象 | 调整 |
|------|------|
| 完全不动 | `--action-scale 1.0` 或 `2.0` |
| 乱飞/抖动 | `--action-scale 0.2` 或 `0.1` |
| 换策略配置 | `--policy-setup widowx_bridge` |

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
| `import vla` 报错 | 确认执行了第 2 步 ①②（prismatic 复制 + 依赖） |
| 显存 OOM | 48GB 足够 7B bf16，若仍 OOM 换 `--action-model DiT-S` |
| LIBERO 安装 | `cd code/LIBERO && pip install -e .`（需先 git clone LIBERO） |
| 下载慢 | HF_ENDPOINT 已设 hf-mirror；仍慢可用 `aria2c -x16` 多线程 |
