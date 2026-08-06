# Phase 5 图书馆机器操作手册（Windows + RTX 5880 Ada 48GB）

> 目标：MoLe-VLA / CogACT 在 LIBERO 仿真上完成推理，产出视频
> 时间预算：3 小时。按顺序执行，每步有验证点。

## 第 0 步：拉取代码（5 分钟）

```powershell
git clone https://github.com/Xhlging/embodied-intelligence-learning.git
cd embodied-intelligence-learning
```

✅ 验证：`ls phase5-vla-reproduction/code/` 能看到 `lib_libero_eval.py` 和 `MoLe-VLA-Pytorch/`

## 第 1 步：安装 Miniconda + 创建环境（15 分钟）

1. 安装 [Miniconda Windows x86_64](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)（一路默认）
2. 打开 **Anaconda Prompt**，执行：

```powershell
conda create -n mole python=3.10 -y
conda activate mole
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow numpy opencv-python einops transforms3d
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple imageio imageio-ffmpeg matplotlib
```

✅ 验证：`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` → `2.x.x True`

## 第 2 步：安装 LIBERO（15 分钟）

```powershell
cd phase5-vla-reproduction/code
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

> 如果 LIBERO 要求 robosuite，执行 `pip install robosuite==1.4.0`（LIBERO 官方配套版本）

✅ 验证：`python -c "from libero.libero import benchmark; print('LIBERO OK')"`

## 第 3 步：安装 MoLe-VLA 代码依赖（10 分钟）

```powershell
cd ..\MoLe-VLA-Pytorch

# ① 关键：prismatic_new 复制为 prismatic（代码 import 的是 prismatic，仓库只带 prismatic_new）
xcopy /E /I prismatic_new prismatic

# ② 直接装依赖（仓库无 setup 文件，是纯源码运行，不要 pip install -e .）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers==4.40.1 sentencepiece==0.1.99 timm==0.9.10 tokenizers==0.19.1 peft==0.11.1 accelerate draccus einops json-numpy jsonlines rich protobuf wandb
```

> ⚠️ `transformers==4.40.1` 是硬 pin（代码依赖其 API），不要升级
> tensorflow 2.15 仅训练/RLBench 数据需要，推理可跳过

✅ 验证：`python -c "from vla import load_vla; print('VLA import OK')"`

## 第 4 步：下载权重（30-60 分钟，可提前开始！）

**下载 CogACT-Base（30GB，MoLe-VLA 的 backbone）**：

```powershell
pip install -U "huggingface-hub>=0.24"
huggingface-cli download CogACT/CogACT-Base --local-dir D:\models\CogACT-Base
```

> 如果 huggingface.co 慢：设置 `$env:HF_ENDPOINT="https://hf-mirror.com"` 再执行
> 权重保存到 `D:\models\CogACT-Base`（本地路径，脚本用 `--ckpt D:\models\CogACT-Base` 加载）

✅ 验证：`D:\models\CogACT-Base\checkpoints\CogACT-Base.pt` 存在（约 30GB）

## 第 5 步：生成 LIBERO 演示数据（5 分钟，可选但推荐）

```powershell
cd ..\LIBERO
python scripts/collect_demonstrations.py --task-name libero_spatial --num-episodes 10
```

> 生成在 `LIBERO/datasets/` 下。不做也行——直接推理。

## 第 6 步：推理 + 录视频（30 分钟）

```powershell
cd ..\..\code
conda activate mole
python lib_libero_eval.py --ckpt D:\models\CogACT-Base --action-model DiT-B --benchmark libero_spatial --task-idx 0 --num-episodes 5
```

✅ 验证：`phase5-vla-reproduction/output/` 出现 `libero_spatial_task0.gif`

### 如果机器人不动或乱动（动作尺度不匹配）

| 现象 | 调整 |
|------|------|
| 机器人完全不动 | `--action-scale 1.0` 或 `2.0` |
| 机器人乱飞/抖动 | `--action-scale 0.2` 或 `0.1` |
| 换 policy_setup | `--policy-setup widowx_bridge`（bridge 数据训练更接近桌面操作） |

## 第 7 步：回传结果

```powershell
git add phase5-vla-reproduction/output/
git commit -m "Phase 5: LIBERO 推理视频"
git push
```

## 常见问题

| 问题 | 解决 |
|------|------|
| `OffScreenRenderEnv` 失败（OSMesa 不支持 Windows） | 脚本已自动回退 `RobosuiteEnv`（弹窗渲染，有显示器即可） |
| CUDA 版本不对 | RTX 5880 Ada 是 Ada 架构，cu121 驱动 >= 525 即可 |
| 显存 OOM | 48GB 足够 7B bf16，若仍 OOM 加 `--action-model DiT-S` |
| LIBERO 安装报 robosuite 版本冲突 | `pip install robosuite==1.4.0` |
