# MLP / CNN / ViT 对比实验

## 快速开始

```bash
# 安装依赖
pip install torch torchvision matplotlib

# 训练全部三种模型
python train.py --model all

# 或单独训练
python train.py --model mlp
python train.py --model cnn
python train.py --model vit

# 在 CPU 上训练（如果无 GPU）
python train.py --model all --device cpu

# 可视化对比结果
python plot_results.py
```

## 模型说明

| 模型 | 特点 | 参数量 |
|------|------|--------|
| **MLP** | 三层全连接，展平图像为 3072 维向量，无空间结构感知 | ~3.4M |
| **CNN** | 三层卷积 + 池化，有局部感受野和平移不变性 | ~3.7M |
| **ViT** | 将图像切为 4×4 patch，用 6 层 Transformer 处理全局关系 | ~5.6M |

## 实验目的

1. **理解不同架构的归纳偏置差异** — MLP 无空间先验，CNN 有局部性，ViT 有全局注意力
2. **观察不同模型在相同数据上的收敛行为** — 训练/验证曲线、收敛速度
3. **为具身智能中的视觉编码器选择提供直觉** — 不同任务和约束下适合什么架构

## 输出

- `checkpoints/` — 模型权重
- `results/` — 训练历史 JSON + 对比图表 (PNG)
  - `accuracy_loss_comparison.png` — 准确率和损失曲线
  - `bar_comparison.png` — 最终准确率柱状图
  - `params_vs_accuracy.png` — 参数量 vs 准确率

## 数据集

CIFAR-10: 10 类，50,000 训练 + 10,000 测试，32×32 彩色图像

小组后续会指定具体数据集，替换 `get_dataloaders()` 中的数据集加载逻辑即可。
