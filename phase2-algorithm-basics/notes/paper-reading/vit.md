# ViT 论文阅读笔记

> 论文: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)  
> Dosovitskiy et al. · ICLR 2021

## 历史地位

ViT（Vision Transformer）是第一个**在大规模图像分类上超越 CNN 的纯 Transformer 架构**。论文标题很巧妙——"An Image is Worth 16x16 Words" 是对 "A picture is worth a thousand words" 的戏仿，传达了核心思想: **把图像切成 16×16 的 patch，当作 NLP 中的 word token 来处理**。

## 核心贡献

1. **证明了 Transformer 无需 CNN 的归纳偏置即可在视觉任务上达到 SOTA** — CNN 的局部性、平移不变性不是必须的，纯注意力也可以
2. **简洁的架构设计** — 几乎就是把 NLP Transformer 直接搬到图像上，只改了输入处理方式
3. **揭示了"大数据预训练"对 Transformer 的关键作用** — 小数据上不如 CNN，大数据上超越 CNN

## 架构概览

```
Input: 224×224×3
    │
    ▼
Patch Embedding: 切分为 14×14 个 16×16 的 patch
每个 patch 展平为 768 维向量（16×16×3 = 768）
    │
    ▼ 加 [class] token + Position Embedding
    │
    ▼
┌────────────────────────────┐
│       Transformer Encoder   │
│  ┌─ Layer Norm ──────────┐  │
│  │ Multi-Head Attention  │  │
│  └─ + 残差 ──────────────┘  │
│  ┌─ Layer Norm ──────────┐  │
│  │ MLP (768→3072→768)   │  │  × L 层
│  └─ + 残差 ──────────────┘  │
└────────────────────────────┘
    │
    ▼ 取 [class] token 的输出
    │
MLP Head → 1000 classes
```

**关键组件**:

### Patch Embedding

这是 ViT 与 NLP Transformer 唯一的本质区别——如何把图像变成 token:

1. 将 $H \times W \times C$ 的图像切成 $N = \frac{HW}{P^2}$ 个 $P \times P \times C$ 的 patch
2. 每个 patch 展平成一个向量，维度为 $P^2 \cdot C$
3. 通过一个可学习的线性投影 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 映射到 $D$ 维（Transformer 的隐藏维度）

默认配置: $H = W = 224$, $P = 16$, $C = 3$, $N = 196$, $D = 768$

### [class] Token

- 借鉴 BERT 的做法，在输入序列前加一个可学习的 `[class]` token $x_{\text{class}}$
- 经过 Transformer 处理后，`[class]` 位置的输出用于分类
- 这个 token 在整个自注意力过程中可以与所有 patch 交互，起到了"聚合全局信息"的作用
- 替代方案: 对所有 patch 输出做全局平均池化（GAP）

### Position Embedding

- 给每个 patch 加上**可学习的 1D 位置编码**
- ViT 实验了 1D、2D、相对位置编码，发现可学习的 1D 编码效果足够好
- 位置编码使模型知道每个 patch 在图像中的位置——没有它，所有 patch 的位置信息就丢失了

## 与 CNN 的对比

| 维度 | CNN | ViT |
|------|-----|-----|
| **归纳偏置** | 强（局部性、平移不变性） | 弱（只有 patch 级别的位置编码） |
| **感受野** | 局部 → 逐层扩大 | 全局（第一层就能看到全图） |
| **参数效率** | 高（卷积核共享） | 需要更多数据 |
| **小数据表现** | 好 | 差（无归纳偏置，容易过拟合） |
| **大数据表现** | 饱和 | 持续提升 |
| **计算复杂度** | $O(k^2 H W C)$ | $O(N^2 D)$，$N$ 为 patch 数 |
| **可解释性** | 特征图可视化 | 注意力图可视化（更直观） |

### 核心发现: 数据规模是关键

| 训练数据 | ViT 表现 |
|----------|---------|
| ImageNet (1.3M) | 不如 ResNet |
| ImageNet-21k (14M) | 持平或略好 |
| JFT-300M (300M) | **超越** CNN |

**原因**: CNN 的局部性和平移不变性是一种**强先验**——小数据时这帮助模型快速收敛；大数据时这种限制反而阻碍模型学习更灵活的模式。Transformer 没有这些限制，在大数据上有更大的潜力。

## 实验亮点

| 模型 | ImageNet Top-1 | 参数量 | 训练数据 |
|------|---------------|--------|---------|
| ViT-H/14 | 88.6% | 632M | JFT-300M |
| ViT-L/16 | 87.8% | 307M | JFT-300M |
| BiT-L (ResNet152×4) | 87.5% | 928M | JFT-300M |
| Noisy Student (EfficientNet-L2) | 88.4% | 480M | JFT-300M + ImageNet |

- ViT-H/14 在多个 benchmark 上达到当时最优
- 而且训练计算量比 CNN 少

## 在具身智能中的应用前景

1. **取代 CNN 作为视觉编码器**: VLA 模型中，ViT 可将相机图像编码为 token 序列，与文本指令 token、动作 token 一起输入 Transformer 做统一处理

2. **多视角融合**: 机器人通常有多路相机（腕部、头部、第三人称），ViT 的 patch 机制天然支持多图像拼接或多路径编码

3. **注意力可视化 → 可解释的机器人行为**: ViT 的注意力图可以显示模型在做抓取决策时"看了图像的哪部分"——是物体本身还是背景？这对调试具身智能模型非常有用

4. **挑战**:
   - ViT 推理比轻量 CNN 慢——对实时机器人控制可能有影响
   - 机器人图像的分布（鱼眼镜头、低分辨率、运动模糊）与 ImageNet 差异大——需要领域适配
   - Patch 大小固定 → 远处的物体可能只占几个 pixel，信息量不足（高分辨率+小 patch 方案计算量激增）
