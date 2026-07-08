# AlexNet 论文阅读笔记

> 论文: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)  
> Krizhevsky, Sutskever, Hinton · NIPS 2012

## 历史地位

AlexNet 是深度学习历史上的**里程碑**。在 2012 年 ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 中，它的 top-5 错误率为 15.3%，比第二名（26.2%）低了将近 11 个百分点。这篇论文标志着深度学习在计算机视觉领域的"寒武纪大爆发"。

## 核心贡献

1. **证明了深度 CNN 在大规模图像分类上的有效性** — 在此之前，人们普遍认为训练如此深的网络是不可行的
2. **GPU 训练的工程突破** — 用两块 GTX 580 GPU（各 3GB 显存）训练了 5-6 天，开创了 GPU 深度学习的先河
3. **一系列至今仍在使用的训练技巧** — ReLU、Dropout、数据增强等

## 网络结构

```
Input: 224×224×3
    │
    ▼
Conv1: 96 filters 11×11, stride 4, ReLU
    │
    ▼ Max Pooling 3×3, stride 2
    │
Conv2: 256 filters 5×5, padding 2, ReLU
    │
    ▼ Max Pooling 3×3, stride 2
    │
Conv3: 384 filters 3×3, padding 1, ReLU
    │
Conv4: 384 filters 3×3, padding 1, ReLU
    │
Conv5: 256 filters 3×3, padding 1, ReLU
    │
    ▼ Max Pooling 3×3, stride 2
    │
    ▼ Flatten
    │
FC6: 4096, ReLU, Dropout 0.5
    │
FC7: 4096, ReLU, Dropout 0.5
    │
FC8: 1000, Softmax
```

**结构特点**:
- 5 个卷积层 + 3 个全连接层 = 8 层（"Deep"的含义）
- 约 6000 万个参数
- 分两组在两张 GPU 上运行（第 3 层卷积才跨 GPU 通信）

## 关键创新点

### 1. ReLU 激活函数

$$\text{ReLU}(x) = \max(0, x)$$

- 之前的主流是 Tanh/Sigmoid，存在梯度饱和问题
- ReLU 训练速度比 Tanh **快 6 倍**（在 CIFAR-10 上达到 25% 错误率所需时间）
- 不饱和 → 梯度不会消失 → 深层网络可训练

### 2. Dropout 正则化

- 训练时以 0.5 概率随机丢弃神经元
- 测试时使用全部神经元，输出乘以 0.5
- 效果: 每次训练一个不同的子网络，等价于**模型集成（Ensemble）**
- 只用在全连接层（FC6、FC7），卷积层不用

### 3. 重叠池化（Overlapping Pooling）

- 池化窗口 3×3，步长 2（传统 CNN 是 2×2 窗口，步长 2）
- 有重叠 → 轻微减少过拟合（top-1 错误率降低 0.4%）

### 4. 数据增强

两种方法:
- **图像平移 + 水平翻转**: 从 256×256 图像中随机裁 224×224 的 patch → 训练集扩大 2048 倍
- **PCA 颜色增强**: 改变训练图像的 RGB 通道强度 → top-1 错误率降低 > 1%

### 5. 局部响应归一化（LRN）

- 在 ReLU 之后做跨通道归一化
- 模拟生物神经元的侧抑制（lateral inhibition）
- 实际效果有限（后来被 Batch Normalization 取代）

### 6. 双 GPU 训练

- 将网络分在两块 GTX 580 上
- 第 3 层卷积从两块 GPU 的上一层都接收输入（其他层只与本 GPU 的前一层连接）
- 这种设计使 top-5 错误率降低约 0.5%

## 实验结果

| 模型 | ImageNet Top-1 | ImageNet Top-5 |
|------|---------------|---------------|
| AlexNet (单模型) | 37.5% | 16.4% |
| AlexNet (5 模型平均) | — | 15.3% |
| ILSVRC-2011 冠军 | — | 25.8% |

**学到的特征可视化**:
- GPU 1 的卷积核: 大量**颜色无关**的滤波器（边缘、斑点检测器）
- GPU 2 的卷积核: 大量**颜色敏感**的滤波器（可能是因为 GPU 之间的连接结构不同）

## 对我的启发

1. **工程细节决定成败** — AlexNet 没有提出全新的理论（CNN + ReLU + Dropout 此前都有），但将这些技术组合在一起并成功训练出当时最深的网络，证明了"系统整合"的价值

2. **训练的规模是壁垒** — 6000 万参数，120 万张图片，5-6 天 GPU 训练。把模型做大、数据做多、训练做久——这套方法论至今仍然成立

3. **从 AlexNet 到具身智能** — CNN 的局部感受野和层次化特征提取天然适合处理视觉输入。在具身智能中，CNN 常作为视觉编码器（Visual Encoder），将 RGB 图像编码为嵌入向量，供下游的 policy 网络使用

4. **数据增强在具身智能中更加重要** — 机器人视觉面临光照变化、视角变化、遮挡等问题，比 ImageNet 的挑战更大。域随机化（Domain Randomization）本质上就是数据增强在仿真环境中的延伸
