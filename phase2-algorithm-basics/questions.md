# Phase 2: 算法基础 — 问题回答

> 阅读材料: AlexNet, Transformer, ViT 论文 + B站《动手学深度学习》

---

## 问题 1: 什么是深度学习，什么是机器学习？

### 机器学习（Machine Learning）

机器学习是人工智能的一个分支，其核心思想是：**让计算机从数据中学习规律，而非通过显式编程来完成任务**。

Tom Mitchell 的经典定义（1997）:

> 对于某类任务 $T$ 和性能度量 $P$，如果一个计算机程序在 $T$ 上以 $P$ 衡量的性能随着经验 $E$ 而自我完善，那么我们称这个计算机程序从经验 $E$ 中学习。

**传统机器学习的典型流程**:

$$\text{原始数据} \rightarrow \boxed{\text{人工特征工程}} \rightarrow \text{特征向量} \rightarrow \boxed{\text{简单模型}} \rightarrow \text{预测}$$

关键特征: **特征由人类专家手工设计**——SIFT、HOG 用于图像，TF-IDF 用于文本。

常见算法: 线性回归、逻辑回归、SVM、决策树、随机森林、KNN、K-Means 等。

### 深度学习（Deep Learning）

深度学习是机器学习的一个子集，使用**多层神经网络**自动学习数据的层次化特征表示。

**深度学习的典型流程**:

$$\text{原始数据} \rightarrow \boxed{\text{多层神经网络}} \rightarrow \text{层次化特征} \rightarrow \text{预测}$$

关键特征: **特征由网络自动学习**——底层学习边缘/纹理，中层学习部件，高层学习语义概念。

### 核心区别

| 维度 | 传统机器学习 | 深度学习 |
|------|------------|---------|
| 特征工程 | **人工设计**（依赖领域知识） | **自动学习**（端到端） |
| 模型复杂度 | 简单（几十到几百参数） | 复杂（百万到千亿参数） |
| 数据需求 | 较少（几千到几万） | 大量（几万到几亿） |
| 计算需求 | CPU 足够 | GPU/TPU 必需 |
| 可解释性 | 较好 | 较差（黑箱） |
| 代表模型 | SVM, Random Forest, XGBoost | CNN, Transformer, GPT |

### 两者的关系

$$\text{AI} \supset \text{Machine Learning} \supset \text{Deep Learning}$$

深度学习是机器学习的一种方法——它仍然遵循"从数据中学习"的基本范式，但通过深层网络结构自动完成了特征学习这一步，使得模型可以直接从原始数据（像素、字符）端到端地学习。

---

## 问题 2: 前向反向传播为什么能够发挥作用？

### 前向传播（Forward Propagation）

前向传播将输入逐层变换为输出。每一层执行两个操作:

1. **线性变换**: $z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$
2. **非线性激活**: $a^{(l)} = \sigma(z^{(l)})$

整个网络构成一个**复合函数**:

$$\hat{y} = f(x; \theta) = f^{(L)}(f^{(L-1)}(...f^{(1)}(x)))$$

其中 $\theta = \{W^{(1)}, b^{(1)}, ..., W^{(L)}, b^{(L)}\}$ 是所有可学习参数。

**为什么多层很重要？**

- **万能逼近定理（Universal Approximation Theorem）**: 单隐层网络可以逼近任意连续函数，但需要的神经元数量可能是指数级的
- **深层网络用更少的参数实现相同的表达能力**: 深度带来参数效率——每一层学习一种抽象层次，层层组合产生指数级的表示能力
- **层次化特征学习**: 浅层学习局部模式（边缘），深层学习全局语义（物体类别）

### 反向传播（Backpropagation）

反向传播是通过**链式法则（Chain Rule）**高效计算损失函数对每个参数的梯度的算法。

**核心思想**:

$$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \frac{\partial a^{(L)}}{\partial z^{(L)}} \cdot \frac{\partial z^{(L)}}{\partial a^{(L-1)}} \cdot ... \cdot \frac{\partial z^{(l+1)}}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}$$

**为什么能发挥作用——三个关键**:

#### 1. 链式法则 = 动态规划

反向传播本质上是一种**动态规划（Dynamic Programming）**:

- 从输出层开始，逐层向后计算梯度
- 每一层复用前一层的计算结果（误差项 $\delta^{(l)}$）
- 将计算复杂度从 $O(n^2)$ 降到 $O(n)$（$n$ 为参数个数）

如果没有反向传播，计算 $n$ 个参数的梯度需要 $n$ 次前向传播（数值微分）；有了反向传播，一次前向 + 一次反向就够了。

#### 2. 梯度指明了改进方向

梯度 $\frac{\partial L}{\partial \theta}$ 告诉模型: **"参数往哪个方向调整，损失下降最快"**

$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \frac{\partial L}{\partial \theta}$$

这是一个**迭代优化**过程——每一步沿着局部最陡下降方向微调参数，最终收敛到损失函数的局部最小值。

#### 3. 非线性激活使深层有意义

如果没有非线性激活函数（$\sigma(x) = x$），多层网络等价于单层:

$$W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W'x + b'$$

非线性（ReLU, Sigmoid 等）打破了这种退化，使每一层真正学到不同的表示。

### 整个过程

```
输入 x
  │
  ▼ 前向传播
┌──────────────────────────────────┐
│ z(1)=W(1)x+b(1) → a(1)=σ(z(1))  │
│ z(2)=W(2)a(1)+b(2) → a(2)=σ(z(2)) │
│ ...                              │
│ z(L)=W(L)a(L-1)+b(L) → ŷ=a(L)   │
└──────────────────────────────────┘
  │
  ▼ 计算损失
L = loss(ŷ, y)
  │
  ▼ 反向传播
┌──────────────────────────────────┐
│ δ(L) = ∂L/∂z(L)                 │
│ δ(L-1) = (W(L))ᵀδ(L) ⊙ σ'(z(L-1))│
│ ...                              │
│ ∂L/∂W(l) = δ(l)(a(l-1))ᵀ        │
└──────────────────────────────────┘
  │
  ▼ 参数更新
W(l) = W(l) - η·∂L/∂W(l)
```

---

## 问题 3: 损失函数是如何影响模型的训练的？

损失函数 $L(\hat{y}, y)$ 衡量**模型预测 $\hat{y}$ 与真实标签 $y$ 之间的差距**。它通过三个层面影响训练:

### 1. 损失函数定义了"好"的标准

不同的损失函数对应不同的优化目标:

| 损失函数 | 公式 | 隐含假设 | 适用场景 |
|----------|------|---------|----------|
| **MSE** | $$\frac{1}{n}\sum(\hat{y}_i - y_i)^2$$ | 误差服从高斯分布 | 回归 |
| **MAE** | $$\frac{1}{n}\sum|\hat{y}_i - y_i|$$ | 误差服从拉普拉斯分布 | 回归（对异常值鲁棒） |
| **Cross-Entropy** | $$-\sum y_i \log(\hat{y}_i)$$ | 输出为概率分布 | 分类 |
| **Hinge Loss** | $$\max(0, 1 - y \cdot \hat{y})$$ | 关注分类边界的 margin | SVM / 二分类 |

### 2. 损失函数的梯度决定了学习信号

参数更新的唯一依据是损失函数的梯度:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L$$

**MSE vs Cross-Entropy 的关键差异**:

考虑一个二分类问题，真实标签 $y=1$:

- **MSE + Sigmoid**: $L = (\sigma(z) - 1)^2$，梯度 $\frac{\partial L}{\partial z} = 2(\sigma(z)-1) \cdot \sigma(z)(1-\sigma(z))$
  - 当模型非常自信地错了（$\sigma(z) \approx 0$）时，梯度接近 0 → **学不动**
  
- **Cross-Entropy + Sigmoid**: $L = -\log(\sigma(z))$，梯度 $\frac{\partial L}{\partial z} = \sigma(z) - 1$
  - 当模型非常自信地错了时，梯度接近 -1 → **仍然在学习**

这就是为什么分类任务用交叉熵而不用 MSE——交叉熵在模型错误时提供更强的学习信号。

### 3. 损失函数塑造了损失景观（Loss Landscape）

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

- **凸损失函数**（如线性回归的 MSE）: 只有一个全局最小值，梯度下降保证收敛
- **非凸损失函数**（如神经网络的损失）: 存在多个局部最小值、鞍点。Adam/RMSprop 等自适应优化器通过动量和自适应学习率来更有效地探索损失景观

### 4. 实际影响总结

| 影响 | 说明 |
|------|------|
| **收敛速度** | Cross-Entropy 比 MSE 在分类任务上收敛更快 |
| **最终性能** | 选择匹配任务的损失函数（分类用 CE，回归用 MSE） |
| **对异常值的敏感性** | MSE 对异常值敏感（平方惩罚），MAE 更鲁棒 |
| **类不平衡处理** | 可在 Cross-Entropy 中加类别权重 $\alpha_i$ |
| **多任务学习** | 不同任务用不同损失，加权求和 $L_{\text{total}} = \sum \lambda_i L_i$ |

---

## 问题 4: 为什么会梯度消失/爆炸？如何解决？

### 梯度消失（Vanishing Gradient）

**原因**:

在深层网络中，反向传播的梯度是连乘积:

$$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial a^{(L)}} \cdot \prod_{k=2}^{L} \left( \frac{\partial a^{(k)}}{\partial z^{(k)}} \cdot \frac{\partial z^{(k)}}{\partial a^{(k-1)}} \right) \cdot \frac{\partial a^{(1)}}{\partial z^{(1)}} \cdot \frac{\partial z^{(1)}}{\partial W^{(1)}}$$

其中 $\frac{\partial a^{(k)}}{\partial z^{(k)}} = \sigma'(z^{(k)})$ 是激活函数的导数。

**Sigmoid 的导数为 $\sigma'(x) = \sigma(x)(1-\sigma(x))$**，最大值为 0.25。当 $L$ 层连乘后:

$$0.25^L \xrightarrow{L \to \infty} 0$$

网络越深，浅层梯度越小 → 浅层参数几乎不更新 → 网络学不到有效特征。

**Tanh 同理**: $\tanh'(x) = 1 - \tanh^2(x)$，最大值为 1，但饱和区导数趋近于 0。

### 梯度爆炸（Exploding Gradient）

**原因**:

如果权重初始化过大，$\frac{\partial z^{(k)}}{\partial a^{(k-1)}} = W^{(k)}$ 中元素 > 1，连乘导致:

$$\prod |W| > 1 \xrightarrow{L \to \infty} \infty$$

梯度指数级增长 → 参数更新步长过大 → 损失震荡甚至发散（变成 NaN）。

### 解决方案

#### 1. 更好的激活函数 — ReLU

$$\text{ReLU}(x) = \max(0, x)$$

- 正半轴导数为 1（不衰减），直接解决了梯度消失的核心问题
- 计算简单（无指数运算）
- 变体: **Leaky ReLU**（负半轴有小斜率）、**GELU**（Transformer 常用）等

```python
# ReLU 及其导数
def relu(x):      return max(0, x)
def relu_grad(x): return 1 if x > 0 else 0
```

#### 2. 合理的权重初始化

| 初始化方法 | 公式（均匀分布范围） | 适用激活函数 |
|-----------|---------------------|-------------|
| **Xavier/Glorot** | $$W \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}, \frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}\right]$$ | Sigmoid, Tanh |
| **He/Kaiming** | $$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$ | ReLU |

核心思想: 保持各层激活值和梯度的方差一致，避免逐层衰减或放大。

#### 3. 批归一化（Batch Normalization）

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

- 将每层输入标准化为零均值、单位方差
- 减少内部协变量偏移（Internal Covariate Shift）
- 使梯度流动更稳定，允许使用更大的学习率
- 有一定的正则化效果（mini-batch 的噪声）

#### 4. 残差连接（Residual Connection）

$$\text{output} = \mathcal{F}(x) + x$$

- ResNet 的核心创新: 让梯度可以通过"跳跃连接"直接传到浅层
- 恒等映射的导数为 1，保证了梯度不会消失
- 使得训练 100+ 层甚至 1000+ 层网络成为可能

#### 5. 梯度裁剪（Gradient Clipping）

```python
# 按范数裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 按值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

- 防止梯度爆炸的最直接方法
- 在 RNN/LSTM/Transformer 训练中几乎是标配

#### 6. 更好的优化器

- **Adam**: 结合动量和自适应学习率，对梯度尺度不敏感
- **RMSprop**: 按梯度平方的移动平均调整学习率

#### 方法对比

| 方法 | 主要解决 | 额外收益 | 代价 |
|------|---------|---------|------|
| ReLU | 梯度消失 | 计算更快 | 神经元可能"死亡"（永远输出 0） |
| Xavier/He 初始化 | 两者 | 收敛更快 | 几乎无 |
| Batch Norm | 两者 | 正则化、加速训练 | 额外计算、对小 batch 不稳定 |
| Residual Connection | 梯度消失 | 可训练极深网络 | 增加参数量和计算量 |
| Gradient Clipping | 梯度爆炸 | 训练更稳定 | 可能限制有效学习信号 |

---

## 问题 5: 机器学习的深度学习算法如何设计和训练？

不需要设计具体的算法，只需提供思路。

### 整体流程

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ 问题定义  │ → │ 数据准备  │ → │ 模型设计  │ → │ 训练优化  │ → │ 评估部署  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

### 步骤 1: 问题定义

| 问题类型 | 典型任务 | 输出形式 |
|----------|---------|---------|
| **分类** | 图像识别、情感分析 | 类别标签 |
| **回归** | 房价预测、轨迹预测 | 连续值 |
| **生成** | 文本生成、图像生成 | 序列/图像 |
| **序列决策** | 机器人控制、游戏 AI | 动作序列 |

明确: 输入是什么？输出是什么？评价标准是什么？

### 步骤 2: 数据准备

1. **数据收集**: 获取足够多、有代表性的数据
2. **数据清洗**: 处理缺失值、异常值、重复数据
3. **数据标注**: 分类需要标签，生成可能需要 text-image pair
4. **数据增强**: 旋转、翻转、颜色抖动 → 增加数据多样性
5. **数据划分**: 训练集（70%）/ 验证集（15%）/ 测试集（15%）
6. **归一化**: 将输入缩放到合理范围（如 [0,1] 或均值 0 方差 1）

### 步骤 3: 模型设计

**宏观架构选择**:

| 数据类型 | 推荐架构 | 原因 |
|----------|---------|------|
| 图像（空间结构） | CNN / ViT | 局部感受野 / 自注意力捕捉全局关系 |
| 序列（时序依赖） | RNN / LSTM / Transformer | 循环连接 / 自注意力处理长程依赖 |
| 表格数据 | MLP | 无特殊结构，全连接即可 |
| 多模态 | 多分支 + 融合层 | 不同模态用不同编码器 |

**微观组件选择**:

1. **激活函数**: 隐藏层默认 ReLU/GELU，输出层依任务而定（分类 → Softmax，回归 → 恒等）
2. **层数和宽度**: 从简单开始，逐步增加。观察训练/验证损失判断是否需要更大模型
3. **归一化层**: BatchNorm（CNN）、LayerNorm（Transformer）
4. **正则化**: Dropout、Weight Decay、数据增强
5. **跳跃连接**: 网络 > 10 层时考虑

### 步骤 4: 训练配置

```
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 前向传播
        output = model(batch.x)
        
        # 2. 计算损失
        loss = criterion(output, batch.y)
        
        # 3. 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 4. 参数更新
        optimizer.step()
    
    # 5. 验证
    val_loss = evaluate(model, val_loader)
    
    # 6. 调整学习率
    scheduler.step(val_loss)
    
    # 7. 早停检查
    if no_improvement_for_n_epochs:
        break
```

**关键超参数**:

| 超参数 | 作用 | 典型值 |
|--------|------|--------|
| 学习率 $\eta$ | 控制参数更新步长 | $10^{-4}$ ~ $10^{-2}$ |
| Batch Size | 每次迭代的样本数 | 32 ~ 256 |
| Epochs | 完整遍历数据集的次数 | 几十到几百 |
| Optimizer | 参数更新策略 | Adam（首选）、SGD+Momentum |
| Weight Decay | L2 正则化强度 | $10^{-5}$ ~ $10^{-3}$ |

### 步骤 5: 训练技巧

1. **学习率调度**: 余弦退火（Cosine Annealing）、阶梯衰减、ReduceLROnPlateau
2. **早停（Early Stopping）**: 验证损失不再下降时停止，防止过拟合
3. **模型检查点（Checkpoint）**: 保存验证集上表现最好的模型
4. **混合精度训练**: 用 FP16 加速，节省显存
5. **梯度累积**: 小显存下模拟大 batch size

### 步骤 6: 评估与诊断

**观察训练/验证损失曲线判断问题**:

| 现象 | 诊断 | 对策 |
|------|------|------|
| 训练损失不下降 | 学习率太小/模型容量不足 | 增大学习率 / 增加层数 |
| 训练损失降但验证损失升 | **过拟合** | 加正则化 / 数据增强 / 减小模型 |
| 两者都不降 | 学习率太大 / 数据有问题 | 减小学习率 / 检查数据和标签 |
| 训练损失远低于验证损失 | 过拟合 | 更多数据 / Dropout / 早停 |

### 步骤 7: 测试与部署

- 在**测试集**上做最终评估（不用于调参）
- 导出模型（TorchScript / ONNX）
- 考虑推理速度和内存占用（模型压缩、量化、蒸馏）

### 完整思路总结

好的深度学习算法设计 = **理解问题 + 选对架构 + 充分数据 + 合理训练 + 持续迭代**。不需要在一开始就设计出完美模型——从一个可工作的简单基线开始，逐步添加复杂度，用验证指标指导每一步决策。

> 这一思路在具身智能中同样适用：先用简单 MLP 做 policy baseline，再尝试 CNN 处理视觉输入，最后上 Transformer 处理多模态和时序——Phase 2 的 MLP/CNN/ViT 对比实验正是这个逻辑的体现。
