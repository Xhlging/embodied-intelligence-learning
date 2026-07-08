# Transformer 论文阅读笔记

> 论文: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
> Vaswani et al. · NeurIPS 2017

## 历史地位

Transformer 打破了 NLP 领域 RNN/LSTM 长达数十年的统治。论文标题 "Attention Is All You Need" 是一句宣言: **只用注意力机制就够了，不需要循环和卷积**。如今 Transformer 已经不仅是 NLP 的基础架构，更扩展到了计算机视觉（ViT）、多模态（CLIP）、机器人控制（ACT、RT-2）等领域。

## 核心贡献

1. **完全基于自注意力的序列建模架构** — 没有 RNN、没有 CNN，纯注意力
2. **并行化训练** — RNN 必须串行处理序列（$O(n)$ 时间步），Transformer 可以一次性并行处理整个序列（$O(1)$ 时间步）
3. **长程依赖建模** — 自注意力的全局感受野使任意两个位置的 token 可以直接交互

## 架构概览

```
                Output Probabilities
                       ▲
                  Softmax
                       ▲
                   Linear
                       ▲
              ┌─ Add & Norm ─┐
              │ Feed Forward │
              └─ Add & Norm ─┘
              │Multi-Head    │
              │  Attention   │
              └─ Add & Norm ─┘  × N  (编码器 N=6)
              │   Masked     │
              │Multi-Head    │
              │  Attention   │
              └─ Add & Norm ─┘
              │Multi-Head    │
              │  Attention   │
              └──────────────┘
    ▲                              ▲
Input Embedding            Output Embedding
    ▲                              ▲
Input tokens              Output tokens (shifted right)
```

**三个核心组件**:

| 组件 | 英文 | 作用 |
|------|------|------|
| 多头自注意力 | Multi-Head Self-Attention | 捕捉 token 间的全局关系 |
| 前馈网络 | Feed-Forward Network (FFN) | 对每个 token 独立做非线性变换 |
| 残差 + 层归一化 | Add & Norm | 稳定训练，防止梯度消失 |

### Self-Attention 机制

**核心公式**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中:
- $Q$（Query）: 我在"查找"什么
- $K$（Key）: 我有什么"标签"可供匹配
- $V$（Value）: 匹配后我要"提取"什么信息
- $\sqrt{d_k}$: 缩放因子，防止点积过大导致 softmax 梯度消失

**步骤拆解**:

```
输入: X = [x₁, x₂, ..., xₙ]  (n 个 token)
    │
    ▼ 线性投影
Q = X·W_Q,  K = X·W_K,  V = X·W_V
    │
    ▼ 计算注意力分数
Scores = Q·Kᵀ / √d_k
    │
    ▼ Softmax（得到注意力权重）
Weights = softmax(Scores)
    │
    ▼ 加权求和
Output = Weights·V
```

**直觉理解**: 假设一句话 "The cat sat on the mat"，当处理 "sat" 时:
- Query("sat") 与 Key("cat") 的相似度很高 → 高注意力权重
- Query("sat") 与 Key("the") 的相似度很低 → 低注意力权重
- 最终 "sat" 的输出 = 高权重 × Value("cat") + 低权重 × Value("the") + ...

### Multi-Head Attention

单头注意力可能只关注一种模式（如语法关系），多头可以同时关注多种模式（语法 + 语义 + 位置等）:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

- 论文中使用 $h=8$ 个头
- 每个头处理 $d_k = d_{\text{model}} / h = 512 / 8 = 64$ 维的子空间
- 不同头可以关注不同的位置关系——有点类似于 CNN 中不同卷积核学不同特征

### Positional Encoding

自注意力本身**没有序列位置信息**——交换两个 token 的位置，注意力输出不变。因此需要人为注入位置信息:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

- 使用正弦/余弦函数编码位置 → 模型可以外推到比训练时更长的序列
- 也探索了可学习的位置编码（效果类似）

## 为什么 Transformer 比 RNN/LSTM 好

| 维度 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 计算方式 | 串行（$O(n)$ 步） | 并行（$O(1)$ 步） |
| 长程依赖 | 需要梯度穿过所有时间步（消失/爆炸） | 任意两位置直接交互（$O(1)$ 路径长度） |
| 训练速度 | 慢（无法并行） | 快（充分利用 GPU 并行） |
| 序列长度限制 | 受限于梯度传播距离 | 受限于 $O(n^2)$ 的内存（长序列是挑战） |
| 可解释性 | 隐状态难以解释 | 注意力权重可视化 |

## 与具身智能的关联

1. **ACT（Action Chunking with Transformers）** 直接使用 Transformer 作为 policy 网络: 输入多模态观测序列（图像 + 关节状态），输出动作块（action chunk）

2. **Mole-VLA / RT-2** 等 VLA 模型的核心是 Transformer: 视觉 token + 文本 token + 动作 token → 统一的序列建模

3. **Transformer 的自注意力非常适合机器人**:
   - 多模态融合: 图像 patch、文本指令、关节状态作为不同的 token，自注意力自动学习它们之间的关联
   - 时序建模: 历史观测 → 未来动作，Transformer 一次性看到完整上下文
   - 动作分块: 一次性输出多个未来动作（action chunking），解决高频控制与低频推理的不匹配

4. **挑战**:
   - $O(n^2)$ 复杂度——机器人需要高频实时推理，大 Transformer 的延迟是个问题
   - 机器人场景的 token 序列远不如文本规整——图像 patch、传感器读数、历史动作的分布差异很大
