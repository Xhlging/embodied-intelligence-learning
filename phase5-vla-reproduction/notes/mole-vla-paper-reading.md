# MoLe-VLA 论文阅读笔记

> 论文: [MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation](https://arxiv.org/abs/2503.20384)  
> Zhang et al. · AAAI 2026  
> 官方代码: https://github.com/RoyZry98/MoLe-VLA-Pytorch

## 核心贡献

**MoLe（Mixture-of-Layers，层混合）机制**——让 VLA 模型的 LLM 层**动态跳层**，在保持任务精度的前提下大幅降低推理算力：

1. **选择性层激活**：每层 LLM 是一个"专家"，轻量 router 根据机器人状态生成二值门控向量，只激活 top-k 层；被跳过的层直接透传（$h_k = G_k \cdot \pi_k(h_{k-1}) + (1-G_k) \cdot h_{k-1}$）
2. **STAR router（Spatial-Temporal Aware Router）**：从视觉特征算空间路由权重、从文本特征算时间路由权重，投影到共享空间 + Gumbel-Softmax 可微分层选择
3. **CogKD（Cognition Self-Knowledge Distillation）**：教师-学生框架（完整模型=教师，跳层模型=学生），用可学习 cognition tokens 识别关键 token（ToIs），重加权模仿损失 + Reverse-KL 保留任务关键信息

**动机**：LLM 层在机器人任务中存在高冗余（相邻层输出余弦相似度 >90%），且语义信息集中在最终层——简单剪枝会丢关键语义。

## 方法概述

### 为什么需要跳层

- 7B VLA 在 RTX 4090 上只有 5-12 Hz，远低于 Franka 机械臂 50-1000 Hz 控制频率
- 已有的稀疏化方法（early exit、token pruning、MoD）丢弃的是**最终层**——而最终层编码了机器人任务最相关的语义

### 关键结果

| 指标 | MoLe-VLA | 对比 |
|------|---------|------|
| LLM 计算量 | 985.8 GFLOPs | CogAct 1935.8 GFLOPs（**~5.6× 降低**） |
| 推理频率 | 15.7 Hz（8bit 量化, 4090D） | CogAct FP16 9.8 Hz |
| 显存 | 8887 MB | CogAct 16055 MB |
| RLBench 10 任务成功率 | 60.8% | CogAct 57.2%、DeeR 59.2%、MoD 56.4% |
| 真实环境（Franka 3 任务） | 70.0% | CogAct 66.7% |

### 架构（基于 CogACT）

```
图像 + 指令 → SigLIP/DINOv2 视觉编码 → Llama2-7B（MoLe 跳层 router 注入）→ DiT 动作头 → 7-DoF 动作
```

MoLe 是**插件式效率架构**：可加载在任意 CogACT 权重上（论文验证 MoLe-CogAct 和 MoLe-OpenVLA 两个变体）。

## 与 ACT 的对比

| 维度 | ACT | MoLe-VLA |
|------|-----|----------|
| 模型规模 | ~80M | 7B（+MoLe 跳层） |
| 动作生成 | CVAE + Transformer | Flow Matching / DiT |
| 语言条件 | ❌ 无 | ✅ VLM 指令理解 |
| 效率机制 | 动作分块 | 层跳变（MoLe） |
| 训练数据 | 50 演示/任务 | 大规模机器人数据 |
| 通用性 | 单任务 | 多任务零样本 |

**本质区别**：ACT 是"动作策略"，MoLe-VLA 是"通用 VLA + 效率优化"。

## 在 LEBERO（LIBERO）上的适配

MoLe-VLA 论文实验在 **RLBench + 真实环境**（未用 LIBERO）——这正是 Phase 5 任务要求（论文未用环境）。

本阶段采用的**小规模复现路线**（任务要求"小规模 VLA"）：
- 0.5B SmolVLA（同为 VLA 范式：VLM + 动作头）
- 在 LIBERO 完成 LoRA 微调 + 推理（训练-仿真交互闭环）
- MoLe 跳层机制作为效率优化思想，未在小模型上复现（7B 权重 30GB 超出单卡训练成本，见 report.md）

## 关键收获

1. **层冗余是真实存在的**：相邻 LLM 层输出相似度 >90%——跳层省算力是合理设计
2. **最终层不可丢**：语义集中在深层，稀疏化必须保留
3. **Router 是效率-精度权衡的核心**：STAR 用视觉+文本双路信号决策跳层
4. **对硬件受限场景的意义**：8bit 量化 + 跳层使 7B VLA 达到 15.7Hz 实时控制——具身智能部署的关键路径
5. **本复现的启示**：小模型（0.5B）+ LoRA 是 4060/5880 级别硬件上可行的 VLA 训练方案（本阶段实证：48 分钟完成 20K 步）
