# VLA 综述论文阅读笔记

> 论文: Vision Language Action Models in Robotic Manipulation: A Systematic Review  
> 作者: Muhammad U. Din, Waseem Akram, Lyes Saad Saoud, Jan Rosell, Irfan Hussain  
> 机构: Khalifa University (KUCARS), Universitat Politecnica de Catalunya  
> 链接: https://arxiv.org/abs/2507.10672  
> 配套仓库: https://github.com/Muhayyuddin/VLAs

## 论文概述

这篇综述系统性地梳理了 VLA（Vision-Language-Action）模型在机器人操纵领域的全貌，是一篇兼具技术参考和概念路线图意义的综述。核心数据：

- **102 个 VLA 模型**：从 2022-2025 年，涵盖 CLIPort、RT-1、RT-2、Gato、VIMA、Octo、OpenVLA、ACT、Pi-0、MoLe-VLA 等
- **26 个基础数据集**：从 EmbodiedQA 到 Open X-Embodiment、DROID、LIBERO、AgiBot World、Kaiwu 等
- **12 个仿真平台**：AI2-THOR、Habitat、Isaac Sim、Gazebo、MuJoCo、PyBullet、SAPIEN 等

论文提出了一个**二维数据集评测框架**：以任务复杂度（Task Complexity）为 x 轴、模态丰富度（Modality Richness）为 y 轴，发现当前数据集在"高复杂度 + 多模态融合"区域存在明显空白。

论文的四大贡献：
1. **VLA 架构的结构化分类法**：按视觉编码器、语言编码器、动作解码器三类组件组织
2. **VLA 数据集的量化基准评估**：引入 $$ \psi_{\text{task}} $$ 和 $$ \psi_{\text{mod}} $$ 两个评分
3. **仿真平台的深入审查**：按传感器模态、应用场景、技术能力和支持的数据集四个维度
4. **挑战与未来路线图**：架构、数据集、仿真三方面的挑战和方向

---

## VLA 模型架构

### 核心定义

VLA（Vision-Language-Action）模型是一类将**视觉感知、自然语言理解和具身控制**统一在单一学习框架中的模型。其核心思想是：
- 机器人观察环境（Vision）
- 理解人类指令（Language）
- 生成物理动作（Action）

### 通用架构（图 7 所示）

典型的 VLA 架构由三个并行编码器 + 一个动作解码器组成：

```
输入: 场景图像 + 自然语言指令 + 机器人内部状态
  │            │              │
  ▼            ▼              ▼
视觉编码器   语言编码器    状态编码器
(ViT/CNN)  (LLaMA/T5)   (MLP/Small Transformer)
  │            │              │
  └────────────┼──────────────┘
               ▼
         Token 拼接 + LLM 融合
               │
               ▼
          动作解码器
    (Diffusion Transformer /
     Autoregressive Head / MLP)
               │
               ▼
        连续控制信号
   (关节角度/末端执行器速度/力矩)
```

### 三大核心组件

#### 1. 视觉编码器（Vision Encoder）

将原始图像（RGB/深度/语义分割图）转换为固定长度的特征 Token：

| 类型 | 代表模型 | 使用场景 | 示例 VLA |
|------|---------|---------|---------|
| **CLIP/SigLIP ViT** | CLIP-ViT, SigLIP | 视觉-文本语义对齐强 | CLIPort, OpenVLA, Octo |
| **DINOv2** | DINOv2 | 长程空间依赖建模 | Gato, HybridVLA, CogACT |
| **CNN 系** | ResNet, EfficientNet | 轻量快速 | ACT, RT-1, QUAR-VLA |
| **Qwen2-VL** | Qwen2-ViT | 通用多模态 | Chain-of-Affordance, NORA |
| **多编码器融合** | SigLIP + DINOv2 | 互补特征 | OpenVLA, Edge VLA, DexGraspVLA |

#### 2. 语言编码器（Language Encoder）

将自然语言指令映射到共享的潜在空间：

| 类型 | 代表模型 | 特点 |
|------|---------|------|
| **LLaMA/Vicuna 系** | LLaMA-2, Vicuna-7B | 零样本推理能力强，最广泛使用 |
| **T5 系** | T5-base, T5-XXL | 灵活的编码器-解码器结构 |
| **GPT/Qwen 系** | GPT-4, Qwen2 | 通用性和紧凑部署平衡 |
| **Gemma 系** | Gemma-2B | 轻量部署（如 Pi-0, FAST） |
| **CLIP Text** | CLIP text encoder | 最小化对齐任务（如 CLIPort） |

#### 3. 动作解码器（Action Decoder）

将多模态嵌入转换为实际机器人指令：

| 类型 | 原理 | 代表模型 |
|------|------|---------|
| **Diffusion Transformer** | 迭代去噪生成连续轨迹 | Octo, RDT-1B, DexGraspVLA |
| **Autoregressive Transformer** | 逐 token 生成动作序列 | Gato, RT-2, GRAPE |
| **MLP/Token Predictor** | 直接映射到控制信号 | OpenVLA, RoboMamba |
| **CVAE** | 条件变分自编码器采样 | ACT, RoboAgent |
| **Flow Matching** | 流匹配生成 | Pi-0, SmolVLA |

### 架构趋势

- **端到端 (End-to-End)** 趋势明显：越来越多的模型直接从原始视觉和语言输入映射到控制命令，无需手工设计的中间表示
- **模块化/即插即用**：Vision encoder、Language encoder、Action decoder 可以独立更换和升级
- **Diffusion 主导**：Diffusion Transformer 成为动作解码的主流方案，因为它能建模复杂的多模态动作分布
- **多编码器融合**：许多模型采用 CLIP ViT + DINOv2 双视觉编码器架构以获得互补特征

---

## 动作空间（Action Space）

动作空间定义了 VLA 模型输出动作的表示形式和取值范围。主要分为三类：

### 1. 连续动作空间

输出连续的实数值，如关节角度增量 $$ \Delta\theta_i $$、末端执行器位姿增量 $$ \Delta x, \Delta y, \Delta z, \Delta roll, \Delta pitch, \Delta yaw $$ 或力矩 $$ \tau $$。

- **优点**：运动平滑、精度高，适合精细操作
- **缺点**：难以建模多模态分布（同一场景可能有多种合理动作）
- **解决**：Diffusion Policy 通过迭代去噪建模复杂分布

### 2. 离散动作空间

将动作空间离散化为有限类别（如"向前移动"、"左转"、"抓取"、"放下"）。RT-1 使用离散化动作 token，OpenVLA 预测 7 个离散动作 token。

- **优点**：与语言模型的 token 预测范式一致，训练简单
- **缺点**：丢失连续控制的精度

### 3. 混合/Token化动作空间

Gato 的贡献：将视觉、语言和控制任务统一 token 化，使用同一个 autoregressive Transformer 处理。FAST 引入频域动作 token 化，将推理速度提升至原来的 15 倍。

---

## Joint Point 与 EEF

这是机器人运动学中两种基本的控制空间：

### Joint Point（关节空间 / Joint Space）

控制对象是机器人各关节的角度位置 $$ \mathbf{q} = [q_1, q_2, ..., q_n] $$（n 为自由度）。

- **优点**：
  - 直接控制电机，无奇异性问题
  - 关节限位可以简单处理
  - 无逆运动学（IK）求解开销
- **缺点**：
  - 不直观——关节角度和末端位置没有直接的几何对应关系
  - 不同构型的机器人关节数量/排列不同，泛化困难
  - 末端轨迹形状难以预测

### EEF（末端执行器空间 / End-Effector Frame / Task Space）

控制对象是末端执行器（如夹爪）在笛卡尔空间中的位姿 $$ \mathbf{x} = [x, y, z, roll, pitch, yaw] $$。

- **优点**：
  - 直观——我们自然用"移动到桌上的杯子"而不是"移动关节1到30°、关节2到45°..."
  - 跨构型泛化——不同机器人都可以用相同的 EEF 位姿描述目标
  - 轨迹规划更自然（直线、圆弧等）
- **缺点**：
  - 奇异点（singularity）——某些姿态下关节速度需求趋于无穷
  - 需要逆运动学（IK）将末端位置解算为关节角度
  - 关节限位和自碰撞在笛卡尔空间中不易表达

### 论文中的对应关系

论文中大多数 VLA 模型默认在 EEF 空间输出动作。例如：
- RT-2: 输出 EEF 位姿变化量
- 一些模型（如 RoboMamba）输出 SE(3) 位姿（即三维旋转 + 平移）

也有模型直接在**关节空间**输出动作：
- **ACT**: 输出目标关节位置（单臂 7-DoF / 双臂 14-DoF，含夹爪开合），通过 ALOHA 遥操作数据学习。ACT 走的是"关节空间 + 动作分块"路线，不依赖逆运动学
- **QUAR-VLA**（四足机器人）、**Humanoid-VLA**（人形机器人）: 全身控制需要在关节层面协调，也输出关节指令

> 注意：ACT 在论文表 1 中标注 **LanguageEncoder: none**——它是纯视觉运动策略（由任务切换而非自然语言驱动），不属于语言条件化的 VLA 严格定义，但常作为 VLA 家族的一员被讨论。

---

## Environment 与 Observation

### Environment（环境）

Environment 是机器人与之交互的外部世界。在 VLA 研究中，分为两类：

#### 真实环境（Real Environment）
- 物理机器人 + 真实场景
- 数据质量高但采集昂贵、不可控
- 代表数据集：Open X-Embodiment、DROID、BridgeData V2

#### 仿真环境（Simulation Environment）
- 论文列出 12 个仿真平台：AI2-THOR、Habitat、Isaac Sim、Gazebo、PyBullet、CoppeliaSim、Webots、Unity ML-Agents、MuJoCo、iGibson、UniSim、SAPIEN
- 数据可大规模生成（百万级轨迹）、可复现
- 存在 sim-to-real gap：物理精度、光照、纹理等方面的差异

仿真环境的关键能力评估维度：
- 传感器模态（RGB/D/力/触觉/IMU/LiDAR）
- 物理引擎精度（接触建模、软体形变、摩擦力）
- 语言指令对接 API
- 多机器人/多智能体支持

### Observation（观测）

Observation 是机器人通过传感器从环境中获得的信息。VLA 模型中常见的观测模态包括：

| 模态 | 内容 | 典型编码方式 |
|------|------|-------------|
| **RGB 图像** | 彩色相机画面 | ViT、CNN → feature tokens |
| **深度图 (Depth)** | 每个像素到相机的距离 | 与 RGB 拼接或独立编码 |
| **体素 (Voxel)** | 3D 空间的占据网格 | 3D CNN / Perceiver（如 PerAct） |
| **点云 (Point Cloud)** | 3D 空间中的点集 | PointNet / Transformer |
| **本体感知 (Proprioception)** | 关节角度、EEF 位姿、夹爪状态 | MLP → proprioceptive tokens |
| **力/力矩 (Force/Torque)** | 接触力、关节力矩 | MLP / 时序编码器 |
| **触觉 (Tactile)** | 指尖触觉阵列 | CNN / Transformer |
| **音频 (Audio)** | 环境声音、语音指令 | 音频编码器 |
| **语言指令 (Language)** | 自然语言任务描述 | LLM tokenizer + embedding |

**Observation 与 State 的区别**：
- Observation 是机器人**感知到**的信息（可能包含噪声、遮挡、部分可观测）
- State 是环境的**真实完整状态**（仿真中可直接获取，现实中不可得）
- VLA 模型通常只接受 observation 作为输入，而不是 ground-truth state

**部分可观测性 (Partial Observability)**：
在真实环境中，机器人永远无法获得完整的环境状态（被遮挡的物体、未知的摩擦力等）。VLA 模型需要通过历史观测序列（引入记忆）来处理这种部分可观测性。

---

## 关键挑战与未来方向

论文第 8 章总结了三大类挑战：

### 架构挑战
1. **Token 化与词汇对齐**：如何将异构输入（图像 patch、文本 token、连续关节角度）统一编码
2. **模态融合**：简单拼接不足以对齐视觉和语言的不同统计特性
3. **跨构型泛化**：固定动作词汇限制了模型迁移到不同机器人
4. **运动平滑性**：预测离散动作 token 往往忽视连续轨迹的质量

### 数据集挑战
1. **任务多样性不足**：大多数数据集专注单一狭窄任务
2. **模态不平衡**：缺乏同步的多模态数据（RGB + 深度 + 力 + 触觉）
3. **标注成本高**：人工标注 6-DoF 位姿和自然语言解释极其耗时
4. **真实性与规模权衡**：真实数据高保真但量少，仿真数据量大但有域差异

### 仿真挑战
1. **物理精度**：基本 Coulomb 摩擦模型和点接触近似无法捕捉真实交互
2. **视觉保真度与吞吐量权衡**：照片级渲染慢，快速渲染有域差异
3. **语言 grounding API 缺乏**：仿真器原生不支持自然语言指令映射
4. **多机器人支持不一致**：各仿真器对不同机器人格式（URDF/SDF）的支持差异大

---

## 关键收获

1. **VLA 不是一个模型，而是一个范式**——它将视觉、语言、动作三大模态统一到端到端可学习的框架中，目标是通用机器人智能体

2. **架构遵循"编码-融合-解码"模式**：独立 encode 视觉/语言/状态 → Transformer 融合 → 动作解码器生成控制信号

3. **Diffusion Policy 是当前主流动作生成方法**：相比离散 token 预测，diffusion 能更好地建模连续动作的多模态分布和平滑轨迹

4. **数据集是瓶颈**：论文的二维 benchmark 揭示了一个关键空白——同时具备高任务复杂度和多模态丰富度的数据集严重缺乏

5. **EEF 空间是 VLA 模型的默认控制空间**：因为它在不同机器人构型间提供自然的抽象，但跨构型泛化仍然是开放问题

6. **Sim-to-Real Gap 是核心挑战**：仿真提供规模，真实环境提供保真度，弥合两者需要物理精度的提升和更好的域自适应方法

7. **VLA 的终极目标**：构建能理解自然语言指令、感知复杂环境并自主执行物理任务的通用机器人智能体
