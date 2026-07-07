# 具身智能学习项目初始化 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建具身智能学习项目的完整目录骨架、CLAUDE.md、README.md 以及 5 个阶段的模板文件。

**Architecture:** 纯文件初始化任务。按阶段平铺的目录结构，每个阶段内部含 `notes/`、`code/`、`output/` 等子目录。所有内容为 Markdown 格式，中文为主。

**Tech Stack:** Markdown, Git, Bash

## Global Constraints

- 所有文档和笔记使用中文
- 数学公式在 .md 文件中必须用 `$$` 格式包裹（LaTeX）
- 代码注释中英皆可
- 项目位于 `/home/wszxh/claude_projects/embodied-intelligence-learning/`
- Git 仓库已初始化，设计文档已提交

---

### Task 1: 创建完整目录骨架

**Files:**
- Create: 所有目录（见下方列表）

**Interfaces:**
- Consumes: 无
- Produces: 完整的空目录结构，供后续任务写入文件

- [ ] **Step 1: 创建所有目录**

```bash
cd /home/wszxh/claude_projects/embodied-intelligence-learning
mkdir -p \
  phase1-robot-system/notes \
  phase2-algorithm-basics/notes/paper-reading \
  phase2-algorithm-basics/code/model-comparison \
  phase3-vla-survey/notes \
  phase3-vla-survey/output \
  phase4-act-reproduction/notes \
  phase4-act-reproduction/code \
  phase5-vla-reproduction/notes \
  phase5-vla-reproduction/code \
  phase5-vla-reproduction/output
```

- [ ] **Step 2: 验证目录结构**

```bash
find . -type d | sort
```

Expected: 列出所有创建的目录，与设计文档中的结构一致。

- [ ] **Step 3: 为每个空目录添加 .gitkeep**

```bash
find . -type d -empty -not -path './.git/*' -exec touch {}/.gitkeep \;
```

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "feat: 创建项目目录骨架

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: 编写 CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

**Interfaces:**
- Consumes: 无
- Produces: CLAUDE.md — AI 助手每次会话自动加载的项目上下文

- [ ] **Step 1: 写入 CLAUDE.md**

写入文件 `CLAUDE.md`，内容如下：

```markdown
# 具身智能学习项目

## 项目背景

这是 VPX 具身智能小组的结构化学习档案项目。具身智能（Embodied Intelligence）是指有物理载体（如机器人）的 AI，能利用感知、决策和物理交互能力在真实世界中执行任务并自主学习进化。

## 学习路线

| 阶段 | 名称 | 核心任务 |
|------|------|----------|
| Phase 1 | 认识机器人系统 | 阅读《Robotics: Modelling, Planning and Control》，建立对机器人闭环系统的认知 |
| Phase 2 | 算法基础 | 深度学习基础概念 + AlexNet/Transformer/ViT 论文阅读 + MLP/CNN/ViT 对比实验 |
| Phase 3 | VLA 综述 | 阅读 VLA 综述论文，制作介绍 PPT，理解动作空间/EEF/environment 等概念 |
| Phase 4 | ACT 复现 | 阅读 ACT 论文，了解 ALOHA 设备，在仿真环境跑通 ACT |
| Phase 5 | Mole-VLA 复现 | Mole-VLA 论文复现，在 LEBERO 仿真环境完成训练推理 |

## 项目结构

```
phase1-robot-system/         # 阶段一：认识机器人系统
phase2-algorithm-basics/     # 阶段二：算法基础
phase3-vla-survey/           # 阶段三：VLA 综述
phase4-act-reproduction/     # 阶段四：ACT 复现
phase5-vla-reproduction/     # 阶段五：Mole-VLA 复现
```

每个阶段内部包含：
- `notes/` — 阅读笔记、论文笔记
- `code/` — 实验代码
- `questions.md` 或 `report.md` — 任务回答或报告
- `output/` — 产出物（PPT/视频等）

## 关键资源

- 小组 GitHub 资料库: https://github.com/TianxingChen/Embodied-AI-Guide
- Phase 3 综述: https://arxiv.org/abs/2507.10672
- ACT 论文: https://arxiv.org/abs/2304.13705
- Mole-VLA 论文: https://arxiv.org/abs/2503.20384
- 深度学习入门: B站《动手学深度学习》系列

## AI 协作约定

- 语言：笔记和文档使用简体中文，代码注释中英皆可
- 数学：.md 文件中数学公式使用 `$$` LaTeX 格式
- 习题解答：必须完整，不可省略步骤
- 笔记风格：从第一性原理出发，清晰解释概念
```

- [ ] **Step 2: 提交**

```bash
git add CLAUDE.md
git commit -m "feat: 添加 CLAUDE.md 项目上下文文件

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: 编写 README.md

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: 无
- Produces: README.md — 项目主页，展示给人类阅读

- [ ] **Step 1: 写入 README.md**

写入文件 `README.md`，内容如下：

```markdown
# 🤖 具身智能学习

> VPX 具身智能小组 · 结构化学习档案  
> Embodied Intelligence Learning Journey

## 关于本项目

这是我在 VPX 具身智能小组的系统学习记录。具身智能（Embodied Intelligence）研究有物理载体的 AI 系统——机器人如何在真实世界中感知、决策、行动并自主学习。

本项目按照小组任务分阶段组织，每个阶段包含阅读笔记、代码实验和任务产出。

## 学习路线

| 阶段 | 目录 | 内容 | 状态 |
|------|------|------|------|
| 🏗️ Phase 1 | `phase1-robot-system/` | 认识机器人系统 — 阅读《Robotics: Modelling, Planning and Control》 | 🔴 待开始 |
| 🧠 Phase 2 | `phase2-algorithm-basics/` | 算法基础 — 深度学习概念 + 论文阅读 + 模型对比实验 | 🔴 待开始 |
| 📖 Phase 3 | `phase3-vla-survey/` | VLA 综述 — 综述论文阅读 + 介绍 PPT | 🔴 待开始 |
| 🦾 Phase 4 | `phase4-act-reproduction/` | ACT 复现 — ACT 论文 + ALOHA + 仿真环境 | 🔴 待开始 |
| 🧪 Phase 5 | `phase5-vla-reproduction/` | Mole-VLA 复现 — 在新仿真环境复现 VLA 模型 | 🔴 待开始 |

## 技术栈

- **语言:** Python（主力）、C++（可能涉及）
- **框架:** PyTorch
- **仿真:** MuJoCo / Isaac Gym / LEBERO
- **硬件:** ALOHA 机械臂

## 目录结构

```
.
├── README.md                   # 本文件
├── CLAUDE.md                   # AI 助手上下午
├── phase1-robot-system/        # 阶段一
│   ├── notes/                  # 阅读笔记
│   └── questions.md            # 任务问题回答
├── phase2-algorithm-basics/    # 阶段二
│   ├── notes/                  # 概念笔记 + 论文阅读
│   ├── code/                   # 实验代码
│   └── questions.md
├── phase3-vla-survey/          # 阶段三
│   ├── notes/
│   ├── output/                 # PPT 产出
│   └── questions.md
├── phase4-act-reproduction/    # 阶段四
│   ├── notes/
│   ├── code/
│   └── report.md
├── phase5-vla-reproduction/    # 阶段五
│   ├── notes/
│   ├── code/
│   ├── output/                 # 推理视频
│   └── report.md
└── docs/                       # 项目文档
    └── superpowers/
        ├── specs/              # 设计文档
        └── plans/              # 实施计划
```

## 快速开始

```bash
# 克隆项目
git clone <repo-url>
cd embodied-intelligence-learning

# 从 Phase 1 开始
cd phase1-robot-system
```

## 参考资料

- [Embodied-AI-Guide](https://github.com/TianxingChen/Embodied-AI-Guide) — 小组资料库
- [动手学深度学习 (B站)](https://www.bilibili.com/video/BV15UREYsEN8/)
- [VLA 综述 (arXiv:2507.10672)](https://arxiv.org/abs/2507.10672)
- [ACT (arXiv:2304.13705)](https://arxiv.org/abs/2304.13705)
- [Mole-VLA (arXiv:2503.20384)](https://arxiv.org/abs/2503.20384)
```

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "feat: 添加 README.md 项目主页

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: 编写 Phase 1 模板文件

**Files:**
- Create: `phase1-robot-system/questions.md`
- Create: `phase1-robot-system/notes/robotics-modelling-reading.md`

**Interfaces:**
- Consumes: 无
- Produces: Phase 1 的两个模板文件，包含任务问题和笔记模板

- [ ] **Step 1: 写入 questions.md**

写入文件 `phase1-robot-system/questions.md`：

```markdown
# Phase 1: 认识机器人系统 — 问题回答

> 阅读材料: 《Robotics: Modelling, Planning and Control》(Sciavicco) 节选

## 问题 1: 什么是具身智能特有的，深度学习中不会考虑的？

（待回答）

## 问题 2: 什么是 policy，如何设计一个 policy？

（待回答）

## 问题 3: "具身"体现在什么地方？

（待回答）

## 问题 4: 这些东西将来怎么影响你训练一个能抓东西的机器人？

（待回答）
```

- [ ] **Step 2: 写入阅读笔记模板**

写入文件 `phase1-robot-system/notes/robotics-modelling-reading.md`：

```markdown
# 《Robotics: Modelling, Planning and Control》阅读笔记

## 核心概念速查

| 术语 | 解释 |
|------|------|
| （待补充） | |

## 机器人系统的闭环结构

（待补充）

## 关键收获

（待补充）

## 与具身智能的关联思考

（待补充）
```

- [ ] **Step 3: 提交**

```bash
git add phase1-robot-system/
git commit -m "feat: 添加 Phase 1 模板文件

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: 编写 Phase 2 模板文件

**Files:**
- Create: `phase2-algorithm-basics/questions.md`
- Create: `phase2-algorithm-basics/notes/deep-learning-basics.md`
- Create: `phase2-algorithm-basics/notes/paper-reading/alexnet.md`
- Create: `phase2-algorithm-basics/notes/paper-reading/transformer.md`
- Create: `phase2-algorithm-basics/notes/paper-reading/vit.md`

**Interfaces:**
- Consumes: 无
- Produces: Phase 2 的所有模板文件

- [ ] **Step 1: 写入 questions.md**

写入文件 `phase2-algorithm-basics/questions.md`：

```markdown
# Phase 2: 算法基础 — 问题回答

> 阅读材料: AlexNet, Transformer, ViT 论文 + B站《动手学深度学习》

## 问题 1: 什么是深度学习，什么是机器学习？

（待回答）

## 问题 2: 前向反向传播为什么能够发挥作用？

（待回答）

## 问题 3: 损失函数是如何影响模型的训练的？

（待回答）

## 问题 4: 为什么会梯度消失/爆炸？如何解决？

（待回答）

## 问题 5: 机器学习的深度学习算法如何设计和训练？

不需要设计具体的算法，只需提供思路。

（待回答）
```

- [ ] **Step 2: 写入深度学习基础笔记**

写入文件 `phase2-algorithm-basics/notes/deep-learning-basics.md`：

```markdown
# 深度学习基础概念

## 机器学习 vs 深度学习

（待补充）

## 前向传播与反向传播

（待补充）

## 损失函数

### 常见损失函数

| 损失函数 | 公式 | 适用场景 |
|----------|------|----------|
| MSE | $$ \frac{1}{n}\sum(y_i - \hat{y}_i)^2 $$ | 回归 |
| Cross-Entropy | $$ -\sum y_i \log(\hat{y}_i) $$ | 分类 |

## 梯度消失与梯度爆炸

（待补充）

## 深度学习算法设计思路

（待补充）
```

- [ ] **Step 3: 写入论文阅读模板**

写入文件 `phase2-algorithm-basics/notes/paper-reading/alexnet.md`：

```markdown
# AlexNet 论文阅读笔记

> 论文: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)  
> Krizhevsky, Sutskever, Hinton · NIPS 2012

## 核心贡献

（待补充）

## 网络结构

（待补充）

## 关键创新点

（待补充）

## 对我的启发

（待补充）
```

写入文件 `phase2-algorithm-basics/notes/paper-reading/transformer.md`：

```markdown
# Transformer 论文阅读笔记

> 论文: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
> Vaswani et al. · NeurIPS 2017

## 核心贡献

（待补充）

## 架构概览

（待补充）

### Self-Attention 机制

（待补充）

### Multi-Head Attention

（待补充）

### Positional Encoding

（待补充）

## 与具身智能的关联

（待补充）
```

写入文件 `phase2-algorithm-basics/notes/paper-reading/vit.md`：

```markdown
# ViT 论文阅读笔记

> 论文: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)  
> Dosovitskiy et al. · ICLR 2021

## 核心贡献

（待补充）

## 架构概览

（待补充）

### Patch Embedding

（待补充）

### 与 CNN 的对比

（待补充）

## 在具身智能中的应用前景

（待补充）
```

- [ ] **Step 4: 提交**

```bash
git add phase2-algorithm-basics/
git commit -m "feat: 添加 Phase 2 模板文件

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: 编写 Phase 3 模板文件

**Files:**
- Create: `phase3-vla-survey/questions.md`
- Create: `phase3-vla-survey/notes/vla-survey-reading.md`

**Interfaces:**
- Consumes: 无
- Produces: Phase 3 的两个模板文件

- [ ] **Step 1: 写入 questions.md**

写入文件 `phase3-vla-survey/questions.md`：

```markdown
# Phase 3: VLA 综述 — 问题回答

> 阅读材料: VLA 综述论文 (arXiv:2507.10672)

## 问题 1: 什么是 VLA（Vision-Language-Action）？

（待回答）

## 问题 2: VLA 是如何工作的？

（待回答）

## 问题 3: 什么是动作空间（Action Space）？

（待回答）

## 问题 4: Joint Point 与 EEF 的区别是什么？

（待回答）

## 问题 5: 什么是 environment 和 observation？

（待回答）
```

- [ ] **Step 2: 写入综述阅读笔记**

写入文件 `phase3-vla-survey/notes/vla-survey-reading.md`：

```markdown
# VLA 综述论文阅读笔记

> 论文: (arXiv:2507.10672)  
> 链接: https://arxiv.org/abs/2507.10672

## 论文概述

（待补充）

## VLA 模型架构

（待补充）

## 动作空间

### Joint Point（关节空间）

（待补充）

### EEF（末端执行器空间 / End-Effector Frame）

（待补充）

## Environment 与 Observation

（待补充）

## 关键收获

（待补充）
```

- [ ] **Step 3: 提交**

```bash
git add phase3-vla-survey/
git commit -m "feat: 添加 Phase 3 模板文件

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: 编写 Phase 4 模板文件

**Files:**
- Create: `phase4-act-reproduction/report.md`
- Create: `phase4-act-reproduction/notes/act-paper-reading.md`
- Create: `phase4-act-reproduction/notes/aloha-guide.md`

**Interfaces:**
- Consumes: 无
- Produces: Phase 4 的三个模板文件

- [ ] **Step 1: 写入 report.md**

写入文件 `phase4-act-reproduction/report.md`：

```markdown
# Phase 4: ACT 复现报告

## 实验环境

- 仿真环境: （待填写）
- 硬件: ALOHA
- 框架: （待填写）

## 复现步骤

1. （待补充）

## 实验结果

（待补充）

## 遇到的问题与解决方案

（待补充）

## 总结

（待补充）
```

- [ ] **Step 2: 写入论文阅读笔记**

写入文件 `phase4-act-reproduction/notes/act-paper-reading.md`：

```markdown
# ACT 论文阅读笔记

> 论文: [Action Chunking with Transformers](https://arxiv.org/abs/2304.13705)  
> Zhao et al. · 2023

## 核心贡献

（待补充）

## 方法概述

### Action Chunking 是什么

（待补充）

### 为什么需要 Action Chunking

（待补充）

## 网络结构

（待补充）

## 实验结果

（待补充）

## 与 ALOHA 硬件的关系

（待补充）
```

写入文件 `phase4-act-reproduction/notes/aloha-guide.md`：

```markdown
# ALOHA 操作指南

> 设备: 松灵 ALOHA

## 硬件概览

（待补充）

## 基本操作

（待补充）

## 常见问题

（待补充）
```

- [ ] **Step 3: 提交**

```bash
git add phase4-act-reproduction/
git commit -m "feat: 添加 Phase 4 模板文件

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: 编写 Phase 5 模板文件

**Files:**
- Create: `phase5-vla-reproduction/report.md`
- Create: `phase5-vla-reproduction/notes/mole-vla-paper-reading.md`

**Interfaces:**
- Consumes: 无
- Produces: Phase 5 的两个模板文件

- [ ] **Step 1: 写入 report.md**

写入文件 `phase5-vla-reproduction/report.md`：

```markdown
# Phase 5: Mole-VLA 复现报告

## 实验环境

- 仿真环境: LEBERO
- 模型: Mole-VLA
- 框架: （待填写）

## 复现步骤

1. （待补充）

## 模型与仿真环境的交互

（待补充）

## 推理结果

（待补充）

## 视频 Demo

（待上传至 `output/` 目录）

## 遇到的问题与解决方案

（待补充）

## 总结

（待补充）
```

- [ ] **Step 2: 写入论文阅读笔记**

写入文件 `phase5-vla-reproduction/notes/mole-vla-paper-reading.md`：

```markdown
# Mole-VLA 论文阅读笔记

> 论文: (arXiv:2503.20384)  
> 链接: https://arxiv.org/abs/2503.20384

## 核心贡献

（待补充）

## 方法概述

（待补充）

## 与 ACT 的对比

（待补充）

## 在 LEBERO 上的适配

（待补充）

## 关键收获

（待补充）
```

- [ ] **Step 3: 提交**

```bash
git add phase5-vla-reproduction/
git commit -m "feat: 添加 Phase 5 模板文件

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: 添加 .gitignore

**Files:**
- Create: `.gitignore`

**Interfaces:**
- Consumes: 无
- Produces: `.gitignore` 忽略常见的 Python/IDE/OS 无关文件

- [ ] **Step 1: 写入 .gitignore**

写入文件 `.gitignore`：

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
dist/
build/
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local

# Model checkpoints
*.ckpt
*.pth
*.pt
checkpoints/

# Simulation outputs
*.mp4
*.gif
*.log

# Large files (datasets, models)
data/
models/
```

- [ ] **Step 2: 提交**

```bash
git add .gitignore
git commit -m "feat: 添加 .gitignore

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: 清理 .gitkeep 并最终验证

**Files:**
- Modify: 删除所有 `.gitkeep` 文件（现在各目录已有实际内容或将被填充）

**Interfaces:**
- Consumes: Tasks 1-9 产物
- Produces: 干净的项目结构

- [ ] **Step 1: 删除仍有实际文件目录中的 .gitkeep**

```bash
cd /home/wszxh/claude_projects/embodied-intelligence-learning
find . -name ".gitkeep" -delete
```

- [ ] **Step 2: 验证最终结构**

```bash
tree -a -I '.git'
```

Expected: 目录结构与设计文档一致，所有模板文件就位，无 `.gitkeep` 残留。

- [ ] **Step 3: 最终提交**

```bash
git add -A
git commit -m "chore: 清理 .gitkeep 文件

Co-Authored-By: Claude <noreply@anthropic.com>"
```
