# 🤖 具身智能学习

> VPX 具身智能小组 · 结构化学习档案  
> Embodied Intelligence Learning Journey

## 关于本项目

这是我在 VPX 具身智能小组的系统学习记录。具身智能（Embodied Intelligence）研究有物理载体的 AI 系统——机器人如何在真实世界中感知、决策、行动并自主学习。

本项目按照小组任务分阶段组织，每个阶段包含阅读笔记、代码实验和任务产出。

## 学习路线

| 阶段 | 目录 | 内容 | 状态 |
|------|------|------|------|
| 🏗️ Phase 1 | `phase1-robot-system/` | 认识机器人系统 — 阅读《Robotics: Modelling, Planning and Control》 | 🟢 已完成 |
| 🧠 Phase 2 | `phase2-algorithm-basics/` | 算法基础 — 深度学习概念 + 论文阅读 + 模型对比实验 | 🟢 已完成 |
| 📖 Phase 3 | `phase3-vla-survey/` | VLA 综述 — 综述论文阅读 + 介绍 PPT | 🟢 已完成（PPT 待补） |
| 🦾 Phase 4 | `phase4-act-reproduction/` | ACT 复现 — ACT 论文 + ALOHA + 仿真环境 | 🟢 已完成 |
| 🧪 Phase 5 | `phase5-vla-reproduction/` | 小规模 VLA 复现 — SmolVLA × LIBERO 训练推理 | 🟢 已完成 |

## 阶段成果速览

| 阶段 | 核心产出 |
|------|---------|
| Phase 1 | 机器人学教材笔记 + 4 问解答 |
| Phase 2 | 3 篇论文笔记 + MLP/CNN/ViT 对比实验（58.3% / 90.1% / 79.0%） |
| Phase 3 | VLA 综述（28 页）精读笔记 + 5 问解答 |
| Phase 4 | ACT × gym-aloha：训练 20K 步、10 个推理视频、复现报告 |
| Phase 5 | SmolVLA 0.5B × LIBERO：LoRA 微调 20K 步（loss 0.111）、20 个推理视频、完整报告 |

## 技术栈

- **语言:** Python（主力）、C++（可能涉及）
- **框架:** PyTorch / LeRobot
- **仿真:** MuJoCo（gym-aloha）/ LIBERO（robosuite）
- **硬件:** ALOHA 机械臂 / RTX 4060 / RTX 5880 Ada

## 目录结构

```
.
├── README.md                   # 本文件
├── CLAUDE.md                   # AI 助手上下文
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
