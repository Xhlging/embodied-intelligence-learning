# 具身智能学习项目 — 设计文档

**日期:** 2026-07-07  
**状态:** 已确认  
**作者:** VPX 具身智能小组  

## 项目概述

这是一个**结构化学习档案项目**，用于系统记录 VPX 具身智能小组的分阶段学习过程。项目以笔记 + 代码一体化的方式组织，每个阶段一个顶层目录，内部按 `notes/`、`code/`、`output/` 等子目录归类。目标是让整个学习路径可追溯、可复现、可持续维护。

## 目标用户

- 作为学习者的自己（主要用户）
- VPX 小组的其他成员（可能的协作者）
- 将来回顾学习历程的自己

## 完整阶段路线图

| 阶段 | 名称 | 核心产出 |
|------|------|----------|
| Phase 1 | 认识机器人系统 | 《Robotics: Modelling, Planning and Control》阅读笔记 + 问题回答 |
| Phase 2 | 算法基础 | 深度学习概念笔记 + AlexNet/Transformer/ViT 论文阅读 + MLP/CNN/ViT 对比实验 |
| Phase 3 | VLA 综述 | VLA 综述论文阅读 + 介绍 PPT + 核心概念问答 |
| Phase 4 | ACT 复现 | ACT 论文阅读 + ALOHA 设备了解 + 仿真环境跑通 + 复现报告 |
| Phase 5 | Mole-VLA 复现 | Mole-VLA 论文阅读 + LEBERO 仿真环境训练推理 + 视频 demo |

## 目录结构

```
embodied-intelligence-learning/
├── README.md                              # 项目概述、小组信息、完整学习路线图
├── CLAUDE.md                              # 项目级 AI 助手指令（加载上下文）
├── phase1-robot-system/
│   ├── notes/
│   │   └── robotics-modelling-reading.md
│   └── questions.md
├── phase2-algorithm-basics/
│   ├── notes/
│   │   ├── deep-learning-basics.md
│   │   └── paper-reading/
│   │       ├── alexnet.md
│   │       ├── transformer.md
│   │       └── vit.md
│   ├── code/
│   │   └── model-comparison/
│   └── questions.md
├── phase3-vla-survey/
│   ├── notes/
│   │   └── vla-survey-reading.md
│   ├── output/
│   └── questions.md
├── phase4-act-reproduction/
│   ├── notes/
│   │   ├── act-paper-reading.md
│   │   └── aloha-guide.md
│   ├── code/
│   └── report.md
└── phase5-vla-reproduction/
    ├── notes/
    │   └── mole-vla-paper-reading.md
    ├── code/
    ├── report.md
    └── output/
```

## 设计原则

1. **阶段递进** — 从认知 → 基础 → 综述 → 经典复现 → 独立迁移，环环相扣
2. **笔记与代码共存** — 同一阶段的笔记和代码放在一起，上下文不丢失
3. **最小化但完整** — 每个阶段只有必要的目录，不预设用不到的结构
4. **CLAUDE.md 驱动** — 项目根目录的 CLAUDE.md 让 AI 助手每次都能加载完整上下文
5. **中文为首选语言** — 笔记和文档使用中文，代码注释中英皆可

## 技术栈（预期）

- **编程语言:** Python（主力）、可能涉及 C++
- **深度学习框架:** PyTorch
- **仿真环境:** MuJoCo / Issac Gym / LEBERO
- **硬件相关:** ALOHA 机械臂
- **文档:** Markdown
- **版本控制:** Git
