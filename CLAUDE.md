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
