# 项目状态快照（SessionStart 自动注入）

> 本文件由 `.claude/settings.json` 的 SessionStart hook 自动加载，确保上下文压缩/新会话不丢失关键状态。
> 手动维护：项目有重大变化时更新本文件。

## 项目定位

- **VPX 具身智能小组**结构化学习档案（Level 2 五个阶段任务）
- 路径：`/home/wszxh/claude_projects/AI与深度学习/embodied-intelligence-learning`
- GitHub：`Xhlging/embodied-intelligence-learning`（public）
- 笔记/文档用简体中文；.md 公式行内 `$...$`、独立行 `$$...$$`

## 阶段进度（全部完成）

| 阶段 | 状态 | 核心产出 |
|------|:--:|---------|
| Phase 1 认识机器人系统 | ✅ | 教材笔记 + 4 问 |
| Phase 2 算法基础 | ✅ | 3 篇论文笔记 + MLP/CNN/ViT 实验（58.3%/90.1%/79.0%） |
| Phase 3 VLA 综述 | ✅ | 28 页论文精读 + 5 问（**PPT 待补**） |
| Phase 4 ACT 复现 | ✅ | gym-aloha 训练 20K 步 + 10 视频 + report.md |
| Phase 5 小规模 VLA | ✅ | SmolVLA 0.5B × LIBERO LoRA 训练 + 20 视频 + report.md |

## 协作模式（重要）

- **用户物理执行命令**（图书馆机器 WSL2 + RTX 5880 Ada 48GB），我提供命令与排错
- **禁止编造依赖版本**：版本号必须来自官方文件（environment.yml/requirements）或实测输出
- 排错方法论：让用户贴完整报错 → 基于报错实证定位 → 不猜

## Phase 4/5 关键实证（防重复踩坑）

**Phase 4（ACT × gym-aloha）**：
- `train_aloha.py` 包装脚本解决 3 个兼容问题（gym_aloha 预注册、task_description 补丁、SyncVectorEnv）
- 环境：gym-aloha 0.1.4 + LeRobot 0.6.1 + mujoco 3.11 + torch 2.11

**Phase 5（SmolVLA × LIBERO）**：
- 图书馆 lerobot **0.4.4**（非本机 0.6.1！参数名有差异）
- **`--policy.load_vlm_weights true` 是关键**（默认 False = 随机初始化 VLM，白训）
- eval 必须 `--eval.batch_size 1 --eval.n_episodes 1`（默认 50 并行环境会卡死）
- 缺 ffmpeg 训练直接报错
- 0.4.4 不支持本地 `pretrained_path`，必须 HF repo id
- 渲染需 `libosmesa6-dev libegl1 libgl1`
- 训练实测：9.5 step/s（batch 4），20K 步 ≈ 48 分钟，loss → 0.111
- 结果：成功率 0%（视频显示有完成任务趋势，精度不足）

## 遗留项

1. **Phase 3 PPT**（用户暂缓，可随时补）
2. Phase 5 成功率提升（可续训 50-100K 步 / LoRA r16）
3. 本地推理已放弃（0.4.4 checkpoint adapter-only 格式与本地 0.6.1 不兼容）

## 文档索引

- Phase 5 操作手册：`phase5-vla-reproduction/README_PHASE5.md`
- Phase 5 报告：`phase5-vla-reproduction/report.md`
- Phase 4 报告：`phase4-act-reproduction/report.md`
- 各阶段笔记：`phaseN-*/notes/`
