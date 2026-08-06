"""
MoLe-VLA / CogACT 在 LIBERO 仿真环境上的推理评估脚本（Phase 5）

用法:
    python lib_libero_eval.py --ckpt CogACT/CogACT-Base --benchmark libero_spatial --task-idx 0 --num-episodes 5

功能:
    1. 加载 CogACT / MoLe-VLA 权重（HF 或本地路径）
    2. 在 LIBERO 任务上跑 rollout（观测图像 + 语言指令 -> 7-DoF 动作）
    3. 录制每个 episode 的推理视频到 output/ 目录

动作映射:
    CogACT 输出: world_vector(3) + rot_axangle(3) + gripper(1)
    LIBERO 动作: [dx, dy, dz, ax, ay, az, gripper]  (robosuite OSC_POSE)
"""

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np

# ============ Windows 兼容: stub tensorflow 训练链 ============
# 推理只用 CogACT (PyTorch)，tensorflow/tfds/dlimp 是 openvla 训练链的依赖。
# tensorflow_datasets 的 core/shuffle.py 会 import 'resource' —— Unix-only 标准库，
# Windows 上不存在，真包必然 ImportError。stub 掉即可（推理不调用它们）。
def _stub_module(name: str):
    m = types.ModuleType(name)
    m.__spec__ = types.SimpleNamespace(name=name)  # base_llm.py 会检查 __spec__
    sys.modules[name] = m


for _mod in ("tensorflow", "tensorflow_datasets", "dlimp"):
    _stub_module(_mod)

# ============ 权重加载（MoLe-VLA 机制）============
# MoLe-VLA 仓库要求 TRAIN_ROUTE 环境变量来选择 MoLe 模型加载路径
os.environ.setdefault("TRAIN_ROUTE", "FALSE")  # TRUE=MoLe 模型, FALSE=标准 CogACT
sys.path.insert(0, str(Path(__file__).parent / "MoLe-VLA-Pytorch"))

import torch  # noqa: E402
from sim_cogact.cogact_policy import CogACTInference  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="MoLe-VLA on LIBERO")
    parser.add_argument("--ckpt", type=str, default="CogACT/CogACT-Base",
                        help="模型路径: CogACT/CogACT-Small|Base|Large 或本地路径")
    parser.add_argument("--action-model", type=str, default="DiT-B",
                        help="DiT 类型: DiT-S / DiT-B / DiT-L，与权重匹配")
    parser.add_argument("--benchmark", type=str, default="libero_spatial",
                        help="LIBERO benchmark: libero_spatial / libero_object / libero_goal / libero_10")
    parser.add_argument("--task-idx", type=int, default=0, help="任务索引")
    parser.add_argument("--num-episodes", type=int, default=5, help="评估 episode 数")
    parser.add_argument("--max-steps", type=int, default=300, help="每个 episode 最大步数")
    parser.add_argument("--action-scale", type=float, default=0.5,
                        help="动作缩放（CogACT 训练数据与 LIBERO 尺度不匹配时的调参）")
    parser.add_argument("--video-dir", type=str, default="../output", help="视频输出目录")
    parser.add_argument("--policy-setup", type=str, default="google_robot",
                        help="CogACT 策略配置: google_robot / widowx_bridge")
    return parser.parse_args()


def make_libero_env(benchmark_name: str, task_idx: int, use_offscreen: bool = True):
    """创建 LIBERO 环境。优先 OffScreenRenderEnv（无窗口），失败则回退 RobosuiteEnv。"""
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv, RobosuiteEnv

    bench = benchmark.get_benchmark(benchmark_name)()
    task = bench.get_task(task_idx)
    env_cls = OffScreenRenderEnv if use_offscreen else RobosuiteEnv
    try:
        env = env_cls(task)
        print(f"[LIBERO] 环境创建成功: {env_cls.__name__} | 任务: {task.name}")
    except Exception as e:
        print(f"[LIBERO] {env_cls.__name__} 失败 ({e})，回退 RobosuiteEnv")
        env = RobosuiteEnv(task)
    return env, task


def render_frame(env, camera_name="agentview", height=256, width=256):
    """从 LIBERO 渲染一帧（robosuite 接口）"""
    try:
        return env.sim.render(camera_name=camera_name, height=height, width=width)
    except Exception:
        return env.env.sim.render(camera_name=camera_name, height=height, width=width)


def run_episode(env, task, policy, max_steps, action_scale):
    """跑一个 episode，返回 (是否成功, 视频帧列表)"""
    obs = env.reset()
    task_desc = task.language_instruction
    print(f"  指令: {task_desc}")

    policy.reset(task_desc)
    video_frames = []

    for step in range(max_steps):
        # 1. 取观测图像（agentview 相机，CogACT 期望 224x224）
        img = obs["agentview_image"]  # (H, W, 3) uint8
        if img.shape[0] != 224:
            import cv2
            img = cv2.resize(img, (224, 224))

        # 2. VLA 推理 -> 7-DoF 动作
        raw_action, action = policy.step(image=img, task_description=task_desc)

        # 3. 映射动作: world_vector + rot_axangle + gripper
        a = np.concatenate([
            action["world_vector"].flatten(),
            action["rot_axangle"].flatten(),
            action["gripper"].flatten(),
        ]).astype(np.float64) * action_scale
        a = np.clip(a, -1.0, 1.0)

        # 4. 执行 + 渲染
        obs, reward, done, info = env.step(a)
        frame = render_frame(env)
        if frame is not None:
            video_frames.append(frame)

        if done:
            break

    success = bool(info.get("success", False))
    return success, video_frames


def main():
    args = parse_args()

    # 创建 LIBERO 环境
    env, task = make_libero_env(args.benchmark, args.task_idx)

    # 加载 VLA 模型（MoLe-VLA / CogACT）
    print(f"[模型] 加载 {args.ckpt} (action_model={args.action_model}) ...")
    policy = CogACTInference(
        saved_model_path=args.ckpt,
        policy_setup=args.policy_setup,
        action_model_type=args.action_model,
        action_scale=args.action_scale,
        use_bf16=True,
        future_action_window_size=15,
    )
    print("[模型] 加载完成")

    # 视频输出（每个 episode 一个视频）
    os.makedirs(args.video_dir, exist_ok=True)
    import imageio.v2 as iio

    # 评估 N 个 episodes
    success_count = 0
    for ep in range(args.num_episodes):
        print(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")
        ok, frames = run_episode(env, task, policy, args.max_steps, args.action_scale)
        success_count += ok
        print(f"  -> {'✅ 成功' if ok else '❌ 失败'} ({len(frames)} 帧)")

        # 保存该 episode 的视频（imageio: append_data 逐帧写入）
        if frames:
            video_path = os.path.join(
                args.video_dir, f"{args.benchmark}_task{args.task_idx}_ep{ep}.mp4")
            writer = iio.get_writer(video_path, fps=10)
            for f in frames:
                writer.append_data(f)
            writer.close()
            print(f"  [视频] {video_path}")

    print(f"\n[结果] 成功率: {success_count}/{args.num_episodes} = {success_count / args.num_episodes:.0%}")
    print(f"[结果] 视频目录: {os.path.abspath(args.video_dir)}")


if __name__ == "__main__":
    main()
