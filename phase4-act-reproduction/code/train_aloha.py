"""
ACT 训练包装脚本。

解决 lerobot-train 的两个 gym-aloha 兼容问题:
1. gym_aloha 环境只在 import 时注册 —— 提前 import 让评估子进程也能找到
2. lerobot eval 通过 env.call("task_description") 获取任务描述 —— 给 AlohaEnv 补上属性

用法: python train_aloha.py <与 lerobot-train 相同的参数>
"""

import sys

import gym_aloha  # noqa: F401  # 注册 gymnasium 环境
from gym_aloha.env import AlohaEnv


# lerobot 的 eval 代码调用 env.call("task_description")，gym-aloha 0.1.4 没有该属性
def _get_task_description(self):
    return self.task if hasattr(self, "task") else "aloha_insertion"


AlohaEnv.task_description = property(_get_task_description)

from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
