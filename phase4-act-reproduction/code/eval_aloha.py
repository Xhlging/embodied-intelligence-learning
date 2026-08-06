"""
ACT 独立评估包装脚本: 完整评估并渲染全部 n_episodes 个视频。

与 train_aloha.py 相同的两个兼容补丁:
1. 提前 import gym_aloha 注册 gymnasium 环境
2. 给 AlohaEnv 补 task_description 属性

用法: python eval_aloha.py --config_path <checkpoint>/pretrained_model [其他 lerobot-eval 参数]
"""

import sys

import gym_aloha  # noqa: F401
from gym_aloha.env import AlohaEnv


def _get_task_description(self):
    return self.task if hasattr(self, "task") else "aloha_insertion"


AlohaEnv.task_description = property(_get_task_description)

from lerobot.scripts.lerobot_eval import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
