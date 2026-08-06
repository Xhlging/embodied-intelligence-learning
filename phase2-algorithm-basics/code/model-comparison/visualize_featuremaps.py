"""
更直观的感受训练作用: 同一张图经过"训练前"和"训练后"的卷积层
输出 64 个 feature map —— 随机噪声 vs 清晰的边缘/纹理检测

用法: python visualize_training_effect.py
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from train import SimpleCNN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CIFAR10_CLASSES = ['飞机', '汽车', '鸟', '猫', '鹿',
                   '狗', '青蛙', '马', '船', '卡车']


def main():
    import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 取一张测试图片
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    img, label = test_set[99]  # 固定一张图
    img_batch = img.unsqueeze(0)  # (1, 3, 32, 32)

    # 训练前: 随机权重
    cnn_before = SimpleCNN()
    # 训练后: 加载 checkpoint
    cnn_after = SimpleCNN()
    state = torch.load('checkpoints/cnn_best.pth', map_location='cpu')
    cnn_after.load_state_dict(state)

    # 只跑 Conv2d（绕过 BatchNorm + ReLU，保留正负值）
    conv_before = cnn_before.conv1[0]   # nn.Conv2d(3, 64, 3, padding=1)
    conv_after  = cnn_after.conv1[0]

    with torch.no_grad():
        feat_before = conv_before(img_batch)  # (1, 64, 32, 32)
        feat_after  = conv_after(img_batch)   # (1, 64, 32, 32)

    feat_before = feat_before.squeeze(0)  # (64, 32, 32)
    feat_after  = feat_after.squeeze(0)

    # 反归一化原图用于显示
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    show_img = (img * std + mean).clamp(0, 1)

    # 画图: 两行, 每行 16 个 feature map + 1 个原图
    n_show = 16  # 展示前 16 个通道

    fig = plt.figure(figsize=(18, 6))

    # ---- 第一行: 训练前 ----
    # 原图
    ax = plt.subplot(2, n_show + 2, 1)
    ax.imshow(show_img.permute(1, 2, 0))
    ax.set_title('原图\n' + CIFAR10_CLASSES[label], fontsize=9, color='white', fontweight='bold',
                 backgroundcolor='#333', y=1.0)
    ax.axis('off')
    # 箭头
    ax2 = plt.subplot(2, n_show + 2, 2)
    ax2.text(0.5, 0.5, '→\n随机\n卷积核', fontsize=7, ha='center', va='center', color='gray')
    ax2.axis('off')

    for i in range(n_show):
        ax = plt.subplot(2, n_show + 2, 3 + i)
        fm = feat_before[i]
        vmax = fm.abs().quantile(0.95).item()
        ax.imshow(fm, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('训练前\n(随机)', fontsize=11, fontweight='bold', color='crimson')

    # ---- 第二行: 训练后 ----
    ax = plt.subplot(2, n_show + 2, n_show + 3)
    ax.imshow(show_img.permute(1, 2, 0))
    ax.axis('off')
    ax2 = plt.subplot(2, n_show + 2, n_show + 4)
    ax2.text(0.5, 0.5, '→\n训练后\n卷积核', fontsize=7, ha='center', va='center', color='gray')
    ax2.axis('off')

    # 用同一个 vmax 范围，便于对比两行的"强度"
    all_vals = torch.cat([feat_before[:n_show].flatten(), feat_after[:n_show].flatten()])
    global_max = all_vals.abs().quantile(0.95).item()

    for i in range(n_show):
        ax = plt.subplot(2, n_show + 2, n_show + 5 + i)
        fm = feat_after[i]
        ax.imshow(fm, cmap='RdBu_r', vmin=-global_max, vmax=global_max)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('训练后', fontsize=11, fontweight='bold', color='green')

    fig.suptitle(
        '同一张图经过 CNN 第一层卷积（Conv2d）后的 16 个特征图\n'
        '上排=随机卷积核(训练前), 下排=训练好的卷积核\n'
        '🔴红色=正激活(该模式匹配成功)  🔵蓝色=负激活(该模式被抑制)  ⬜白色=不激活',
        fontsize=13, y=0.98
    )
    plt.tight_layout()
    plt.savefig('results/training_effect_featuremaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/training_effect_featuremaps.png")


if __name__ == '__main__':
    main()
