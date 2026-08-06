"""
直观感受"训练的作用": 未训练(随机权重) vs 已训练(最优权重) 对比
- 图1: 8 张测试图片, 每个模型展示"训练前"与"训练后"的预测
- 图2: CNN 第一层卷积核可视化 (训练前噪声 → 训练后边缘/颜色检测器)

用法: python visualize_training_effect.py
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from train import SimpleMLP, SimpleCNN, SimpleViT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CIFAR10_CLASSES = ['飞机', '汽车', '鸟', '猫', '鹿',
                   '狗', '青蛙', '马', '船', '卡车']

MODELS = {
    'MLP': (SimpleMLP, 'checkpoints/mlp_best.pth'),
    'CNN': (SimpleCNN, 'checkpoints/cnn_best.pth'),
    'ViT': (SimpleViT, 'checkpoints/vit_best.pth'),
}


def load_models():
    """返回 {名称: (模型, 状态字典)}"""
    result = {}
    for name, (cls, ckpt) in MODELS.items():
        model = cls()
        state = torch.load(ckpt, map_location='cpu')
        result[name] = (model, state)
    return result


def get_test_samples(n=8, seed=42):
    """从测试集取 n 张固定图片"""
    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    indices = torch.randperm(len(test_set))[:n]
    imgs, labels = [], []
    for i in indices:
        img, label = test_set[i]
        imgs.append(img)
        labels.append(label)
    return torch.stack(imgs), torch.tensor(labels), indices


def predict(model, state, imgs):
    """加载权重并预测, 返回 (预测类别, 置信度)"""
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        conf = probs.max(dim=1).values
    return pred, conf


def plot_prediction_comparison(imgs, labels, results):
    """图1: 每个模型一行, 两列: 训练前 vs 训练后"""
    n = len(imgs)
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 2 * n, figsize=(2.2 * 2 * n, 3.2 * n_models))

    for m_idx, (name, (untrained, trained)) in enumerate(results.items()):
        # 反归一化用于显示
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        show_imgs = (imgs * std + mean).clamp(0, 1)

        for i in range(n):
            # 左列: 训练前 (随机权重)
            ax = axes[m_idx, i]
            ax.imshow(show_imgs[i].permute(1, 2, 0))
            pred, conf = untrained[0][i], untrained[1][i]
            label = '真实: ' + CIFAR10_CLASSES[labels[i]]
            ax.set_title(f'{label}\n训练前: {CIFAR10_CLASSES[pred]}\n(置信度 {conf:.0%})',
                         fontsize=7, color='crimson' if pred != labels[i] else 'green')
            ax.axis('off')

            # 右列: 训练后
            ax = axes[m_idx, n + i]
            ax.imshow(show_imgs[i].permute(1, 2, 0))
            pred, conf = trained[0][i], trained[1][i]
            ax.set_title(f'训练后: {CIFAR10_CLASSES[pred]}\n(置信度 {conf:.0%})',
                         fontsize=7, color='crimson' if pred != labels[i] else 'green')
            ax.axis('off')

        # 模型名标签
        fig.text(0.005, 1 - (m_idx + 0.5) / n_models, name, fontsize=13, fontweight='bold', va='center')

    fig.suptitle('训练的作用: 左列=随机权重(训练前), 右列=训练后\n绿色=预测正确, 红色=预测错误',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('results/training_effect_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/training_effect_predictions.png")


def plot_cnn_kernels(results=None):
    """图2: CNN 第一层卷积核可视化: 训练前 vs 训练后"""
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))

    # 训练前: 独立实例，随机初始化
    model_before = SimpleCNN()
    kernels_before = model_before.conv1[0].weight.data.clone()  # clone 防止后续被覆盖
    # 训练后: 另一个独立实例，加载训练好的权重
    model_after = SimpleCNN()
    state = torch.load(MODELS['CNN'][1], map_location='cpu')
    model_after.load_state_dict(state)
    kernels_after = model_after.conv1[0].weight.data

    for i in range(8):
        # 每行取一个通道, 3 通道分别显示
        for c in range(3):
            axes[0, i].imshow(kernels_before[i, c], cmap='gray', vmin=-0.5, vmax=0.5)
            axes[1, i].imshow(kernels_after[i, c], cmap='gray', vmin=-0.5, vmax=0.5)
        axes[0, i].axis('off')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('训练前\n(随机)', fontsize=10)
    axes[1, 0].set_ylabel('训练后', fontsize=10)
    fig.suptitle('CNN 第一层卷积核: 随机噪声 → 边缘/颜色检测器', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/training_effect_kernels.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/training_effect_kernels.png")


def main():
    imgs, labels, indices = get_test_samples()
    models = load_models()

    results = {}
    for name, (model, state) in models.items():
        # 训练前: 随机权重模型 (新建实例)
        untrained_model = type(model)()
        # 注意: 需要确保两个模型结构一致
        pred_before, conf_before = predict(untrained_model, untrained_model.state_dict(), imgs)
        pred_after, conf_after = predict(model, state, imgs)
        results[name] = (
            (pred_before, conf_before),
            (pred_after, conf_after),
        )
        acc_b = (pred_before == labels).float().mean().item()
        acc_a = (pred_after == labels).float().mean().item()
        print(f"{name:>4}: 训练前正确率 {acc_b:.1%} (n={len(labels)}) → 训练后 {acc_a:.1%}")

    plot_prediction_comparison(imgs, labels, results)
    plot_cnn_kernels(results)
    print("Done!")


if __name__ == '__main__':
    main()
