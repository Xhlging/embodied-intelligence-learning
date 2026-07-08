"""
对比实验可视化工具
用法:
    python plot_results.py                    # 使用最新的 results 文件
    python plot_results.py results/xxx.json   # 使用指定文件
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无头模式，不弹窗

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_comparison(results, save_dir='results'):
    """绘制四种对比图"""
    os.makedirs(save_dir, exist_ok=True)
    colors = {'MLP': '#e74c3c', 'CNN': '#3498db', 'ViT': '#2ecc71'}
    markers = {'MLP': 's', 'CNN': 'o', 'ViT': '^'}

    # ---- 图1：准确率对比（训练 + 测试）----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, r in results.items():
        epochs = range(1, len(r['train_acc_list']) + 1)
        c = colors.get(name, 'gray')
        axes[0].plot(epochs, r['train_acc_list'], color=c, linestyle='--', alpha=0.6,
                     marker=markers.get(name, 'o'), markevery=max(1, len(epochs)//8), markersize=4)
        axes[0].plot(epochs, r['test_acc_list'], color=c, linewidth=2,
                     marker=markers.get(name, 'o'), markevery=max(1, len(epochs)//8), markersize=4,
                     label=f"{name} (Best: {r['best_acc']:.2%})")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy: Train (dashed) vs Test (solid)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # ---- 图2：损失对比 ----
    for name, r in results.items():
        epochs = range(1, len(r['train_loss_list']) + 1)
        c = colors.get(name, 'gray')
        axes[1].plot(epochs, r['train_loss_list'], color=c, linestyle='--', alpha=0.6)
        axes[1].plot(epochs, r['test_loss_list'], color=c, linewidth=2,
                     label=f"{name} (Final: {r['test_loss_list'][-1]:.3f})")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss: Train (dashed) vs Test (solid)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/accuracy_loss_comparison.png")

    # ---- 图3：最终准确率柱状图 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    accs = [results[n]['best_acc'] for n in names]
    bar_colors = [colors.get(n, 'gray') for n in names]

    bars = ax.bar(names, accs, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.5)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Model Comparison: Best Test Accuracy on CIFAR-10')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.2%}', ha='center', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/bar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/bar_comparison.png")

    # ---- 图4：参数量 vs 准确率 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        ax.scatter(r['params'] / 1e6, r['best_acc'], s=200,
                   color=colors.get(name, 'gray'), edgecolors='black', linewidth=1, zorder=5)
        ax.annotate(f"  {name}\n  ({r['params']:,} params)",
                    (r['params'] / 1e6, r['best_acc']),
                    fontsize=10, textcoords="offset points", xytext=(10, -5))

    ax.set_xlabel('Parameters (Millions)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Parameter Count vs Accuracy')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/params_vs_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/params_vs_accuracy.png")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # 找最新 results 文件
        files = sorted([f for f in os.listdir('results') if f.endswith('.json')])
        if not files:
            print("No results files found in results/. Run train.py first.")
            sys.exit(1)
        filepath = f'results/{files[-1]}'
        print(f"Using latest results: {filepath}")

    results = load_results(filepath)
    plot_comparison(results)
    print("Done!")
