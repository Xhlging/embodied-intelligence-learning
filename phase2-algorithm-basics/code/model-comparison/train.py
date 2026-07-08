"""
Phase 2: MLP / CNN / ViT 对比实验
在 CIFAR-10 数据集上训练三种模型，对比性能和训练行为

用法:
    python train.py --model mlp    # 训练 MLP
    python train.py --model cnn    # 训练 CNN
    python train.py --model vit    # 训练 ViT
    python train.py --model all    # 训练全部三种并对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
import json
from datetime import datetime
import time


# ============================================================
# 模型定义
# ============================================================

class SimpleMLP(nn.Module):
    """多层感知机 — 基准模型"""
    def __init__(self, input_size=3*32*32, hidden_sizes=[1024, 512, 256], num_classes=10, dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        return self.net(x)


class SimpleCNN(nn.Module):
    """卷积神经网络 — 带残差连接的小型 CNN"""
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(dropout)
        )
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(dropout)
        )
        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SimpleViT(nn.Module):
    """简化版 Vision Transformer — 展示核心 patch + attention 机制"""
    def __init__(self, image_size=32, patch_size=4, num_classes=10,
                 dim=256, depth=6, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)

        # 可学习的 [class] token 和位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size

        # 切 patch 并嵌入
        x = x.unfold(2, P, P).unfold(3, P, P)               # (B, C, H/P, W/P, P, P)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()        # (B, H/P, W/P, C, P, P)
        x = x.view(B, -1, C * P * P)                         # (B, N, C*P*P)
        x = self.patch_embed(x)                               # (B, N, dim)

        # 加 [class] token 和位置编码
        cls_tokens = self.cls_token.expand(B, -1, -1)        # (B, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)                 # (B, N+1, dim)
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # 取 [class] token 的输出做分类
        return self.head(x[:, 0])


# ============================================================
# 训练框架
# ============================================================

def get_dataloaders(batch_size=128):
    """加载 CIFAR-10 数据集"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        # 梯度裁剪（ViT 训练稳定）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def train_model(model, train_loader, test_loader, device, config):
    """训练一个模型并返回训练历史"""
    name = config['name']
    epochs = config['epochs']
    lr = config['lr']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'epoch_times': []}
    best_acc = 0

    print(f"\n{'='*60}")
    print(f"  Training: {name} | Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs: {epochs} | LR: {lr} | Device: {device}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - t0
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)

        if test_acc > best_acc:
            best_acc = test_acc

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3%} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.3%} | "
              f"Time: {epoch_time:.1f}s")

    print(f"\n  ✓ Best Test Accuracy: {best_acc:.3%}\n")
    return history, best_acc


# ============================================================
# 主程序
# ============================================================

def get_model_configs():
    return {
        'mlp': {
            'name': 'MLP',
            'model': SimpleMLP(),
            'epochs': 30,
            'lr': 1e-3,
            'weight_decay': 1e-4
        },
        'cnn': {
            'name': 'CNN',
            'model': SimpleCNN(),
            'epochs': 30,
            'lr': 1e-3,
            'weight_decay': 1e-4
        },
        'vit': {
            'name': 'ViT',
            'model': SimpleViT(image_size=32, patch_size=4, dim=256, depth=6, heads=8),
            'epochs': 50,
            'lr': 5e-4,
            'weight_decay': 5e-5
        }
    }


def main():
    parser = argparse.ArgumentParser(description='MLP/CNN/ViT 对比实验')
    parser.add_argument('--model', type=str, default='all',
                        choices=['mlp', 'cnn', 'vit', 'all'],
                        help='选择训练模型 (默认: all)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # 选择模型
    all_configs = get_model_configs()
    if args.model == 'all':
        models_to_train = ['mlp', 'cnn', 'vit']
    else:
        models_to_train = [args.model]

    results = {}

    for model_key in models_to_train:
        config = all_configs[model_key]
        model = config['model'].to(device)

        history, best_acc = train_model(model, train_loader, test_loader, device, config)
        results[config['name']] = {
            'best_acc': best_acc,
            'params': sum(p.numel() for p in model.parameters()),
            'test_acc_list': history['test_acc'],
            'train_acc_list': history['train_acc'],
            'test_loss_list': history['test_loss'],
            'train_loss_list': history['train_loss'],
        }

        # 保存模型
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/{model_key}_best.pth')

    # 打印对比结果
    if len(models_to_train) > 1:
        print(f"\n{'='*60}")
        print("  模型对比结果")
        print(f"{'='*60}")
        print(f"  {'模型':<8} {'参数量':>10} {'最佳准确率':>12}")
        print(f"  {'-'*35}")
        for name, r in results.items():
            print(f"  {name:<8} {r['params']:>10,} {r['best_acc']:>11.3%}")
        print(f"{'='*60}\n")

    # 保存结果
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results/comparison_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to results/comparison_{timestamp}.json")


if __name__ == '__main__':
    main()
