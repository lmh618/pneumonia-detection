import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from model import SimpleCNN


def evaluate_model(model, loader, criterion, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * images.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return avg_loss, acc, precision, recall, f1, auc


def main():
    # =========================
    # 1. 路径
    # =========================
    BASE_DIR = r"C:\Users\23617\Downloads\archive\chest_xray"
    train_dir = os.path.join(BASE_DIR, "train")
    val_dir = os.path.join(BASE_DIR, "val")
    test_dir = os.path.join(BASE_DIR, "test")

    # =========================
    # 2. 设备
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =========================
    # 3. 超参数
    # =========================
    batch_size = 16
    num_epochs = 15
    learning_rate = 1e-4

    # =========================
    # 4. 数据增强与预处理
    # =========================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # =========================
    # 5. 数据集
    # =========================
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    print("Classes   :", train_dataset.classes)
    print("Train size:", len(train_dataset))
    print("Val size  :", len(val_dataset))
    print("Test size :", len(test_dataset))

    # =========================
    # 6. 类别统计
    # =========================
    targets = train_dataset.targets
    num_classes = len(train_dataset.classes)

    class_counts = [targets.count(i) for i in range(num_classes)]
    total_samples = sum(class_counts)

    print("Class counts:", class_counts)

    # 用于损失函数的类别权重
    class_weights = [total_samples / c for c in class_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Loss class weights:", class_weights)

    # =========================
    # 7. WeightedRandomSampler
    # =========================
    # 样本级权重：少数类样本被抽到的概率更高
    sample_class_weights = [1.0 / class_counts[label] for label in targets]
    sampler = WeightedRandomSampler(
        weights=sample_class_weights,
        num_samples=len(sample_class_weights),
        replacement=True
    )

    # =========================
    # 8. DataLoader
    # =========================
    # 训练集：用 sampler，不再 shuffle=True
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # 验证集、测试集：保持原始分布，不重采样
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # =========================
    # 9. 模型
    # =========================
    model = SimpleCNN(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # =========================
    # 10. 训练
    # =========================
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    train_losses = []

    print("\n开始训练...\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_dataset)
        train_acc = correct / total
        train_losses.append(avg_train_loss)

        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = evaluate_model(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val Precision: {val_precision:.4f} | "
            f"Val Recall: {val_recall:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step(val_f1)

    # =========================
    # 11. 加载最佳模型
    # =========================
    model.load_state_dict(best_model_wts)

    # =========================
    # 12. 保存 loss 曲线
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("loss_balanced.png", dpi=300)
    plt.show()

    # =========================
    # 13. 官方测试集评估
    # =========================
    test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = evaluate_model(
        model, test_loader, criterion, device
    )

    print("\n===== Final Test Results on Official Test Set =====")
    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Test Accuracy  : {test_acc*100:.2f}%")
    print(f"Test Precision : {test_precision:.4f}")
    print(f"Test Recall    : {test_recall:.4f}")
    print(f"Test F1-score  : {test_f1:.4f}")
    print(f"Test AUC       : {test_auc:.4f}")

    torch.save(model.state_dict(), "se_densenet121_pneumonia_balanced.pth")
    print("模型已保存为 se_densenet121_pneumonia_balanced.pth")
    print("Loss 曲线已保存为 loss_balanced.png")


if __name__ == "__main__":
    main()