import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# 任务1：环境准备
# ------------------------------
print("===== 任务1：环境准备 =====")
print("PyTorch 版本:", torch.__version__)
print("GPU 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("当前 GPU:", torch.cuda.get_device_name(0))
else:
    print("使用 CPU 训练")

# 简单张量操作测试
x = torch.randn(2, 3)
print("张量测试:\n", x)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 任务2：加载 MNIST 数据集
# ------------------------------
print("\n===== 任务2：加载数据集 =====")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载并加载 MNIST 训练集和测试集
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 将训练集划分为训练集(50000)和验证集(10000)
train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 显示至少8张样本图像（从完整训练集取）
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    img, label = full_train_dataset[i]
    img = img.squeeze().numpy()
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.axis('off')
plt.suptitle("Training Samples", fontsize=16)
plt.tight_layout()
plt.savefig('training_samples.png')   # 保存样本图
plt.close()
print("训练样本图已保存为 training_samples.png")

# 数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------
# 任务3：定义 CNN 模型
# ------------------------------
print("\n===== 任务3：定义 CNN 模型 =====")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)          # 28x28 -> 14x14

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)          # 14x14 -> 7x7

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)         # 输出10类

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)
print(model)

# ------------------------------
# 任务4 & 5：训练和验证
# ------------------------------
print("\n===== 任务4 & 5：训练和验证 =====")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(1, epochs + 1):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = correct / total
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = val_correct / val_total
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    print(f"Epoch {epoch}/{epochs} | "
          f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

# ------------------------------
# 任务6：测试模型
# ------------------------------
print("\n===== 任务6：测试模型 =====")

model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
all_preds = []          # 收集所有预测标签
all_labels_list = []    # 收集所有真实标签

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())     # 存储预测
        all_labels_list.extend(labels.cpu().numpy())  # 存储真实标签

test_loss = test_loss / len(test_dataset)
test_acc = test_correct / test_total
print(f"测试集 Loss: {test_loss:.4f} | 测试集 Accuracy: {test_acc:.4f}")

# 显示至少8张测试图像及其真实和预测类别
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
indices = np.random.choice(len(test_dataset), 8, replace=False)
for i, ax in enumerate(axes.flat):
    img, label = test_dataset[indices[i]]
    img_disp = img.squeeze().numpy()
    pred = all_preds[indices[i]]
    ax.imshow(img_disp, cmap='gray')
    ax.set_title(f"True: {label}, Pred: {pred}", fontsize=10)
    ax.axis('off')
plt.suptitle("Test Predictions", fontsize=16)
plt.tight_layout()
plt.savefig('test_predictions.png')   # 保存预测结果图
plt.close()
print("测试预测结果已保存为 test_predictions.png")

# ------------------------------
# 任务7：绘制训练曲线
# ------------------------------
print("\n===== 任务7：绘制曲线 =====")

epochs_list = range(1, epochs + 1)

plt.figure(figsize=(12, 5))

# Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs_list, train_losses, 'b-o', label='Training Loss')
plt.plot(epochs_list, val_losses, 'r-o', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs_list, train_accs, 'b-o', label='Training Accuracy')
plt.plot(epochs_list, val_accs, 'r-o', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png')    # 保存曲线图
plt.close()
print("训练曲线已保存为 training_curves.png")

print("\n所有实验任务完成！")