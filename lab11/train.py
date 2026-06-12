import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- 配置 --------------------------
DATA_DIR = 'processed_data'
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_FRAMES = 30          # 与预处理一致
INPUT_DIM = 132
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
NUM_CLASSES = 6
# ----------------------------------------------------------

# 加载标签映射
with open(os.path.join(DATA_DIR, 'label_map.json'), 'r') as f:
    label_map = json.load(f)
idx_to_class = {v: k for k, v in label_map.items()}

# 自定义数据集
class SkeletonDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# Transformer 模型 (同实验要求)
class SkeletonTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_fc = nn.Linear(INPUT_DIM, D_MODEL)
        self.pos_embedding = nn.Parameter(torch.randn(1, TARGET_FRAMES, D_MODEL))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.classifier = nn.Linear(D_MODEL, NUM_CLASSES)

    def forward(self, x):
        x = self.input_fc(x)                      # [B, T, D]
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)                   # [B, T, D]
        x = x.mean(dim=1)                         # global pooling
        x = self.classifier(x)
        return x

# 训练函数
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        if criterion:
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    acc = correct / total
    if criterion:
        return total_loss / total, acc, all_preds, all_labels
    return acc, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print('Confusion matrix saved as confusion_matrix.png')

def inference_single(model):
    """使用测试集第一个样本演示单样本推理"""
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    idx = 0
    sample = torch.tensor(X_test[idx:idx+1], dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(sample)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)
    pred_class = idx_to_class[pred.item()]
    true_class = idx_to_class[y_test[idx]]
    print(f'Predicted class: {pred_class}')
    print(f'Confidence: {conf.item():.2f}')
    print(f'True label: {true_class}')

def main():
    # 加载数据集
    train_set = SkeletonDataset(os.path.join(DATA_DIR, 'X_train.npy'),
                                os.path.join(DATA_DIR, 'y_train.npy'))
    test_set = SkeletonDataset(os.path.join(DATA_DIR, 'X_test.npy'),
                               os.path.join(DATA_DIR, 'y_test.npy'))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = SkeletonTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
        print(f'Epoch {epoch:2d}/{EPOCHS}: Train Loss {train_loss:.4f}, '
              f'Train Acc {train_acc:.4f}, Test Loss {test_loss:.4f}, Test Acc {test_acc:.4f}')

    # 最终评估
    _, final_acc, all_preds, all_labels = evaluate(model, test_loader, criterion)
    print(f'\nFinal Test Accuracy: {final_acc:.4f}')
    class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    plot_confusion_matrix(all_labels, all_preds, class_names)

    # 保存模型
    torch.save(model.state_dict(), 'skeleton_transformer.pth')
    print('Model saved as skeleton_transformer.pth')

    # 单样本推理
    print('\n--- Single Sample Inference ---')
    inference_single(model)

if __name__ == '__main__':
    main()