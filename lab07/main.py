"""
传统机器学习方法在手写数字图像分类中的应用
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Task 1: Data Preparation
# ============================================================
print("Task 1: Data Preparation")
digits = load_digits()
X = digits.data
y = digits.target
images = digits.images

print(f"Number of images: {X.shape[0]}")
print(f"Image size: 8 x 8")
print(f"Class labels: {np.unique(y)}")

# Display sample images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')
plt.tight_layout()
plt.savefig('task1_samples.png')
plt.close()

# ============================================================
# Task 2: Data Splitting
# ============================================================
print("\n\nTask 2: Data Splitting")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print("Training set: for model training; Test set: for evaluating generalization")

# ============================================================
# Task 3: Feature Representation
# ============================================================
print("\n\nTask 3: Feature Representation")
demo_vector = images[0].flatten()
print(f"8x8 image flattened to vector dimension: {demo_vector.shape[0]}")
print("Method: concatenate each row of pixels into a 1D vector")
print("Reason: traditional ML algorithms require fixed-length numeric vectors")
print("Advantage: simple, preserves all pixel info")
print("Limitation: loses spatial structure, sensitive to geometric transforms")

# ============================================================
# Task 4: Model Training
# ============================================================
print("\n\nTask 4: Model Training")
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)
    print(f"{name}: {results[name]:.4f}")

# ============================================================
# Task 5: Result Comparison
# ============================================================
print("\n\nTask 5: Result Comparison")
print("\nModel                    Accuracy")
print("-" * 40)
for name, acc in results.items():
    print(f"{name:<25} {acc:.4f}")

best = max(results, key=results.get)
worst = min(results, key=results.get)
print(f"\nHighest accuracy: {best} ({results[best]*100:.2f}%)")
print(f"Lowest accuracy: {worst} ({results[worst]*100:.2f}%)")
print("Reason: different models have different assumptions and decision boundaries")

# Accuracy bar chart
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [v*100 for v in results.values()])
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('task5_accuracy.png')
plt.close()

# ============================================================
# Task 6: Error Analysis
# ============================================================
print("\n\nTask 6: Error Analysis (using SVM)")
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('SVM Confusion Matrix')
plt.tight_layout()
plt.savefig('task6_confusion_matrix.png')
plt.close()

# Find misclassified samples
error_idx = np.where(y_pred_svm != y_test)[0]
print(f"Number of misclassified samples: {len(error_idx)}")

# Analyze most confused digit pairs
confusion_pairs = {}
for idx in error_idx:
    pair = (y_test[idx], y_pred_svm[idx])
    confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

print("\nMost confused digit pairs:")
for (true_l, pred_l), count in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True):
    print(f"  True {true_l} -> Predicted {pred_l}: {count} times")

# Display misclassified samples
n_show = min(10, len(error_idx))
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()
for i in range(n_show):
    idx = error_idx[i]
    axes[i].imshow(X_test[idx].reshape(8, 8), cmap='gray')
    axes[i].set_title(f'True:{y_test[idx]} Pred:{y_pred_svm[idx]}', color='red')
    axes[i].axis('off')
for i in range(n_show, 10):
    axes[i].axis('off')
plt.suptitle('SVM Misclassified Samples')
plt.tight_layout()
plt.savefig('task6_error_samples.png')
plt.close()

print("\nError causes: similar digit shapes at low resolution, loss of spatial structure from flattening")