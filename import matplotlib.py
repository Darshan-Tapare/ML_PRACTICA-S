import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data = load_breast_cancer()
X, y = data.data, data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


kernels = {"Linear": "linear", "Polynomial (Poly)": "poly", "Nonlinear (RBF)": "rbf"}

results = {}


for name, kernel in kernels.items():
    model = SVC(kernel=kernel, C=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = [acc, prec, rec, f1]


metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
print(f"{'Kernel':<18} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 70)
for kernel_name, metrics in results.items():
    print(
        f"{kernel_name:<18} | {metrics[0]:.4f}     | {metrics[1]:.4f}     | {metrics[2]:.4f}     | {metrics[3]:.4f}"
    )


x = np.arange(len(metrics_names))
width = 0.25

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

for i, (kernel_name, metrics) in enumerate(results.items()):
    plt.bar(x + (i * width), metrics, width, label=kernel_name)

plt.title("SVM Kernel Performance Comparison", fontsize=14, fontweight="bold")
plt.xlabel("Evaluation Metrics", fontsize=12)
plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
plt.xticks(x + width, metrics_names)
plt.ylim(0.85, 1.05)  # Zoom in to see performance differences clearly
plt.legend(title="SVM Kernel", loc="lower right")
plt.tight_layout()
plt.show()