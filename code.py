import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# 1️⃣ Load your dataset
# -----------------------------
data_root = r"C:\Users\J\desktop\4th\Graduate-Project\UCI HAR Dataset"

def load_dataset(split="train"):
    base_path = f"{data_root}/{split}"
    X = pd.read_csv(f"{base_path}/X_{split}.txt", sep=r"\s+", header=None)
    y = pd.read_csv(f"{base_path}/y_{split}.txt", header=None, names=["Activity"])
    subject = pd.read_csv(f"{base_path}/subject_{split}.txt", header=None, names=["Subject"])
    return X, y, subject

X_train, y_train, subject_train = load_dataset("train")
X_test, y_test, subject_test = load_dataset("test")

# -----------------------------
# 2️⃣ Define kernel functions
# -----------------------------
def gaussian_kernel(dist_sq, h=1.0):
    """Gaussian kernel: exp(-dist^2 / (2 * h^2))"""
    return np.exp(-dist_sq / (2 * h**2))

def categorical_distance(x1, x2):
    """0 if equal, 1 if different"""
    return (x1 != x2).astype(float)

def euclidean_distance(X, x):
    """Euclidean distance squared"""
    return np.sum((X - x)**2, axis=1)

# -----------------------------
# 3️⃣ Weighted kernel prediction
# -----------------------------
def kernel_predict(X_train, y_train, x, weights=None, h=1.0):
    """
    X_train: (n_samples, n_features) continuous predictors
    y_train: (n_samples,) target
    x: (n_features,) query point
    weights: (n_features,) weights for each feature
    h: bandwidth
    """
    if weights is None:
        weights = np.ones(X_train.shape[1])
    # weighted squared Euclidean distance
    dist_sq = np.sum(weights * (X_train - x)**2, axis=1)
    K = gaussian_kernel(dist_sq, h)
    return np.sum(K * y_train.squeeze()) / np.sum(K)

# -----------------------------
# 4️⃣ Example: leave-one-out cross-validation for one bandwidth
# -----------------------------
h = 1.0
weights = np.ones(X_train.shape[1])  # initial equal weights
y_train_values = y_train.values

y_pred_loocv = []
for i in range(len(X_train)):
    # leave-one-out
    X_loo = np.delete(X_train.values, i, axis=0)
    y_loo = np.delete(y_train_values, i, axis=0)
    x_i = X_train.values[i]
    y_hat = kernel_predict(X_loo, y_loo, x_i, weights, h)
    y_pred_loocv.append(y_hat)

y_pred_loocv = np.array(y_pred_loocv)

# Compute LOOCV MSE
mse_loocv = np.mean((y_pred_loocv - y_train_values.squeeze())**2)
print("LOOCV MSE:", mse_loocv)

# -----------------------------
# 5️⃣ Kernel classification for HAR activities (continuous features only)
# -----------------------------
def kernel_classify(X_train, y_train, x, weights=None, h=1.0):
    classes = np.unique(y_train)
    probs = []
    for c in classes:
        mask = (y_train.values.squeeze() == c)
        K = gaussian_kernel(np.sum(weights * (X_train.values - x)**2, axis=1), h)
        probs.append(np.sum(K * mask) / np.sum(K))
    probs = np.array(probs)
    return classes[np.argmax(probs)], probs

# Example: classify first 5 samples in test set
for i in range(5):
    x_i = X_test.iloc[i].values
    pred, prob = kernel_classify(X_train, y_train, x_i, weights, h)
    print(f"True: {y_test.iloc[i,0]}, Pred: {pred}, Probabilities: {prob}")


# Predict on test set
y_pred_test = []
for i in range(len(X_test)):
    x_i = X_test.iloc[i].values
    pred, _ = kernel_classify(X_train, y_train, x_i, weights, h)
    y_pred_test.append(pred)

y_pred_test = np.array(y_pred_test)
y_true_test = y_test.values.squeeze()

# ✅ Compute accuracy
acc = accuracy_score(y_true_test, y_pred_test)
print(f"Test Accuracy: {acc:.4f}")

# 🔍 Confusion matrix
cm = confusion_matrix(y_true_test, y_pred_test)
print("Confusion Matrix:\n", cm)

# Detailed performance metrics
print("\nClassification Report:\n", classification_report(y_true_test, y_pred_test))


plt.figure(figsize=(6,6))
plt.scatter(y_train_values, y_pred_loocv, alpha=0.5)
plt.plot([y_train_values.min(), y_train_values.max()],
         [y_train_values.min(), y_train_values.max()],
         color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel("True Y")
plt.ylabel("LOOCV Predicted Y")
plt.title("Kernel Regression LOOCV: True vs Predicted")
plt.legend()
plt.show()

plt.figure(figsize=(10,4))
plt.bar(range(len(weights)), weights)
plt.xlabel("Feature Index")
plt.ylabel("Weight (ω)")
plt.title("Feature Importance from Weighted Kernel Method")
plt.show()