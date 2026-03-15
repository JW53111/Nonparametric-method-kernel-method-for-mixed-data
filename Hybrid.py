"""
The hybrid model uses the strengths of the previous methods.
Data loading:
-Read original 561-dimensional continuous features
-Spline coefficients
-Subject identifier
Hybrid model:
- Standardizes features and trains a multinomial logistic regression 
- Full evaluation
- Visualization

"""
import numpy as np
import pandas as pd
import os
import time
import tracemalloc
import warnings
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Setting paths and parameters (alighed with Step-3 data)

NEW_ROOT = "/Users/augleovo/PycharmProjects/PythonProject spline/.venv/bin/Integrated_HAR_Dataset_Sampled"
M_SPLINE = 14
# P_DIM is no longer used for splitting X_cts( only using for index calculations in plottig logic)
P_DIM_DRAW = 561


def get_spline_coefficients(data, n_knots=M_SPLINE):
    n_samples, n_points = data.shape
    x = np.linspace(0, 1, n_points)
    coefs = []
    for i in range(n_samples):
        c = np.interp(np.linspace(0, 1, n_knots), x, data[i])
        coefs.append(c)
    return np.array(coefs)


def load_har_for_hybrid(split="train"):
    base_path = NEW_ROOT
    y = pd.read_csv(f"{base_path}/{split}/y_{split}.txt", header=None, delim_whitespace=True)[0].values
    indices = np.arange(len(y))

    # Read all 561 continuous features
    X_cts = pd.read_csv(f"{base_path}/{split}/X_{split}.txt", header=None, delim_whitespace=True).iloc[indices].values

    sub = pd.read_csv(f"{base_path}/{split}/subject_{split}.txt", header=None, delim_whitespace=True).iloc[
        indices].values

    signal_names = ['total_acc_x', 'total_acc_y', 'total_acc_z',
                    'body_acc_x', 'body_acc_y', 'body_acc_z',
                    'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    fun_coef_list = []
    for sig in signal_names:
        data = pd.read_csv(f"{base_path}/{split}/Inertial Signals/{sig}_{split}.txt",
                           header=None, delim_whitespace=True).values[indices]
        fun_coef_list.append(get_spline_coefficients(data))

    X_fun = np.hstack(fun_coef_list)
    return X_cts, X_fun, sub, y



# Run Hybrid Model (Standard Lasso + CV)

def run_hybrid_final_reporting():
    print("--- Running Hybrid Standard Lasso (Reporting Mode - Full Features) ---")
    tracemalloc.start()

    # A. load data
    X_tr_cts, X_tr_fun, sub_tr, y_tr = load_har_for_hybrid("train")
    X_te_cts, X_te_fun, sub_te, y_te = load_har_for_hybrid("test")

    X_train = np.hstack([X_tr_cts, X_tr_fun, sub_tr])
    X_test = np.hstack([X_te_cts, X_te_fun, sub_te])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # B. Training (CV find optimal automatically)
    clf = LogisticRegressionCV(
        Cs=10, cv=5, penalty='l1', solver='saga',
        multi_class='multinomial', max_iter=3000,
        tol=1e-3, random_state=42, n_jobs=-1
    )

    fit_start = time.time()
    clf.fit(X_tr_s, y_tr)
    fit_time = time.time() - fit_start

    # C. Inference and memory monitoring
    inf_start = time.time()
    y_pred = clf.predict(X_te_s)
    y_prob = clf.predict_proba(X_te_s)
    inf_latency = time.time() - inf_start

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # D. Evaluation
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='macro')
    recall = recall_score(y_te, y_pred, average='macro')
    f1 = f1_score(y_te, y_pred, average='macro')

    # ROC-AUC (For Multi-class and using One-vs-Rest method)
    y_te_bin = pd.get_dummies(y_te).values
    roc_auc = roc_auc_score(y_te_bin, y_prob, multi_class='ovr', average='macro')

    print("\n" + "=" * 45)
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 45)
    print(f"{'Accuracy':<25} | {acc:.4f}")
    print(f"{'Precision (Macro)':<25} | {prec:.4f}")
    print(f"{'Recall (Macro)':<25} | {recall:.4f}")
    print(f"{'Macro-F1':<25} | {f1:.4f}")
    print(f"{'ROC-AUC (Macro)':<25} | {roc_auc:.4f}")
    print(f"{'Inference Latency (s)':<25} | {inf_latency:.4f}")
    print(f"{'Peak Memory (MB)':<25} | {peak / 10 ** 6:.2f}")
    print(f"{'Fit Time (s)':<25} | {fit_time:.2f}")
    print(f"{'Best C (Regularization)':<25} | {clf.C_[0]:.4f}")
    print("=" * 45)

    # E. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Figure A: Confusion Matrix
    sns.heatmap(confusion_matrix(y_te, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Walking', 'WalkUp', 'WalkDown', 'Sitting', 'Standing', 'Laying'],
                yticklabels=['Walking', 'WalkUp', 'WalkDown', 'Sitting', 'Standing', 'Laying'],
                ax=axes[0])
    axes[0].set_title("A. Confusion Matrix (Hybrid Method)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Figure B: Feature Importance Path
    importance = np.mean(np.abs(clf.coef_), axis=0)

    # (keeping consistency) Calculate indicies dynamically basednon the actual dimensions of loded X_cts
    n_cts = X_tr_cts.shape[1]
    n_fun = X_tr_fun.shape[1]

    axes[1].bar(range(n_cts), importance[:n_cts], color='lightcoral', label='Continuous (CTS)')
    axes[1].bar(range(n_cts, n_cts + n_fun), importance[n_cts:n_cts + n_fun], color='skyblue',
                label='Spline Coefficients')
    axes[1].bar(n_cts + n_fun, importance[n_cts + n_fun], color='gold', label='Subject ID')

    axes[1].set_title("B. Feature Importance (Mean |Coefficient|)")
    axes[1].set_xlabel("Feature Index")
    axes[1].set_ylabel("Magnitude")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_hybrid_final_reporting()
