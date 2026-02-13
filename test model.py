import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model import MixedWeightedKernelClassifier, load_har_mixed_dataset

# 路径设置
data_root = r"C:\Users\J\desktop\4th\Graduate-Project\Integrated_HAR_Dataset"

if __name__ == "__main__":
    # --- 全局计时开始 ---
    overall_start = time.time()
    
    P_DIM = 50 
    
    # 1. 数据加载（不计入方法核心耗时）
    X_tr, y_tr = load_har_mixed_dataset(split="train", p_cts=P_DIM)
    X_te, y_te = load_har_mixed_dataset(split="test", p_cts=P_DIM)
    
    # 2. 初始化模型
    model = MixedWeightedKernelClassifier(p_fun=9, p_cat=1, p_cts=P_DIM)

    # --- 核心计时 A: 参数计算时间 (Optimization Time) ---
    # 对应样条方法的 n_basis selection
    print("\n[Running] Optimizing weights (omega) and bandwidth (h)...")
    param_select_start = time.time()
    # 内部执行距离矩阵预计算和 L-BFGS-B 优化
    model.fit(X_tr, y_tr) 
    param_select_time = time.time() - param_select_start

    # --- 核心计时 B: 核权计算与预测时间 (Kernel Computation Time) ---
    # 对应样条方法的 extract_spline_features
    print("[Running] Computing kernel weights and predicting...")
    prediction_core_start = time.time()
    y_hat = model.predict(X_tr, y_tr, X_te)
    prediction_core_time = time.time() - prediction_core_start

    # --- 数据后处理（不计入方法核心耗时） ---
    y_hat_cleaned = np.nan_to_num(y_hat, nan=np.median(y_tr))
    y_pred = np.clip(np.round(y_hat_cleaned), 1, 6).astype(int)
    
    # 结果指标
    total_acc = accuracy_score(y_te, y_pred)
    actions = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']
    
    # --- 全局计时结束 ---
    overall_time = time.time() - overall_start

    # --- 打印统一的时间报告 ---
    print("\n" + "="*60)
    print(f"{'Performance Metric':<40} {'Value'}")
    print("-" * 60)
    # 1. 单独参数计算步骤的时间 (类比 n_basis selection)
    print(f"{'1. Parameter Optimization Time (LOOCV):':<40} {param_select_time:.4f}s")
    
    # 2. 方法核心计算时间 (不含参数优化的预测过程)
    print(f"{'2. Kernel Prediction Core Time:':<40} {prediction_core_time:.4f}s")
    
    # 3. 方法总时间 (参数优化 + 核心预测)
    total_method_time = param_select_time + prediction_core_time
    print(f"{'3. Total Method Time (1 + 2):':<40} {total_method_time:.4f}s")
    
    # 4. 总体运行时间 (从程序启动到结束)
    print(f"{'4. Overall System Runtime:':<40} {overall_time:.4f}s")
    
    print("-" * 60)
    print(f"{'Final Test Accuracy:':<40} {total_acc*100:.2f}%")
    print("="*60 + "\n")

    # --- 可视化：混淆矩阵 ---
    plt.figure(figsize=(12, 9))
    cm = confusion_matrix(y_te, y_pred, labels=[1,2,3,4,5,6])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=actions, yticklabels=actions,
                cbar_kws={'label': 'Number of Samples'})
    
    # 在标题中整合关键信息
    plt.title(f'Mixed Weighted Kernel Classifier\nAccuracy: {total_acc*100:.2f}% | Total Method Time: {total_method_time:.2f}s', 
              fontsize=14, pad=20)
    plt.ylabel('Actual Activity', fontsize=12)
    plt.xlabel('Predicted Activity', fontsize=12)
    
    # 调整布局并展示
    plt.tight_layout()
    plt.show()

    # 打印简明的分类报告
    print("Classification Report:")
    print(classification_report(y_te, y_pred, target_names=actions))