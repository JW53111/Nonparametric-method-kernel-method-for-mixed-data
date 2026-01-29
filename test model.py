import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 注意：这里不再需要 acorr_ljungbox，因为预处理已经做过了
from model import MixedWeightedKernelClassifier, load_har_mixed_dataset

# 路径指向新生成的文件夹
data_root = r"C:\Users\J\desktop\4th\Graduate-Project\Integrated_HAR_Dataset"

if __name__ == "__main__":
    P_DIM = 50 
    print("\n" + "="*60)
    print(f"STEP: Loading Fixed 7:3 Integrated Dataset")
    print("="*60)
    
    # --- 1. 直接加载预处理好的固定数据 ---
    # 这里的 load_har_mixed_dataset 内部已经不再处理 step 了
    X_tr, y_tr = load_har_mixed_dataset(split="train", p_cts=P_DIM)
    X_te, y_te = load_har_mixed_dataset(split="test", p_cts=P_DIM)
    
    # --- 2. 训练模型 ---
    # 使用你 model.py 里的类
    model = MixedWeightedKernelClassifier(p_fun=9, p_cat=1, p_cts=P_DIM)
    
    print(f"\nTraining on Fixed Train Set (N={len(y_tr)})...")
    start_t = time.time()
    model.fit(X_tr, y_tr)
    print(f"Training completed in {time.time() - start_t:.2f}s")
    
    # --- 3. 预测与评估 ---
    print(f"\nPredicting on Fixed Test Set (N={len(y_te)})...")
    y_hat = model.predict(X_tr, y_tr, X_te)
    
    # 处理数值并转换为 1-6 标签
    y_hat_cleaned = np.nan_to_num(y_hat, nan=np.median(y_tr))
    y_pred = np.clip(np.round(y_hat_cleaned), 1, 6).astype(int)
    
    # 核心评估报告
    total_acc = accuracy_score(y_te, y_pred)
    actions = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']
    
    print("\n" + "="*60)
    print(f"FINAL PERFORMANCE REPORT (Fixed 7:3 Split)")
    print("="*60)
    print(f"Overall Accuracy: {total_acc:.4f}")
    print("\nDetailed Action Breakdown:")
    print(classification_report(y_te, y_pred, target_names=actions, labels=[1,2,3,4,5,6]))

    # --- 4. 绘制混淆矩阵 ---
    cm = confusion_matrix(y_te, y_pred, labels=[1,2,3,4,5,6])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=actions, yticklabels=actions)
    plt.title('Confusion Matrix (Integrated & Sampled Data)')
    plt.ylabel('True Action')
    plt.xlabel('Predicted Action')
    plt.show()