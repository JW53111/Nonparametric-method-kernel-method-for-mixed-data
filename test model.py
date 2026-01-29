import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from statsmodels.stats.diagnostic import acorr_ljungbox
from model import MixedWeightedKernelClassifier, load_har_mixed_dataset

# 路径设置（请确保与你的 code.py 一致）
data_root = r"C:\Users\J\desktop\4th\Graduate-Project\Graduate-Project\UCI HAR Dataset"

def find_best_step_via_ljungbox(split="train", max_step=62):
    """
    通过 Ljung-Box 检验寻找使数据独立的最小步长
    """
    print("\n" + "="*60)
    print("STEP 1: Automated Ljung-Box Independence Test")
    print("="*60)
    
    y_full = pd.read_csv(os.path.join(data_root, split, f"y_{split}.txt"), header=None)[0].values
    
    best_step = 1
    passed = False
    
    results = []
    for s in range(1, max_step + 1):
        y_sampled = y_full[::s]
        # 检验前10个滞后阶数，判断序列整体平稳性和独立性
        res = acorr_ljungbox(y_sampled, lags=[10], return_df=True)
        p_val = res['lb_pvalue'].values[0]
        
        status = "PASS" if p_val > 0.05 else "FAIL"
        results.append((s, p_val, status))
        print(f"Step Size {s}: P-value = {p_val:.4e} | {status}")
        
        if p_val > 0.05 and not passed:
            best_step = s
            passed = True
            
    if not passed:
        print("Warning: No step size passed the 0.05 threshold. Using max_step.")
        best_step = max_step
        
    print(f"\n>>> Selected Best Step Size: {best_step} <<<")
    return best_step

def plot_confusion_matrix(y_true, y_pred, step):
    """
    绘制混淆矩阵，分析具体动作的分类情况
    """
    actions = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=actions, yticklabels=actions)
    plt.title(f'Confusion Matrix (Final Model, Step={step})', fontsize=14, fontweight='bold')
    plt.ylabel('True Action')
    plt.xlabel('Predicted Action')
    plt.tight_layout()
    plt.savefig('final_confusion_matrix.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # --- 1. 寻找训练集的最佳步长 ---
    train_step = find_best_step_via_ljungbox()
    
    # --- 2. 加载数据：训练集分段，测试集不分段 ---
    P_DIM = 50 
    print(f"\n" + "="*60)
    print(f"STEP 2: Data Loading (Train Step={train_step}, Test Step=1)")
    print("="*60)
    
    # 训练集：使用 Ljung-Box 选出的步长
    X_tr, y_tr = load_har_mixed_dataset(split="train", n_samples=None, p_cts=P_DIM, step=train_step)
    
    # 测试集：强制 step=1，使用全部 2947 个样本
    X_te, y_te = load_har_mixed_dataset(split="test", n_samples=None, p_cts=P_DIM, step=1)
    
    # --- 3. 训练模型 ---
    model = MixedWeightedKernelClassifier(p_fun=9, p_cat=1, p_cts=P_DIM)
    print(f"\nTraining on {len(y_tr)} samples...")
    model.fit(X_tr, y_tr)
    
    # --- 4. 预测与评估 ---
    print(f"\nEvaluating on ALL {len(y_te)} test samples...")
    y_hat = model.predict(X_tr, y_tr, X_te)
    
    # 处理可能的 NaN 并转换为 1-6 整数标签
    # 不修改 model.py，在外部进行数据清洗
    y_hat_cleaned = np.nan_to_num(y_hat, nan=np.median(y_tr))
    y_pred = np.clip(np.round(y_hat_cleaned), 1, 6).astype(int)
    
    # 核心评估
    total_acc = accuracy_score(y_te, y_pred)
    actions = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']
    
    # 【关键修复】：增加 labels 参数，防止分类报告因预测越界而崩溃
    report = classification_report(y_te, y_pred, target_names=actions, labels=[1,2,3,4,5,6])
    
    print("\n" + "="*60)
    print(f"FINAL PERFORMANCE REPORT")
    print("="*60)
    print(f"Overall Accuracy: {total_acc:.4f}")
    print("\nDetailed Action Breakdown:")
    print(report)

    # --- 5. 绘制混淆矩阵 ---
    cm = confusion_matrix(y_te, y_pred, labels=[1,2,3,4,5,6])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=actions, yticklabels=actions)
    plt.title(f'Confusion Matrix (Train_Step={train_step}, Test_Full)')
    plt.ylabel('True Action')
    plt.xlabel('Predicted Action')
    plt.show()