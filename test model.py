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
    # --- 1. 寻找最佳步长 ---
    best_s = find_best_step_via_ljungbox()
    
    # --- 2. 加载全量数据 ---
    # 注意：p_cts 设为 50 以获得更高精度，如果内存不足可调回 20
    P_DIM = 50 
    print(f"\n" + "="*60)
    print(f"STEP 2: Loading Full Dataset (Step={best_s}, P={P_DIM})")
    print("="*60)
    
    X_tr, y_tr = load_har_mixed_dataset(split="train", n_samples=None, p_cts=P_DIM, step=best_s)
    X_te, y_te = load_har_mixed_dataset(split="test", n_samples=None, p_cts=P_DIM, step=best_s)
    
    # --- 3. 初始化并训练模型 ---
    model = MixedWeightedKernelClassifier(p_fun=9, p_cat=1, p_cts=P_DIM)
    
    print(f"\n" + "="*60)
    print(f"STEP 3: Training Final Model (N={len(y_tr)})")
    print("="*60)
    
    start_time = time.time()
    model.fit(X_tr, y_tr)
    train_duration = time.time() - start_time
    
    # --- 4. 预测与全方位评估 ---
    print(f"\n" + "="*60)
    print("STEP 4: Final Evaluation")
    print("="*60)
    
    y_hat = model.predict(X_tr, y_tr, X_te)
    y_pred = np.clip(np.round(y_hat), 1, 6).astype(int)
    
    # 总准确率
    total_acc = accuracy_score(y_te, y_pred)
    
    # 动作详细报告 (Precision, Recall, F1)
    actions = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']
    report = classification_report(y_te, y_pred, target_names=actions)
    
    print(f"\n[Final Results]")
    print(f"Sampling Step: {best_s}")
    print(f"Total Training Samples: {len(y_tr)}")
    print(f"Total Test Samples: {len(y_te)}")
    print(f"Overall Accuracy: {total_acc:.4f}")
    print(f"Training Time: {train_duration:.2f}s")
    print("\nDetailed Action Report:")
    print(report)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_te, y_pred, best_s)