import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# --- 1. 配置路径 ---
ORIGINAL_ROOT = r"C:\Users\J\desktop\4th\Graduate-Project\Graduate-Project\UCI HAR Dataset"
NEW_ROOT = r"C:\Users\J\desktop\4th\Graduate-Project\Integrated_HAR_Dataset"

def save_to_uci_format(indices, y_pool, sub_pool, cts_pool, fun_pool, split_name):
    """最后的物理保存逻辑"""
    path = os.path.join(NEW_ROOT, split_name)
    inert_path = os.path.join(path, "Inertial Signals")
    os.makedirs(inert_path, exist_ok=True)
    
    # 导出基础数据
    np.savetxt(os.path.join(path, f"y_{split_name}.txt"), y_pool[indices], fmt='%d')
    np.savetxt(os.path.join(path, f"subject_{split_name}.txt"), sub_pool[indices], fmt='%d')
    np.savetxt(os.path.join(path, f"X_{split_name}.txt"), cts_pool[indices], fmt='%.8f')
    
    # 导出9路信号
    names = ['total_acc_x', 'total_acc_y', 'total_acc_z', 'body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    for j, name in enumerate(names):
        np.savetxt(os.path.join(inert_path, f"{name}_{split_name}.txt"), fun_pool[j][indices], fmt='%.8f')

def run_integration_pipeline():
    # --- STEP 1: 合并 (Merge All) ---
    print("Step 1: Merging Train and Test into one big pool...")
    all_y, all_sub, all_cts = [], [], []
    all_fun = {i: [] for i in range(9)}
    sig_names = ['total_acc_x', 'total_acc_y', 'total_acc_z', 'body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z']

    for s in ['train', 'test']:
        p = os.path.join(ORIGINAL_ROOT, s)
        all_y.extend(pd.read_csv(os.path.join(p, f"y_{s}.txt"), header=None)[0].values)
        all_sub.extend(pd.read_csv(os.path.join(p, f"subject_{s}.txt"), header=None)[0].values)
        all_cts.extend(pd.read_csv(os.path.join(p, f"X_{s}.txt"), sep=r"\s+", header=None).values)
        for j, sig in enumerate(sig_names):
            all_fun[j].extend(pd.read_csv(os.path.join(p, "Inertial Signals", f"{sig}_{s}.txt"), sep=r"\s+", header=None).values)

    all_y = np.array(all_y)
    all_sub = np.array(all_sub)
    all_cts = np.array(all_cts)
    for j in range(9): all_fun[j] = np.array(all_fun[j])

    # --- STEP 2: 采样 (Step Size = 3) ---
    print("Step 2: Sampling data with step size = 3...")
    idx_sampled = np.arange(0, len(all_y), 3)
    
    y_pool = all_y[idx_sampled]
    sub_pool = all_sub[idx_sampled]
    cts_pool = all_cts[idx_sampled]
    fun_pool = {j: all_fun[j][idx_sampled] for j in range(9)}

    # --- STEP 3: 划分 (Fixed 70/30 Split) ---
    print(f"Step 3: Splitting {len(y_pool)} samples into 70% Train and 30% Test...")
    # 使用固定随机种子 42，确保结果永恒不变
    idx_tr, idx_te = train_test_split(
        np.arange(len(y_pool)), 
        test_size=0.3, 
        random_state=42, 
        stratify=y_pool
    )

    # --- STEP 4: 导出 (Physical Export) ---
    print(f"Step 4: Exporting TXT files to {NEW_ROOT}")
    save_to_uci_format(idx_tr, y_pool, sub_pool, cts_pool, fun_pool, "train")
    save_to_uci_format(idx_te, y_pool, sub_pool, cts_pool, fun_pool, "test")

    print("\nDone! Your fixed integrated dataset is ready.")

if __name__ == "__main__":
    run_integration_pipeline()