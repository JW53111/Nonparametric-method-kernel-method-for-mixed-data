
import numpy as np
import pandas as pd
import os

# --- 1. 配置路径 ---
ORIGINAL_ROOT = r"C:\Users\J\desktop\4th\Graduate-Project\Graduate-Project\UCI HAR Dataset"
NEW_ROOT = r"C:\Users\J\desktop\4th\Graduate-Project\Integrated_HAR_Dataset_Sampled"


def save_to_uci_format(y, sub, cts, fun_dict, split_name):
    """物理保存逻辑（与原始UCI格式一致）"""
    path = os.path.join(NEW_ROOT, split_name)
    inert_path = os.path.join(path, "Inertial Signals")
    os.makedirs(inert_path, exist_ok=True)

    # 导出基础数据
    np.savetxt(os.path.join(path, f"y_{split_name}.txt"), y, fmt='%d')
    np.savetxt(os.path.join(path, f"subject_{split_name}.txt"), sub, fmt='%d')
    np.savetxt(os.path.join(path, f"X_{split_name}.txt"), cts, fmt='%.8f')

    # 导出9路信号（固定顺序）
    names = ['total_acc_x', 'total_acc_y', 'total_acc_z',
             'body_acc_x', 'body_acc_y', 'body_acc_z',
             'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    for j, name in enumerate(names):
        np.savetxt(os.path.join(inert_path, f"{name}_{split_name}.txt"),
                   fun_dict[j], fmt='%.8f')


def process_split(split):
    """处理单个split（train 或 test）：加载 → 步长3采样 → 保存"""
    print(f"\nProcessing {split} set...")
    p = os.path.join(ORIGINAL_ROOT, split)

    # ----- 读取该split的全部数据 -----
    # 标签
    y = pd.read_csv(os.path.join(p, f"y_{split}.txt"), header=None)[0].values
    # 受试者ID
    sub = pd.read_csv(os.path.join(p, f"subject_{split}.txt"), header=None)[0].values
    # 561维特征
    cts = pd.read_csv(os.path.join(p, f"X_{split}.txt"), sep=r"\s+", header=None).values
    # 9个原始信号（128维）
    sig_names = ['total_acc_x', 'total_acc_y', 'total_acc_z',
                 'body_acc_x', 'body_acc_y', 'body_acc_z',
                 'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    fun_signals = []
    for sig in sig_names:
        file_path = os.path.join(p, "Inertial Signals", f"{sig}_{split}.txt")
        fun_signals.append(pd.read_csv(file_path, sep=r"\s+", header=None).values)

    # ----- 步长3采样（每隔3行取1行）-----
    idx_sampled = np.arange(0, len(y), 3)
    y_sampled = y[idx_sampled]
    sub_sampled = sub[idx_sampled]
    cts_sampled = cts[idx_sampled]
    fun_sampled = [sig[idx_sampled] for sig in fun_signals]

    # ----- 导出到新目录 -----
    print(f"  Original samples: {len(y)}")
    print(f"  After step-3 sampling: {len(y_sampled)}")
    save_to_uci_format(y_sampled, sub_sampled, cts_sampled,
                       {j: fun_sampled[j] for j in range(9)}, split)


def run_pipeline():
    print("=== Start processing UCI HAR Dataset ===")
    # 分别处理 train 和 test
    process_split("train")
    process_split("test")
    print(f"\nDone! New dataset saved to: {NEW_ROOT}")


if __name__ == "__main__":
    run_pipeline()