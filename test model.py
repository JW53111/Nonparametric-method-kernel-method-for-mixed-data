import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from code import MixedWeightedKernelClassifier, load_har_mixed_dataset

def run_comparison_experiments():
    # --- 1. 定义实验梯度 ---
    p_scenarios = [20, 50, 100]
    n_train_scenarios = [500, 1000, 1500, 2000]
    n_test_scenarios = [200, 500, 800] # 新增测试集梯度
    
    results = []

    # --- 2. 运行实验 A: 特征维度敏感性 (固定 n_train=500, n_test=500) ---
    print("\n>>> 实验 A: 测试不同特征维度 (p_cts) <<<")
    fixed_n_train = 500
    fixed_n_test = 500
    for p in p_scenarios:
        print(f"\n[运行] p_cts={p}, n_train={fixed_n_train}")
        X_train, y_train = load_har_mixed_dataset(split="train", n_samples=fixed_n_train, p_cts=p)
        X_test, y_test = load_har_mixed_dataset(split="test", n_samples=fixed_n_test, p_cts=p)
        
        model = MixedWeightedKernelClassifier(p_fun=9, p_cat=1, p_cts=p)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_hat = model.predict(X_train, y_train, X_test)
        y_pred = np.clip(np.round(y_hat), 1, 6).astype(int)
        acc = accuracy_score(y_test, y_pred)
        results.append({'Experiment': 'Var_P', 'p_cts': p, 'n_train': fixed_n_train, 'n_test': fixed_n_test, 'Accuracy': acc, 'Time': train_time})

    # --- 3. 运行实验 B: 训练量敏感性 (固定 p_cts=20, n_test=500) ---
    print("\n>>> 实验 B: 测试不同训练样本量 (n_train) <<<")
    fixed_p = 20
    for n in n_train_scenarios:
        print(f"\n[运行] n_train={n}, p_cts={fixed_p}")
        X_train, y_train = load_har_mixed_dataset(split="train", n_samples=n, p_cts=fixed_p)
        X_test, y_test = load_har_mixed_dataset(split="test", n_samples=fixed_n_test, p_cts=fixed_p)
        
        model = MixedWeightedKernelClassifier(p_fun=9, p_cat=1, p_cts=fixed_p)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_hat = model.predict(X_train, y_train, X_test)
        y_pred = np.clip(np.round(y_hat), 1, 6).astype(int)
        acc = accuracy_score(y_test, y_pred)
        results.append({'Experiment': 'Var_N_Train', 'p_cts': fixed_p, 'n_train': n, 'n_test': fixed_n_test, 'Accuracy': acc, 'Time': train_time})

    # --- 4. 运行实验 C: 测试量敏感性 (固定 n_train=2000, p_cts=20) ---
    print("\n>>> 实验 C: 测试不同测试样本量 (n_test) <<<")
    base_n_train = 2000
    base_p = 20
    # 为了公平和效率，先训练一个固定模型
    X_train_base, y_train_base = load_har_mixed_dataset(split="train", n_samples=base_n_train, p_cts=base_p)
    model_c = MixedWeightedKernelClassifier(p_fun=9, p_cat=1, p_cts=base_p)
    model_c.fit(X_train_base, y_train_base)

    for nt in n_test_scenarios:
        print(f"\n[运行] n_test={nt}, n_train={base_n_train}")
        X_test_c, y_test_c = load_har_mixed_dataset(split="test", n_samples=nt, p_cts=base_p)
        
        start_pred = time.time()
        y_hat = model_c.predict(X_train_base, y_train_base, X_test_c)
        pred_time = time.time() - start_pred
        
        y_pred = np.clip(np.round(y_hat), 1, 6).astype(int)
        acc = accuracy_score(y_test_c, y_pred)
        # 这里记录的是预测耗时，对分析推理效率很有帮助
        results.append({'Experiment': 'Var_N_Test', 'p_cts': base_p, 'n_train': base_n_train, 'n_test': nt, 'Accuracy': acc, 'Time': pred_time})

    # --- 5. 汇总结果 ---
    df_results = pd.DataFrame(results)
    print("\nexperiment summary report")
    print(df_results)
    df_results.to_csv("model_comparison_results.csv", index=False)

    return df_results

def run_model_case(p_val=50, n_tr=2000, n_te=200):
    print(f"\n>>> Specific Case: p={p_val}, n_train={n_tr}, n_test={n_te} <<<")
    
    # --- 修改点：只接收 2 个返回值 (mixed_data 和 labels) ---
    X_tr_mixed, y_tr = load_har_mixed_dataset(split="train", n_samples=n_tr, p_cts=p_val)
    X_te_mixed, y_te = load_har_mixed_dataset(split="test", n_samples=n_te, p_cts=p_val)
    
    # 2. 训练你的混合核模型 (Mixed Kernel)
    # 确保 p_cts 与传入的 p_val 一致
    model = MixedWeightedKernelClassifier(p_fun=9, p_cat=1, p_cts=p_val)
    
    start_fit = time.time()
    model.fit(X_tr_mixed, y_tr)
    fit_time = time.time() - start_fit
    
    # 3. 预测并评估
    # 注意：predict 通常需要传入训练集数据和标签（基于核方法的非参数特性）
    y_hat = model.predict(X_tr_mixed, y_tr, X_te_mixed)
    y_pred = np.clip(np.round(y_hat), 1, 6).astype(int)
    acc_kernel = accuracy_score(y_te, y_pred)
    
    print("-" * 40)
    print(f"Proposed Kernel Accuracy: {acc_kernel:.4f}")
    print(f"Training Time: {fit_time:.2f}s")
    print("-" * 40)
    
    return acc_kernel

if __name__ == "__main__":
    run_comparison_experiments()

if __name__ == "__main__":
    # --- 1. 运行全量敏感性分析 ---
    print("\n" + "#"*60 + "\n[Step 1] Running Sensitivity Analysis Experiments\n" + "#"*60)
    df_results = run_comparison_experiments() 
    
    # --- 2. 运行 p=50 特例 ---
    print("\n" + "#"*60 + "\n[Step 2] Running High-Dim Special Case (p=50)\n" + "#"*60)
    acc_150 = run_model_case(p_val=150, n_tr=2000, n_te=200)
    
    # --- 3. 数据整合与保存 ---
    # 统一字段名以便绘图
    df_results = df_results.rename(columns={'Accuracy': 'Accuracy_Kernel', 'Experiment': 'Type'})
    
    # 将特例数据加入表格中作为一个特殊的 Type，方便对比
    special_row = pd.DataFrame([{
        'Type': 'Special_Case_P50', 
        'p_cts': 50, 
        'n_train': 2000, 
        'n_test': 200, 
        'Accuracy_Kernel': acc_150, 
        'Time': 0 # 这里的时间如果你在 run_model_case 里返回了也可以填入
    }])
    df_final = pd.concat([df_results, special_row], ignore_index=True)
    df_final.to_csv("kernel_performance_report.csv", index=False)

    # --- 4. 绘图 A: 敏感性折线图 (针对 p, n_train, n_test) ---
    print("\nGenerating Sensitivity Plots...")
    def plot_sensitivity_only(df):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        scenarios = [
            ('Var_P', 'p_cts', 'Feature Dimension (p)'),
            ('Var_N_Train', 'n_train', 'Training Samples (n)'),
            ('Var_N_Test', 'n_test', 'Test Samples (m)')
        ]
        for i, (t_type, x_col, x_label) in enumerate(scenarios):
            sub = df[df['Type'] == t_type].sort_values(by=x_col)
            axes[i].plot(sub[x_col], sub['Accuracy_Kernel'], marker='o', color='#2E7D32', linewidth=2)
            axes[i].set_title(f'Kernel Accuracy vs {t_type.split("_")[-1]}', fontweight='bold')
            axes[i].set_xlabel(x_label)
            axes[i].set_ylabel('Accuracy')
            axes[i].set_ylim(0.6, 1.0)
        plt.tight_layout()
        plt.savefig('kernel_sensitivity.png', dpi=300)
        plt.show()

    plot_sensitivity_only(df_final)

    # --- 5. 绘图 B: 特例对比柱状图 (Accuracy & Time) ---
    print("\nGenerating Comparison Plot (Regular vs Special)...")
    def plot_case_comparison(df):
        # 挑选几个代表性的案例进行柱状图对比
        # 例如：p=20的常规 vs p=100的常规 vs p=150的特例
        compare_df = df[
            ((df['Type'] == 'Var_P') & (df['p_cts'].isin([20, 100]))) | 
            (df['Type'] == 'Special_Case_P150')
        ].copy()
        
        # 构造标签用于显示
        compare_df['Label'] = "p=" + compare_df['p_cts'].astype(str) + " (n=" + compare_df['n_train'].astype(str) + ")"
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Accuracy 对比
        plt.barplot(x='Label', y='Accuracy_Kernel', data=compare_df, ax=axes[0], palette='Greens_d')
        axes[0].set_title('Kernel Accuracy: High-Dim Comparison', fontweight='bold')
        axes[0].set_ylim(0.6, 1.0)
        
        # Time 对比
        plt.barplot(x='Label', y='Time', data=compare_df, ax=axes[1], palette='YlOrBr')
        axes[1].set_title('Computational Time Cost', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('kernel_case_comparison.png', dpi=300)
        plt.show()

    plot_case_comparison(df_final)