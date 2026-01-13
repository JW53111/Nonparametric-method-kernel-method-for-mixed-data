import numpy as np
import pandas as pd
import time
import os
from sklearn.metrics import accuracy_score
from code import MixedWeightedKernelClassifier, load_har_mixed_dataset

def run_comparison_experiments():
    # --- 1. 定义实验梯度 ---
    p_scenarios = [20, 50, 100]
    n_train_scenarios = [500, 1000, 2000, 3000]
    n_test_scenarios = [500, 1000, 1500] # 新增测试集梯度
    
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
    print("\n实验汇总报告:")
    print(df_results)
    df_results.to_csv("model_comparison_results.csv", index=False)

if __name__ == "__main__":
    run_comparison_experiments()