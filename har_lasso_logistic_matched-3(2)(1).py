"""
参数方法：Lasso进行特征选择+ 逻辑回归
================================================================
处理原始data保持三个方法一样：
  - 分别载入原始HAR数据集的训练集和测试集
  - 每个集都每隔3取一值
  - 不合并且不重新拆分
  - 生成新的HAR数据集

"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import tracemalloc
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ==============================================================================
# 路径设置和+定义活动编号
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_ROOT = "/Users/augleovo/PycharmProjects/PythonProject spline/.venv/bin/UCI HAR Dataset"
NEW_ROOT = "/Users/augleovo/PycharmProjects/PythonProject spline/.venv/bin/Integrated_HAR_Dataset_Sampled"
activity_map = {
    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING', 5: 'STANDING', 6: 'LAYING'
}


# ==============================================================================
# 1: 数据预处理
# ==============================================================================

def save_to_uci_format(y, sub, cts, fun_dict, split_name):
    """将数据按 UCI HAR 数据集的格式保存到磁盘"""
    path = os.path.join(NEW_ROOT, split_name)
    inert_path = os.path.join(path, "Inertial Signals")
    os.makedirs(inert_path, exist_ok=True)
    np.savetxt(os.path.join(path, f"y_{split_name}.txt"), y, fmt='%d')
    np.savetxt(os.path.join(path, f"subject_{split_name}.txt"), sub, fmt='%d')
    np.savetxt(os.path.join(path, f"X_{split_name}.txt"), cts, fmt='%.8f')
    names = ['total_acc_x', 'total_acc_y', 'total_acc_z',
             'body_acc_x', 'body_acc_y', 'body_acc_z',
             'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    for j, name in enumerate(names):
        np.savetxt(os.path.join(inert_path, f"{name}_{split_name}.txt"),
                   fun_dict[j], fmt='%.8f')


def process_split(split):
    """处理原始数据集中的测试集和训练集然后将它们每隔三个取一个样再保存成新的数据集"""
    p = os.path.join(ORIGINAL_ROOT, split)
    y = pd.read_csv(os.path.join(p, f"y_{split}.txt"), header=None, sep=r'\s+')[0].values
    sub = pd.read_csv(os.path.join(p, f"subject_{split}.txt"), header=None, sep=r'\s+')[0].values
    cts = pd.read_csv(os.path.join(p, f"X_{split}.txt"), sep=r'\s+', header=None).values
    sig_names = ['total_acc_x', 'total_acc_y', 'total_acc_z',
                 'body_acc_x', 'body_acc_y', 'body_acc_z',
                 'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    fun_signals = []
    for sig in sig_names:
        file_path = os.path.join(p, "Inertial Signals", f"{sig}_{split}.txt")
        fun_signals.append(pd.read_csv(file_path, sep=r'\s+', header=None).values)

    # 每隔三个取一个
    idx_sampled = np.arange(0, len(y), 3)
    y_sampled = y[idx_sampled]
    sub_sampled = sub[idx_sampled]
    cts_sampled = cts[idx_sampled]
    fun_sampled = [sig[idx_sampled] for sig in fun_signals]

    save_to_uci_format(y_sampled, sub_sampled, cts_sampled,
                       {j: fun_sampled[j] for j in range(9)}, split)


def preprocess_har_data():
    """生成经过抽样的新版 UCI HAR 数据集。
    如果目标文件夹 NEW_ROOT 已经存在，就跳过处理，否则开始处理。"""
    if not os.path.exists(NEW_ROOT):
        print("Preprocessing: generating step-3 sampled dataset...")
        process_split("train")
        process_split("test")
        print("Done!\n")
    else:
        print("Sampled dataset already exists, skipping preprocessing.\n")


# ==============================================================================
# 2: 加载数据
# ==============================================================================

def load_data():
    """将全部561个连续特征和标签载入刚处理好的数据集"""
    # 从原始HAR数据集中读出特征名
    features_path = os.path.join(ORIGINAL_ROOT, "features.txt")
    features_df = pd.read_csv(features_path, sep=r'\s+', header=None,
                              names=['idx', 'feature_name'])
    feature_names = features_df['feature_name'].tolist()

    # 处理重复特征名
    seen = {}
    unique_names = []
    for name in feature_names:
        if name in seen:
            seen[name] += 1
            unique_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique_names.append(name)
    feature_names = unique_names

    # 载入训练集
    X_train = pd.read_csv(os.path.join(NEW_ROOT, "train", "X_train.txt"),
                          sep=r'\s+', header=None, names=feature_names)
    y_train = pd.read_csv(os.path.join(NEW_ROOT, "train", "y_train.txt"),
                          sep=r'\s+', header=None, names=['activity'])
    subject_train = pd.read_csv(os.path.join(NEW_ROOT, "train", "subject_train.txt"),
                                sep=r'\s+', header=None, names=['subject'])

    # 载入测试集
    X_test = pd.read_csv(os.path.join(NEW_ROOT, "test", "X_test.txt"),
                         sep=r'\s+', header=None, names=feature_names)
    y_test = pd.read_csv(os.path.join(NEW_ROOT, "test", "y_test.txt"),
                         sep=r'\s+', header=None, names=['activity'])
    subject_test = pd.read_csv(os.path.join(NEW_ROOT, "test", "subject_test.txt"),
                               sep=r'\s+', header=None, names=['subject'])

    return (X_train, y_train, subject_train,
            X_test, y_test, subject_test,
            feature_names)


# ==============================================================================
# 3: 使用lasso和logistic回归处理分类
# ==============================================================================

def run_classification():
    print("=" * 60)
    print("Parametric Method: LASSO + Logistic Regression")
    print("=" * 60)

    # ---- 跟踪内存使用情况 ----
    tracemalloc.start()

    # ---- 加载数据 ----
    print("\n--- Data loading ---")
    (X_train, y_train, subject_train,
     X_test, y_test, subject_test,
     feature_names) = load_data()

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of classes: {len(activity_map)}")
    print(f"\nActivity labels: {activity_map}")

    # ---- 探索数据 是否有丢失值 训练测试集分别是怎么分布的----
    print("\n--- Data exploration ---")
    print(f"Missing values in training data: {X_train.isnull().sum().sum()}")
    print(f"Missing values in test data: {X_test.isnull().sum().sum()}")

    print("\nClass distribution in training data:")
    train_dist = y_train['activity'].value_counts().sort_index()
    for idx, count in train_dist.items():
        print(f"  {idx} ({activity_map[idx]}): {count} ({100*count/len(y_train):.1f}%)")

    print("\nClass distribution in test data:")
    test_dist = y_test['activity'].value_counts().sort_index()
    for idx, count in test_dist.items():
        print(f"  {idx} ({activity_map[idx]}): {count} ({100*count/len(y_test):.1f}%)")

    print(f"\nFeature value range: [{X_train.values.min():.4f}, {X_train.values.max():.4f}]")

    # ---- 标准化 ----
    print("\n--- Standardization ---")
    t_std_start = time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    t_std_end = time.time()
    time_standardization = t_std_end - t_std_start

    print(f"Standardized training data shape: {X_train_scaled.shape}")
    print(f"Standardized test data shape: {X_test_scaled.shape}")

    # ==========================================================================
    # 使用lasso进行选择特征（one-vs-Rest方法）
    # ==========================================================================
    print("\n--- LASSO feature selection (One-vs-Rest, CV-min alpha + Intersection) ---")
    print("Performing LASSO feature selection...\n")

    selected_features_per_class = {}
    lasso_coefs_per_class = {}
    lasso_cv_objects = {}

    y_train_array = y_train['activity'].values
    classes = sorted(y_train['activity'].unique())

    t_lasso_start = time.time()
    lasso_time_per_class = {}

    for class_label in classes:
        print(f"Processing class {class_label} ({activity_map[class_label]})...")

        t_class_start = time.time()
        y_binary = (y_train_array == class_label).astype(int)

        lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)
        lasso_cv.fit(X_train_scaled, y_binary)

        # 通过交叉验证得到最大alpha值
        non_zero_mask = lasso_cv.coef_ != 0
        selected_features = np.array(feature_names)[non_zero_mask].tolist()

        selected_features_per_class[class_label] = set(selected_features)
        lasso_coefs_per_class[class_label] = lasso_cv.coef_
        lasso_cv_objects[class_label] = lasso_cv

        t_class_end = time.time()
        lasso_time_per_class[class_label] = t_class_end - t_class_start

        print(f"  Best alpha: {lasso_cv.alpha_:.6f}")
        print(f"  Selected features: {len(selected_features)} / {len(feature_names)}")
        print(f"  Time: {lasso_time_per_class[class_label]:.2f}s")

    t_lasso_end = time.time()
    time_lasso_total = t_lasso_end - t_lasso_start

    # 交集策略
    print("\n--- Combine selected features: INTERSECTION of all classes ---")
    feature_sets = list(selected_features_per_class.values())
    intersection_features = feature_sets[0]
    for s in feature_sets[1:]:
        intersection_features = intersection_features & s

    selected_features_list = sorted(list(intersection_features))

    # 如果选出来的特征小于20 就选择更柔和的交集策略
    if len(selected_features_list) < 20:
        print(f"  Intersection yielded only {len(selected_features_list)} features.")
        print("  Fallback: keeping features selected by >= 5 out of 6 classes...")
        all_features_flat = []
        for s in feature_sets:
            all_features_flat.extend(list(s))
        freq = Counter(all_features_flat)
        selected_features_list = sorted([f for f, cnt in freq.items() if cnt >= 5])
        print(f"  Features selected by >= 5 classes: {len(selected_features_list)}")

    print(f"\nFINAL FEATURES SELECTED: {len(selected_features_list)} / {len(feature_names)}")
    print(f"Feature reduction: {100*(1-len(selected_features_list)/len(feature_names)):.1f}%")

    selected_indices = [feature_names.index(f) for f in selected_features_list]

    X_train_selected = X_train_scaled[:, selected_indices]
    X_test_selected = X_test_scaled[:, selected_indices]

    print(f"\nReduced training data shape: {X_train_selected.shape}")
    print(f"Reduced test data shape: {X_test_selected.shape}")

    #  分析特征重要性
    print("\n--- Feature importance analysis ---")
    avg_coefs = np.mean([np.abs(coefs) for coefs in lasso_coefs_per_class.values()], axis=0)

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'avg_abs_coefficient': avg_coefs
    }).sort_values('avg_abs_coefficient', ascending=False)

    print("Top 20 Most Important Features:")
    print("-" * 60)
    for i, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature'][:45]:<45} : {row['avg_abs_coefficient']:.6f}")

    # 训练逻辑回归模型
    print("\n--- Train Logistic Regression ---")
    print("Training Logistic Regression model (C=1.0, default)...")

    lr_model = LogisticRegression(
        C=1.0,
        solver='lbfgs',
        max_iter=2000,
        random_state=42,
        n_jobs=-1
    )

    t_lr_start = time.time()
    lr_model.fit(X_train_selected, y_train_array)
    t_lr_train_end = time.time()
    time_lr_train = t_lr_train_end - t_lr_start

    y_pred = lr_model.predict(X_test_selected)
    y_pred_proba = lr_model.predict_proba(X_test_selected)
    t_lr_pred_end = time.time()
    time_lr_predict = t_lr_pred_end - t_lr_train_end

    print("Model training completed!")
    print(f"  Training time: {time_lr_train:.4f}s")
    print(f"  Prediction time: {time_lr_predict:.4f}s")

    # 评估模型
    print("\n--- Model evaluation ---")
    y_test_array = y_test['activity'].values

    accuracy = accuracy_score(y_test_array, y_pred)
    precision_macro = precision_score(y_test_array, y_pred, average='macro')
    recall_macro = recall_score(y_test_array, y_pred, average='macro')
    f1_macro = f1_score(y_test_array, y_pred, average='macro')
    precision_weighted = precision_score(y_test_array, y_pred, average='weighted')
    recall_weighted = recall_score(y_test_array, y_pred, average='weighted')
    f1_weighted = f1_score(y_test_array, y_pred, average='weighted')

    print("OVERALL METRICS")
    print("-" * 50)
    print(f"Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nMacro-averaged metrics:")
    print(f"  Precision:        {precision_macro:.4f}")
    print(f"  Recall:           {recall_macro:.4f}")
    print(f"  F1-Score:         {f1_macro:.4f}")
    print(f"\nWeighted-averaged metrics:")
    print(f"  Precision:        {precision_weighted:.4f}")
    print(f"  Recall:           {recall_weighted:.4f}")
    print(f"  F1-Score:         {f1_weighted:.4f}")

    # ---- 模型内存使用情况 ----
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nMemory Usage:")
    print(f"  Current:          {mem_current / 1024 / 1024:.2f} MB")
    print(f"  Peak:             {mem_peak / 1024 / 1024:.2f} MB")

    # ---- 分类结果报告---- 
    print("\n--- Classification report ---")
    target_names = [activity_map[i] for i in sorted(activity_map.keys())]
    print(classification_report(y_test_array, y_pred, target_names=target_names))

    # ---- 生成混淆矩阵 ----
    cm = confusion_matrix(y_test_array, y_pred)
    print("Confusion Matrix (rows: actual, columns: predicted):\n")
    print(f"{'':15}", end='')
    for name in target_names:
        print(f"{name[:8]:>10}", end='')
    print()
    for i, name in enumerate(target_names):
        print(f"{name:15}", end='')
        for j in range(len(target_names)):
            print(f"{cm[i,j]:>10}", end='')
        print()

    # 与使用所有feature的模型得出来的结果进行对比
    print("\n--- Comparison with full feature model ---")
    print("Training Logistic Regression on ALL features for comparison...")

    t_full_start = time.time()
    lr_full = LogisticRegression(
        C=1.0,
        solver='lbfgs',
        max_iter=2000,
        random_state=42,
        n_jobs=-1
    )
    lr_full.fit(X_train_scaled, y_train_array)
    y_pred_full = lr_full.predict(X_test_scaled)
    t_full_end = time.time()
    time_lr_full = t_full_end - t_full_start

    accuracy_full = accuracy_score(y_test_array, y_pred_full)
    f1_full = f1_score(y_test_array, y_pred_full, average='macro')

    print("\n" + "-" * 60)
    print(f"{'Metric':<25} {'Selected Features':<20} {'All Features':<20}")
    print("-" * 60)
    print(f"{'Number of Features':<25} {len(selected_features_list):<20} {len(feature_names):<20}")
    print(f"{'Accuracy':<25} {accuracy:.4f} ({accuracy*100:.2f}%){'':<8} {accuracy_full:.4f} ({accuracy_full*100:.2f}%)")
    print(f"{'F1-Score (Macro)':<25} {f1_macro:.4f}{'':<15} {f1_full:.4f}")
    print("-" * 60)

    # ---- 所用时间 ----
    time_pipeline_total = time_standardization + time_lasso_total + time_lr_train + time_lr_predict

    print("\n" + "=" * 70)
    print("TIMING SUMMARY — Parametric Pipeline ")
    print("=" * 70)
    print(f"\n{'Step':<45} {'Time (s)':>10} {'% of Total':>12}")
    print("-" * 70)
    print(f"{'1. Standardization (fit + transform)':<45} {time_standardization:>10.4f} {100*time_standardization/time_pipeline_total:>11.1f}%")
    print(f"{'2. OvR LassoCV (6 classes total)':<45} {time_lasso_total:>10.4f} {100*time_lasso_total/time_pipeline_total:>11.1f}%")
    for cl in classes:
        print(f"{'   - Class ' + str(cl) + ' (' + activity_map[cl][:15] + ')':<45} {lasso_time_per_class[cl]:>10.4f}")
    print(f"{'3. Intersection + feature reduction':<45} {'<0.001':>10} {'~0.0':>11}%")
    print(f"{'4. Logistic Regression (train)':<45} {time_lr_train:>10.4f} {100*time_lr_train/time_pipeline_total:>11.1f}%")
    print(f"{'5. Logistic Regression (predict)':<45} {time_lr_predict:>10.4f} {100*time_lr_predict/time_pipeline_total:>11.1f}%")
    print("-" * 70)
    print(f"{'TOTAL PIPELINE (steps 1-5)':<45} {time_pipeline_total:>10.4f} {'100.0':>11}%")
    print(f"\n{'Baseline: Logistic on all 561 features':<45} {time_lr_full:>10.4f}")
    print("=" * 70)

    # 将结果保存成csv格式
    timing_df = pd.DataFrame({
        'Step': [
            'Standardization',
            'OvR LassoCV (6 classes)',
            '  Lasso - WALKING',
            '  Lasso - WALKING_UPSTAIRS',
            '  Lasso - WALKING_DOWNSTAIRS',
            '  Lasso - SITTING',
            '  Lasso - STANDING',
            '  Lasso - LAYING',
            'Logistic Regression (train)',
            'Logistic Regression (predict)',
            'TOTAL PIPELINE',
            'Baseline: Logistic on all 561'
        ],
        'Time_seconds': [
            time_standardization,
            time_lasso_total,
            lasso_time_per_class[1],
            lasso_time_per_class[2],
            lasso_time_per_class[3],
            lasso_time_per_class[4],
            lasso_time_per_class[5],
            lasso_time_per_class[6],
            time_lr_train,
            time_lr_predict,
            time_pipeline_total,
            time_lr_full
        ]
    })
    timing_df.to_csv('lasso_logistic_timing.csv', index=False)
    print("Saved: lasso_logistic_timing.csv\n")

    # 生成可视化图片
    print("\n--- Generating visualizations ---")

    fig = plt.figure(figsize=(20, 15))

    # 1. 混淆矩阵
    ax1 = fig.add_subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[name[:10] for name in target_names],
                yticklabels=[name[:10] for name in target_names], ax=ax1)
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    # 2. 归一化混淆矩阵
    ax2 = fig.add_subplot(2, 3, 2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[name[:10] for name in target_names],
                yticklabels=[name[:10] for name in target_names], ax=ax2)
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    ax2.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')

    # 3. 15个最重要的特征
    ax3 = fig.add_subplot(2, 3, 3)
    top_features = feature_importance.head(15)
    colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    ax3.barh(range(len(top_features)),
             top_features['avg_abs_coefficient'].values, color=colors_bar)
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels([f[:30] for f in top_features['feature'].values], fontsize=9)
    ax3.invert_yaxis()
    ax3.set_xlabel('Average Absolute LASSO Coefficient', fontsize=12)
    ax3.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')

    # 4.  不同类别分布比较图
    ax4 = fig.add_subplot(2, 3, 4)
    x_pos = np.arange(len(target_names))
    width = 0.35
    train_counts = [train_dist.get(i, 0) for i in sorted(activity_map.keys())]
    test_counts = [test_dist.get(i, 0) for i in sorted(activity_map.keys())]
    ax4.bar(x_pos - width/2, train_counts, width, label='Train', color='steelblue')
    ax4.bar(x_pos + width/2, test_counts, width, label='Test', color='coral')
    ax4.set_xlabel('Activity', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([name[:10] for name in target_names], rotation=45, ha='right')
    ax4.legend()

    # 5. 每个类别的准确率
    ax5 = fig.add_subplot(2, 3, 5)
    class_accuracy = cm_normalized.diagonal()
    colors_acc = plt.cm.RdYlGn(class_accuracy)
    bars = ax5.bar(target_names, class_accuracy, color=colors_acc, edgecolor='black')
    ax5.axhline(y=np.mean(class_accuracy), color='red', linestyle='--',
                label=f'Mean: {np.mean(class_accuracy):.3f}')
    ax5.set_xlabel('Activity', fontsize=12)
    ax5.set_ylabel('Accuracy', fontsize=12)
    ax5.set_title('Per-class Classification Accuracy', fontsize=14, fontweight='bold')
    ax5.set_xticklabels([name[:10] for name in target_names], rotation=45, ha='right')
    ax5.set_ylim(0, 1.1)
    ax5.legend()
    for bar, acc in zip(bars, class_accuracy):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{acc:.2f}', ha='center', va='bottom', fontsize=10)

    # 6. 每个类别所选取的特征
    ax6 = fig.add_subplot(2, 3, 6)
    features_counts = [len(selected_features_per_class[c]) for c in classes]
    colors_cls = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    bars = ax6.bar(target_names, features_counts, color=colors_cls, edgecolor='black')
    ax6.axhline(y=len(selected_features_list), color='red', linestyle='--',
                label=f'Intersection: {len(selected_features_list)}')
    ax6.set_xlabel('Activity', fontsize=12)
    ax6.set_ylabel('Number of Features', fontsize=12)
    ax6.set_title('Features Selected by LASSO per Class', fontsize=14, fontweight='bold')
    ax6.set_xticklabels([name[:10] for name in target_names], rotation=45, ha='right')
    ax6.legend()
    for bar, count in zip(bars, features_counts):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(count), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('har_lasso_logistic_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ROC曲线图
    fig2, ax = plt.subplots(figsize=(10, 8))
    y_test_bin = label_binarize(y_test_array, classes=classes)
    colors_roc = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    auc_scores = []

    for i, (class_label, color) in enumerate(zip(classes, colors_roc)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
        auc_scores.append(auc)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{activity_map[class_label]} (AUC = {auc:.3f})')

    macro_auc = np.mean(auc_scores)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves (One-vs-Rest) - Macro AUC: {macro_auc:.3f}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('har_lasso_logistic_roc.png', dpi=300, bbox_inches='tight')
    plt.show()

    # lassocv的误差曲线-6个OVR类别
    print("\n--- Generating LassoCV error curves ---")
    fig4, axes4 = plt.subplots(2, 3, figsize=(16, 9))
    axes4 = axes4.flatten()
    for i, cl in enumerate(classes):
        obj = lasso_cv_objects[cl]
        mm = np.mean(obj.mse_path_, axis=1)
        ms = np.std(obj.mse_path_, axis=1)
        oa = obj.alpha_
        ns = np.sum(obj.coef_ != 0)
        axes4[i].plot(-np.log10(obj.alphas_), mm, 'b-', linewidth=1.5)
        axes4[i].fill_between(-np.log10(obj.alphas_), mm - ms, mm + ms,
                              alpha=0.2, color='blue')
        axes4[i].axvline(x=-np.log10(oa), color='red', linestyle='--', linewidth=1.5)
        axes4[i].set_title(f'{activity_map[cl]}\n$\\lambda$={oa:.4f}, {ns} features',
                           fontsize=11, fontweight='bold')
        axes4[i].set_xlabel('$-\\log_{10}(\\lambda)$', fontsize=9)
        axes4[i].set_ylabel('MSE', fontsize=9)
        axes4[i].grid(True, alpha=0.3)
    plt.suptitle('LassoCV Error Curves — All 6 OvR Subproblems',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('har_lasso_cv_all_classes.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ---- 保存结果 ----
    print("\n--- Saving results ---")
    with open('selected_features.txt', 'w') as f:
        f.write("LASSO Selected Features for HAR Classification (CV-min + Intersection)\n")
        f.write("=" * 50 + "\n\n")
        for i, feature in enumerate(selected_features_list, 1):
            f.write(f"{i}. {feature}\n")
    print("Saved: selected_features.txt")

    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)',
                   'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)',
                   'ROC-AUC (Macro)',
                   'Features Used', 'Feature Reduction (%)',
                   'Peak Memory (MB)'],
        'Value': [accuracy, precision_macro, recall_macro, f1_macro,
                  precision_weighted, recall_weighted, f1_weighted,
                  macro_auc,
                  len(selected_features_list),
                  100*(1-len(selected_features_list)/len(feature_names)),
                  mem_peak / 1024 / 1024]
    })
    results_df.to_csv('lasso_logistic_results.csv', index=False)
    print("Saved: lasso_logistic_results.csv")


# ==============================================================================
# 判断当前文件是否直接被运行
# ==============================================================================

if __name__ == "__main__":
    preprocess_har_data()
    run_classification()
