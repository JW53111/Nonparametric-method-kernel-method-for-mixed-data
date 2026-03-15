"""
Non-parametric method :  B‑Spline Feature Extraction + Classifier Comparison

data processing : 
  - Loads original UCI HAR Dataset train/test SEPARATELY
  - Step-3 systematic sampling on each split independently
  - NO merging, NO re-splitting
  - Saves sampled data to Integrated_HAR_Dataset_Sampled/
B-spline:
-Load continuous features: the first 50 continuouos features and functional signals: 9 signals from sampled data.
-Uses an elbow method to select the optimal number of B‑spline basis functions in training set
-Compare Logistic Regression, Random Forest, and SVM via 5‑fold CV on training set
-Full evaluation 
"""

import numpy as np
import pandas as pd
import os
import time
import warnings
import seaborn as sns
from scipy.interpolate import BSpline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, label_binarize
import tracemalloc
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Setting path
ORIGINAL_ROOT = "/Users/augleovo/PycharmProjects/PythonProject spline/.venv/bin/UCI HAR Dataset"
NEW_ROOT = "/Users/augleovo/PycharmProjects/PythonProject spline/.venv/bin/Integrated_HAR_Dataset_Sampled"

# data processing
def save_to_uci_format(y, sub, cts, fun_dict, split_name):
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
    p = os.path.join(ORIGINAL_ROOT, split)
    y = pd.read_csv(os.path.join(p, f"y_{split}.txt"), header=None, delim_whitespace=True)[0].values
    sub = pd.read_csv(os.path.join(p, f"subject_{split}.txt"), header=None, delim_whitespace=True)[0].values
    cts = pd.read_csv(os.path.join(p, f"X_{split}.txt"), delim_whitespace=True, header=None).values
    sig_names = ['total_acc_x', 'total_acc_y', 'total_acc_z',
                 'body_acc_x', 'body_acc_y', 'body_acc_z',
                 'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    fun_signals = []
    for sig in sig_names:
        file_path = os.path.join(p, "Inertial Signals", f"{sig}_{split}.txt")
        fun_signals.append(pd.read_csv(file_path, delim_whitespace=True, header=None).values)
    idx_sampled = np.arange(0, len(y), 3)
    y_sampled = y[idx_sampled]
    sub_sampled = sub[idx_sampled]
    cts_sampled = cts[idx_sampled]
    fun_sampled = [sig[idx_sampled] for sig in fun_signals]
    save_to_uci_format(y_sampled, sub_sampled, cts_sampled,
                       {j: fun_sampled[j] for j in range(9)}, split)

def preprocess_har_data():
    if not os.path.exists(NEW_ROOT):
        process_split("train")
        process_split("test")

# Load data
def load_har_data_base(split="train", n_samples=None, p_cts=50, random_state=42):
    base_path = NEW_ROOT
    y = pd.read_csv(f"{base_path}/{split}/y_{split}.txt", header=None, delim_whitespace=True)[0].values
    indices = np.arange(len(y))
    if n_samples is not None:
        indices = indices[:min(n_samples, len(indices))]
    X_cts = pd.read_csv(f"{base_path}/{split}/X_{split}.txt", header=None, delim_whitespace=True).iloc[indices, :p_cts].values
    return indices, y[indices], X_cts, base_path

def load_har_for_spline(split="train", n_samples=None, p_cts=50, random_state=42):
    res = load_har_data_base(split, n_samples, p_cts, random_state)
    final_indices, y, X_cts, base_path = res
    signal_names = ['total_acc_x', 'total_acc_y', 'total_acc_z',
                    'body_acc_x', 'body_acc_y', 'body_acc_z',
                    'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    X_func = np.zeros((len(final_indices), len(signal_names), 128))
    for j, sig in enumerate(signal_names):
        data = pd.read_csv(f"{base_path}/{split}/Inertial Signals/{sig}_{split}.txt",
                           header=None, delim_whitespace=True).values
        X_func[:, j, :] = data[final_indices]
    return {'X_func': X_func, 'X_cts': X_cts, 'y': y, 'time_grid': np.linspace(0, 1, 128)}

# n_basis selection (Figure 1)
def find_optimal_n_basis_with_viz(X_func, time_grid, degree=3, max_basis=25):
    np.random.seed(42)
    n_samples, n_signals, n_points = X_func.shape
    sample_idx = np.random.choice(n_samples, min(20, n_samples), replace=False)
    candidate_range = range(degree + 2, min(max_basis, n_points // 3))
    avg_mses = []
    std_mses = []
    for n in candidate_range:
        n_internal_knots = n - degree - 1
        if n_internal_knots < 1:
            avg_mses.append(np.inf); std_mses.append(0); continue
        knots = np.concatenate(([time_grid[0]]*(degree+1), np.linspace(time_grid[1], time_grid[-2], n_internal_knots), [time_grid[-1]]*(degree+1)))
        mses = []
        for idx in sample_idx:
            for sig_ch in range(min(3, n_signals)):
                signal = X_func[idx, sig_ch, :]
                dm = np.zeros((n_points, n))
                for i in range(n):
                    c = np.zeros(n); c[i] = 1.0
                    dm[:, i] = BSpline(knots, c, degree)(time_grid)
                coeffs = np.linalg.pinv(dm) @ signal
                mses.append(np.mean((signal - dm @ coeffs) ** 2))
        avg_mses.append(np.mean(mses)); std_mses.append(np.std(mses))
    elbow_idx = np.argmin(np.diff(np.diff(avg_mses))) + 1 if len(avg_mses) > 2 else 0
    optimal_n = list(candidate_range)[min(elbow_idx, len(candidate_range) - 1)]
    plt.figure(figsize=(10, 5))
    plt.errorbar(candidate_range, avg_mses, yerr=std_mses, fmt='o-', capsize=5, label='Reconstruction Error')
    plt.axvline(x=optimal_n, color='r', linestyle='--', label=f'Optimal n_basis={optimal_n}')
    plt.xlabel('Number of Basis Functions (n_basis)'); plt.ylabel('MSE'); plt.title('Elbow Method for n_basis Selection')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
    return optimal_n

# extract spline features
def extract_spline_features_fixed(X_func, time_grid, n_basis, degree=3):
    n_samples, n_signals, n_points = X_func.shape
    n_internal_knots = max(1, n_basis - degree - 1)
    knots = np.concatenate(([time_grid[0]]*(degree+1), np.linspace(time_grid[0], time_grid[-1], n_internal_knots+2)[1:-1], [time_grid[-1]]*(degree+1)))
    dm = np.zeros((n_points, n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis); c[i] = 1.0
        dm[:, i] = BSpline(knots, c, degree)(time_grid)
    pseudo_inv = np.linalg.pinv(dm)
    spline_features = np.zeros((n_samples, n_signals * n_basis))
    for i in range(n_samples):
        sample_all_sigs = []
        for j in range(n_signals):
            sample_all_sigs.extend(pseudo_inv @ X_func[i, j, :])
        spline_features[i, :] = sample_all_sigs
    return spline_features

# spline fitting Visualizations(figure 2 & 3)
def plot_spline_fitting_example(X_func, time_grid, n_basis, degree=3, sample_idx=0, signal_ch=0):
    n_int = max(1, n_basis - degree - 1)
    knots = np.concatenate(([time_grid[0]]*(degree+1), np.linspace(time_grid[0], time_grid[-1], n_int+2)[1:-1], [time_grid[-1]]*(degree+1)))
    dm = np.zeros((len(time_grid), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis); c[i] = 1.0
        dm[:, i] = BSpline(knots, c, degree)(time_grid)
    signal = X_func[sample_idx, signal_ch, :]
    recon = dm @ (np.linalg.pinv(dm) @ signal)
    plt.figure(figsize=(10, 4))
    plt.plot(time_grid, signal, 'o-', markersize=2, label='Original (128 points)')
    plt.plot(time_grid, recon, 'r-', linewidth=2, label=f'Spline Reconstruction (n_basis={n_basis})')
    plt.xlabel('Normalized Time'); plt.ylabel('Acceleration'); plt.title(f'Sample {sample_idx}, Signal {signal_ch} - B-spline fitting')
    plt.legend(); plt.grid(alpha=0.3); plt.show()

def plot_spline_comparison(X_func, time_grid, n_basis_list, degree=3, sample_idx=0, signal_ch=0):
    signal = X_func[sample_idx, signal_ch, :]
    plt.figure(figsize=(12, 5))
    plt.plot(time_grid, signal, 'o-', markersize=2, color='black', alpha=0.4, label='Original (128 points)')
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_basis_list)))
    for n, color in zip(n_basis_list, colors):
        n_int = max(1, n - degree - 1)
        knots = np.concatenate(([time_grid[0]]*(degree+1), np.linspace(time_grid[0], time_grid[-1], n_int+2)[1:-1], [time_grid[-1]]*(degree+1)))
        dm = np.zeros((len(time_grid), n))
        for i in range(n):
            c = np.zeros(n); c[i] = 1.0
            dm[:, i] = BSpline(knots, c, degree)(time_grid)
        recon = dm @ (np.linalg.pinv(dm) @ signal)
        # Fix variable's name: use 'recon' instead of 'reconstructed'
        plt.plot(time_grid, recon, color=color, linewidth=2, label=f'n_basis={n} (MSE={np.mean((signal-recon)**2):.4f})')
    plt.xlabel('Normalized Time'); plt.ylabel('Acceleration'); plt.title('B-spline Fitting Comparison')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# mutiple models comparison
def compare_models_with_pipeline(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1),
        "SVM (RBF)": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    }
    results = []
    fitted_pipes = {}
    print("\nComparing different classifiers:")
    for name, clf in models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, n_jobs=-1)
        pipe.fit(X_train, y_train)
        test_acc = pipe.score(X_test, y_test)
        y_pred = pipe.predict(X_test)
        y_pred_proba = pipe.predict_proba(X_test)

        prec_macro = precision_score(y_test, y_pred, average='macro')
        rec_macro = recall_score(y_test, y_pred, average='macro')
        f1_mac = f1_score(y_test, y_pred, average='macro')
        macro_auc = roc_auc_score(label_binarize(y_test, classes=sorted(np.unique(y_train))), y_pred_proba, multi_class='ovr', average='macro')

        results.append({
            'Model': name, 'CV Mean': np.mean(cv_scores), 'CV Std': np.std(cv_scores),
            'Test Acc': test_acc, 'Precision (Macro)': prec_macro, 'Recall (Macro)': rec_macro,
            'F1 (Macro)': f1_mac, 'ROC-AUC (Macro)': macro_auc
        })
        fitted_pipes[name] = pipe

        print(f"\n{name}:")
        print(f"  CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Precision/Recall/F1 (Macro): {prec_macro:.4f} / {rec_macro:.4f} / {f1_mac:.4f}")
        print(f"  ROC-AUC (Macro): {macro_auc:.4f}")
        
    return pd.DataFrame(results), fitted_pipes

# Main process
def main():
    np.random.seed(42)
    overall_start = time.time()
    tracemalloc.start()
    preprocess_har_data()
    train_data = load_har_for_spline("train", p_cts=50, random_state=42)
    test_data = load_har_for_spline("test", p_cts=50, random_state=42)

    n_basis_start = time.time()
    opt_n_basis = find_optimal_n_basis_with_viz(train_data['X_func'], train_data['time_grid'])
    n_basis_time = time.time() - n_basis_start

    spline_start = time.time()
    X_train_spl = extract_spline_features_fixed(train_data['X_func'], train_data['time_grid'], opt_n_basis)
    X_test_spl = extract_spline_features_fixed(test_data['X_func'], test_data['time_grid'], opt_n_basis)
    spline_feature_time = time.time() - spline_start

    print("\nShowing spline fitting example...")
    plot_spline_fitting_example(train_data['X_func'], train_data['time_grid'], opt_n_basis)

    print("\nShowing spline fitting comparison...")
    plot_spline_comparison(train_data['X_func'], train_data['time_grid'], [8, opt_n_basis, 20])

    X_train_combined = np.hstack([X_train_spl, train_data['X_cts']])
    X_test_combined = np.hstack([X_test_spl, test_data['X_cts']])

    results_df, fitted_pipes = compare_models_with_pipeline(X_train_combined, train_data['y'], X_test_combined, test_data['y'])





# Inference Latency for best model
    best_model_name = results_df.loc[results_df['Test Acc'].idxmax()]['Model']
    best_pipe_for_timing = fitted_pipes[best_model_name]
    
    inf_start = time.time()
    _ = best_pipe_for_timing.predict(X_test_combined) # Run prediction
    inference_latency = time.time() - inf_start



    print("\n" + "=" * 60 + "\nSUMMARY OF RESULTS\n" + "=" * 60)
    best_model = results_df.loc[results_df['Test Acc'].idxmax()]
    print(f"\nBest Model: {best_model['Model']}")
    print(f"Test Accuracy: {best_model['Test Acc']:.4f}")
    print(f"CV Accuracy: {best_model['CV Mean']:.4f} (+/- {best_model['CV Std']:.4f})")

    best_pipe = fitted_pipes[best_model['Model']]
    y_pred = best_pipe.predict(X_test_combined)
    y_pred_proba = best_pipe.predict_proba(X_test_combined)
    target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']

    print("\n" + "=" * 60 + "\nTIME BENCHMARK (Spline Method)\n" + "=" * 60)
    print(f"n_basis selection time:      {n_basis_time:.4f} sec")
    print(f"Spline feature extraction:   {spline_feature_time:.4f} sec")
    print(f"Total spline-specific time:  {n_basis_time + spline_feature_time:.4f} sec")
    print(f"Inference Latency (predict): {inference_latency:.4f} sec")
    print("=" * 60)

    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nMemory Usage:\n  Current:          {mem_current / 1024 / 1024:.2f} MB\n  Peak:             {mem_peak / 1024 / 1024:.2f} MB")

    # figure 4: Confusion Matrix
    cm = confusion_matrix(test_data['y'], y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Walking', 'WalkUp', 'WalkDown', 'Sitting', 'Standing', 'Laying'],
                yticklabels=['Walking', 'WalkUp', 'WalkDown', 'Sitting', 'Standing', 'Laying'])
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion Matrix - {best_model["Model"]}')
    plt.tight_layout(); plt.savefig('spline_confusion_matrix.png', dpi=300); plt.show()

    # figure 5: ROC Curves
    classes_list = sorted(np.unique(train_data['y']))
    y_test_bin = label_binarize(test_data['y'], classes=classes_list)
    activity_names = {1: 'WALKING', 2: 'WALKING_UP', 3: 'WALKING_DOWN', 4: 'SITTING', 5: 'STANDING', 6: 'LAYING'}
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    colors_roc = plt.cm.tab10(np.linspace(0, 1, len(classes_list)))
    auc_scores = []
    for i, (cl, color) in enumerate(zip(classes_list, colors_roc)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        auc_val = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
        auc_scores.append(auc_val)
        ax_roc.plot(fpr, tpr, color=color, linewidth=2, label=f'{activity_names[cl]} (AUC = {auc_val:.3f})')
    ax_roc.plot([0, 1], [0, 1], 'k--'); ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'ROC Curves - {best_model["Model"]}'); ax_roc.legend(); ax_roc.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('spline_roc_curves.png', dpi=300); plt.show()

    print(f"\nDetailed classification report for {best_model['Model']}:")
    print(classification_report(test_data['y'], y_pred, target_names=target_names, digits=4))

    print("\n--- Saving results ---")
    results_df.to_csv('spline_model_comparison.csv', index=False)
    print("Saved: spline_model_comparison.csv")
    pd.DataFrame({'Metric': ['Best Model', 'Accuracy'], 'Value': [best_model['Model'], best_model['Test Acc']]}).to_csv('spline_best_model_results.csv', index=False)
    print("Saved: spline_best_model_results.csv")
    pd.DataFrame({'Class': [activity_names[cl] for cl in classes_list], 'AUC': auc_scores}).to_csv('spline_per_class_auc.csv', index=False)
    print("Saved: spline_per_class_auc.csv")

if __name__ == "__main__":
    main()
