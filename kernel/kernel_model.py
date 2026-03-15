import numpy as np
import pandas as pd
import time
import os
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.optimize import minimize
from scipy.integrate import simpson
from numba import njit

# Data process
# Load your dataset from local
ORIGINAL_ROOT = "/Users/augleovo/PycharmProjects/PythonProject spline/.venv/bin/UCI HAR Dataset"
data_root = "/Users/augleovo/PycharmProjects/PythonProject spline/.venv/bin/Integrated_HAR_Dataset_Sampled"

def load_har_mixed_dataset(split="train", p_cts=50):
    base_path = os.path.join(data_root, split)
    
    y = pd.read_csv(os.path.join(base_path, f"y_{split}.txt"), header=None)[0].values
    subject = pd.read_csv(os.path.join(base_path, f"subject_{split}.txt"), header=None)[0].values
    X_cts = pd.read_csv(os.path.join(base_path, f"X_{split}.txt"), sep=r"\s+", header=None).iloc[:, :p_cts].values
    
    signal_names = ['total_acc_x', 'total_acc_y', 'total_acc_z', 'body_acc_x', 'body_acc_y', 'body_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    X_fun_list = []
    for sig in signal_names:
        sig_data = pd.read_csv(os.path.join(base_path, "Inertial Signals", f"{sig}_{split}.txt"), sep=r"\s+", header=None).values
        X_fun_list.append(sig_data)

    mixed_data = []
    for i in range(len(y)):
        mixed_data.append({
            'fun': [X_fun_list[j][i] for j in range(9)],
            'cat': [subject[i]],
            'cts': X_cts[i]
        })
    return mixed_data, y

@njit
def fast_simpson_numba(y, dx):
    """Numba accelerating the simpson integral"""
    n = len(y)
    res = y[0] + y[-1]
    for i in range(1, n-1, 2):
        res += 4 * y[i]
    for i in range(2, n-2, 2):
        res += 2 * y[i]
    return (dx / 3.0) * res

@njit(parallel=True)
def precompute_dist_matrix_numba(X_fun, X_cat, X_cts, scales, p_fun, p_cat, p_cts):
    """preprocessing the metric distance: O(n^2 * p)"""
    n = X_fun.shape[0]
    p_total = p_fun + p_cat + p_cts
    dist_matrix = np.zeros((n, n, p_total))
    dx = 1.0/50.0

    for i in range(n):
        for k in range(i + 1, n):
            # 1. Functional
            for j in range(p_fun):
                diff_sq = (X_fun[i, j] - X_fun[k, j])**2
                dist_matrix[i, k, j] = fast_simpson_numba(diff_sq, dx)
            # 2. Categorical
            for j in range(p_cat):
                dist_matrix[i, k, p_fun + j] = 1.0 if X_cat[i, j] != X_cat[k, j] else 0.0
            # 3. Continuous
            for j in range(p_cts):
                dist_matrix[i, k, p_fun + p_cat + j] = (X_cts[i, j] - X_cts[k, j])**2
            #combine p together for the kernel smoothing
            for j in range(p_total):
                val = dist_matrix[i, k, j] / scales[j]
                dist_matrix[i, k, j] = val
                dist_matrix[k, i, j] = val
    return dist_matrix

# Kernel Smoothing Method
class MixedWeightedKernelClassifier:
    """
    According to Eq.(22) from report achieve the mixed kernel classifier
    Using the HAR dataset: 9 Functional, 1 categorical, 20 continuous
    """
    def __init__(self, p_fun=9, p_cat=1, p_cts=20):
        self.p_fun = p_fun
        self.p_cat = p_cat
        self.p_cts = p_cts
        self.p_total = p_fun + p_cat + p_cts
        
        # evaluate the weighted function and bandwidth: omega_j, h
        self.omega = np.ones(self.p_total)
        self.h = 1.0
        
        # keep value stable
        self.scales = np.ones(self.p_total)

    def _d_fun_sq(self, f1, f2):
        """functional data distance L2: \int (f1-f2)^2 dt"""
        dt = 1/50 
        return simpson((f1 - f2)**2, dx=dt)

    def _compute_all_distances_sq(self, x1, x2):
        """
        Calculate all Eq(22) required distance square or metrics
        return distance of p_total
        """
        d_sq_vec = np.zeros(self.p_total)
        
        # 1. Functional (L2 distance squared)
        for j in range(self.p_fun):
            d_sq_vec[j] = self._d_fun_sq(x1['fun'][j], x2['fun'][j])
            
        # 2. Categorical (Indicator distance: 0 if equal, 1 if not)
        # using square to keep consistency
        for j in range(self.p_cat):
            d_sq_vec[self.p_fun + j] = 1.0 if x1['cat'][j] != x2['cat'][j] else 0.0
            
        # 3. Continuous (Euclidean squared)
        for j in range(self.p_cts):
            d_sq_vec[self.p_fun + self.p_cat + j] = (x1['cts'][j] - x2['cts'][j])**2
            
        return d_sq_vec

    def _set_scales(self, X):
        """initialize scaling fator"""
        temp_dists = []
        for _ in range(50): # random select 50 calculated median
            i, k = np.random.choice(len(X), 2)
            temp_dists.append(self._compute_all_distances_sq(X[i], X[k]))
        self.scales = np.median(np.array(temp_dists), axis=0) + 1e-6

    def _calculate_weights(self, X_train, x_target):
        """
        核心实现 Eq.(22): 计算核权重 W_i
        W_i = exp( - sum(omega_j * d_ij^2) / (2 * h^2) )
        """
        # calculate the distance between target sample and all samples from train dataset
        # 形状: (n_train, p_total)
        diff_matrix_sq = np.array([self._compute_all_distances_sq(x_tr, x_target) for x_tr in X_train])
        
        # Normalization
        norm_diff_sq = diff_matrix_sq / self.scales
        
        # sum(omega * d^2)
        weighted_sq_dist = np.dot(norm_diff_sq, self.omega)
        
        # Gaussian kernel
        ker_weights = np.exp(-weighted_sq_dist / (2 * self.h**2))
        return ker_weights

    def loocv_loss(self, params, X, y):
        """LOOCV function"""
        self.omega = np.exp(params[:-1]) # ensure weight is positive
        self.h = np.exp(params[-1])     # ensure bandwith is positive
        
        n = len(X)
        errors = 0
        for i in range(n):
            X_loo = X[:i] + X[i+1:]
            y_loo = np.concatenate([y[:i], y[i+1:]])
            
            w = self._calculate_weights(X_loo, X[i])
            
            if np.sum(w) < 1e-10:
                y_hat = np.mean(y_loo)
            else:
                y_hat = np.dot(w, y_loo) / np.sum(w)
            
            errors += (y[i] - y_hat)**2
        return errors / n

    def fit(self, X, y):
        self._set_scales(X)
        n = len(X)
        print(f"Step 1: Pre-computing distance matrix for {n} samples...")
        
        # --- preprocessing the distance ---
        # precalculate all distance square for all samples
        # matrix shape: (n, n, p_total)
        dist_matrix_sq = np.zeros((n, n, self.p_total))
        for i in range(n):
            for k in range(i + 1, n):
                d_sq = self._compute_all_distances_sq(X[i], X[k])
                dist_matrix_sq[i, k] = d_sq
                dist_matrix_sq[k, i] = d_sq
        
        # Normalization
        dist_matrix_sq /= self.scales
        
        print("Step 2: Starting optimization (Fast Mode)...")
        init_p = np.zeros(self.p_total + 1)
        
        # find the already calculated LOOCV function
        def fast_loocv(params):
            omega = np.exp(params[:-1])
            h_sq = np.exp(params[-1])**2 * 2
            
            # matrix calculation：(n, n, p) * (p,) -> (n, n)
            weighted_dists = np.dot(dist_matrix_sq, omega)
            K = np.exp(-weighted_dists / h_sq)
            
            # (Leave-one-out)
            np.fill_diagonal(K, 0)
            
            # y_hat
            sum_K = np.sum(K, axis=1)
            # avoid missing 0
            sum_K[sum_K < 1e-10] = 1.0
            y_hat = np.dot(K, y) / sum_K
            
            return np.mean((y - y_hat)**2)

        res = minimize(fast_loocv, init_p, method='L-BFGS-B', options={'maxiter': 50})
        
        self.omega = np.exp(res.x[:-1])
        self.h = np.exp(res.x[-1])
        print("Optimization Complete.")

    def predict(self, X_train, y_train, X_test):
        preds = []
        for x in X_test:
            w = self._calculate_weights(X_train, x)
            y_hat = np.dot(w, y_train) / np.sum(w)
            preds.append(y_hat)
        return np.array(preds)
