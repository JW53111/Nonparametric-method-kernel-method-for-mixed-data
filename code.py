import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.optimize import minimize
from scipy.integrate import simpson


## 打个补丁 目前代码更改就是还是用这个数据库，但是考虑三种情况
#1. acc,gyro total acc都视作functional，2.gyro变成subject_test的cts data
#2. 思考还能加什么内容
# -----------------------------
# 1️⃣ Load your dataset
# -----------------------------
data_root = r"C:\Users\J\desktop\4th\Graduate-Project\UCI HAR Dataset"

def load_dataset(split="train"):
    base_path = f"{data_root}/{split}"
    X = pd.read_csv(f"{base_path}/X_{split}.txt", sep=r"\s+", header=None)
    y = pd.read_csv(f"{base_path}/y_{split}.txt", header=None, names=["Activity"])
    subject = pd.read_csv(f"{base_path}/subject_{split}.txt", header=None, names=["Subject"])
    return X, y, subject

X_train, y_train, subject_train = load_dataset("train")
X_test, y_test, subject_test = load_dataset("test")

class MixedKernelClassifier:
    """
    Implements Mixed-Type Weighted Kernel Method (WKM)
    exactly following Selk & Gertheiss (2023) and your Section 3.2.2–3.2.4.

    Supported covariates:
        - functional curves (L2 integral metric)
        - categorical variables (0/1 mismatch)
        - continuous vector (Euclidean) with ONE weight
    """

    def __init__(self, p_fun=0, p_cat=0, kernel="gaussian"):
        self.p_fun = p_fun        # number of functional covariates
        self.p_cat = p_cat        # number of categorical variables
        self.p_cts = 1            # **one continuous component** per theory
        self.p = p_fun + p_cat + 1

        # weights (one per covariate block)
        self.omega = np.ones(self.p)
        self.h = 1.0

        if kernel == "gaussian":
            self.kernel = lambda d: np.exp(-(d ** 2) / (2 * self.h ** 2))
        else:
            raise NotImplementedError("Only Gaussian kernel supported.")

        self.fun_scale = None  # functional normalization scales

    # -------------------------------------------------------
    # 1. Functional L2 integral distance
    # -------------------------------------------------------
    def d_fun_raw(self, f1, f2):
        """Raw L2 integral distance using Simpson's rule."""
        dt = 1/50   # HAR sampling frequency = 50 Hz
        diff_sq = (f1 - f2)**2
        return np.sqrt(simpson(diff_sq, dx=dt))

    def d_fun(self, f1, f2, j):
        """Scaled functional distance."""
        d = self.d_fun_raw(f1, f2)
        return d / self.fun_scale[j]     # normalize for comparable weight learning

    # -------------------------------------------------------
    # 2. Categorical distance
    # -------------------------------------------------------
    def d_cat(self, x1, x2):
        return 0.0 if x1 == x2 else 1.0

    # -------------------------------------------------------
    # 3. Continuous covariate distance: Euclidean
    # -------------------------------------------------------
    def d_cts(self, v1, v2):
        return np.sqrt(np.sum((v1 - v2)**2))

    # =======================================================
    # UNIFIED WEIGHTED DISTANCE
    # =======================================================
    def total_distance(self, x_i, x):
        """
        x_i and x must be dicts:
        {
            'fun': list of arrays,
            'cat': list,
            'cts': np.array([...])
        }
        """
        d_sum = 0.0
        idx = 0

        # functional block
        for j in range(self.p_fun):
            d_sum += self.omega[idx] * self.d_fun(x_i['fun'][j], x['fun'][j], j)
            idx += 1

        # categorical block
        for j in range(self.p_cat):
            d_sum += self.omega[idx] * self.d_cat(x_i['cat'][j], x['cat'][j])
            idx += 1

        # continuous block (ONE Euclidean)
        d_cts = self.d_cts(x_i['cts'], x['cts'])
        d_sum += self.omega[idx] * d_cts
        idx += 1

        return d_sum

    # -------------------------------------------------------
    # Kernel regression estimator (used for LOOCV)
    # -------------------------------------------------------
    def kernel_estimator(self, X, y, x):
        distances = np.array([self.total_distance(X[i], x) for i in range(len(X))])
        K = self.kernel(distances)
        return np.sum(K * y) / np.sum(K)

    # -------------------------------------------------------
    # LOOCV loss Q(ω,h)
    # -------------------------------------------------------
    def LOOCV_loss(self, params, X, y):
        """
        params = [log(omega_1), ... log(omega_p), log(h)]
        Ensures omega_j > 0 and h > 0.
        """
        logw, logh = params[:-1], params[-1]
        self.omega = np.exp(logw)
        self.h = np.exp(logh)
        self.kernel = lambda d: np.exp(-(d ** 2) / (2 * self.h ** 2))

        n = len(X)
        errors = []

        for i in range(n):
            X_loo = X[:i] + X[i+1:]
            y_loo = np.concatenate([y[:i], y[i+1:]])

            y_hat = self.kernel_estimator(X_loo, y_loo, X[i])
            errors.append((y[i] - y_hat)**2)

        return np.mean(errors)

    # -------------------------------------------------------
    # Fit model: estimate ω and h by LOOCV
    # -------------------------------------------------------
    def compute_fun_scales(self, X):
        """Compute median L2 distance for each functional variable."""
        self.fun_scale = []
        for j in range(self.p_fun):
            dvals = []
            # use small subset for speed
            for i in range(30):
                for k in range(30):
                    dvals.append(self.d_fun_raw(X[i]['fun'][j], X[k]['fun'][j]))
            self.fun_scale.append(max(np.median(dvals), 1e-6))

    def fit(self, X, y):
        print("Computing functional scales...")
        self.compute_fun_scales(X)

        init_params = np.zeros(self.p + 1)

        print("Optimizing LOOCV...")
        res = minimize(lambda p: self.LOOCV_loss(p, X, y),
                       init_params, method="L-BFGS-B",
                       options={'maxiter': 50})

        self.omega = np.exp(res.x[:-1])
        self.h = np.exp(res.x[-1])
        self.kernel = lambda d: np.exp(-(d ** 2) / (2 * self.h ** 2))

        print("Optimal ω:", self.omega)
        print("Optimal h:", self.h)

    # -------------------------------------------------------
    # Probability and classification
    # -------------------------------------------------------
    def predict_class(self, X, y, x):
        classes = np.unique(y)
        probs = []

        distances = np.array([self.total_distance(X[i], x) for i in range(len(X))])
        K = self.kernel(distances)
        Ksum = np.sum(K)

        for c in classes:
            mask = (y == c)
            probs.append(np.sum(K[mask]) / Ksum)

        probs = np.array(probs)
        return classes[np.argmax(probs)], probs

    def predict(self, X_train, y_train, X_test):
        preds = []
        for x in X_test:
            p, _ = self.predict_class(X_train, y_train, x)
            preds.append(p)
        return np.array(preds)
    

