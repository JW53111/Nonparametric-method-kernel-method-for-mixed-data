import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.optimize import minimize

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
    Implements the Selk–Gertheiss Weighted Kernel Method (WKM)
    for mixed-type predictors: functional, categorical, continuous.
    """

    def __init__(self, p_fun=0, p_cat=0, p_cts=0, kernel="gaussian"):
        self.p_fun = p_fun
        self.p_cat = p_cat
        self.p_cts = p_cts
        self.p = p_fun + p_cat + p_cts

        # weights
        self.omega = np.ones(self.p)
        self.h = 1.0

        if kernel == "gaussian":
            self.kernel = lambda d: np.exp(-(d ** 2) / (2 * self.h ** 2))
        else:
            raise NotImplementedError("Only Gaussian kernel supported now.")

    # -------------------------------------------------------
    # Distance functions
    # -------------------------------------------------------
    def d_fun(self, f1, f2):
        return np.sqrt(np.sum((f1 - f2) ** 2))  # L2

    def d_cat(self, x1, x2):
        return 0.0 if x1 == x2 else 1.0

    def d_cts(self, a, b):
        return abs(a - b)

    # unified distance
    def total_distance(self, x_i, x):
        """
        x_i, x are dictionaries with:
        {
            'fun': list of arrays,
            'cat': list of categorical values,
            'cts': list (or np.array) of continuous values
        }
        """
        d_sum = 0.0
        idx = 0

        # functional components
        for j in range(self.p_fun):
            d_sum += self.omega[idx] * self.d_fun(x_i['fun'][j], x['fun'][j])
            idx += 1

        # categorical components
        for j in range(self.p_cat):
            d_sum += self.omega[idx] * self.d_cat(x_i['cat'][j], x['cat'][j])
            idx += 1

        # continuous components
        for j in range(self.p_cts):
            d_sum += self.omega[idx] * self.d_cts(x_i['cts'][j], x['cts'][j])
            idx += 1

        return d_sum

    # -------------------------------------------------------
    # Kernel regression estimator
    # -------------------------------------------------------
    def kernel_estimator(self, X, y, x):
        distances = np.array([self.total_distance(X[i], x) for i in range(len(X))])
        K = self.kernel(distances)
        return np.sum(K * y) / np.sum(K)

    # -------------------------------------------------------
    # LOOCV objective Q(ω,h)
    # -------------------------------------------------------
    def LOOCV_loss(self, params, X, y):
        """
        params = [log(omega_1), ..., log(omega_p), log(h)]
        """
        # exponentiate to enforce positivity
        logw = params[:-1]
        logh = params[-1]

        self.omega = np.exp(logw)
        self.h = np.exp(logh)

        # update kernel with new h
        self.kernel = lambda d: np.exp(-(d ** 2) / (2 * self.h ** 2))

        n = len(X)
        errors = []

        for i in range(n):
            X_loo = X[:i] + X[i+1:]
            y_loo = np.concatenate([y[:i], y[i+1:]])

            y_hat = self.kernel_estimator(X_loo, y_loo, X[i])
            errors.append((y[i] - y_hat) ** 2)

        return np.mean(errors)

    # -------------------------------------------------------
    # Fit model: estimate ω and h by LOOCV
    # -------------------------------------------------------
    def fit(self, X, y):
        init_params = np.zeros(self.p + 1)
        res = minimize(lambda p: self.LOOCV_loss(p, X, y),
                       init_params, method="L-BFGS-B",
                       options={'maxiter': 50})
        print("Optimal Parameters Found")
        print("omega =", np.exp(res.x[:-1]))
        print("h     =", np.exp(res.x[-1]))

        self.omega = np.exp(res.x[:-1])
        self.h = np.exp(res.x[-1])
        self.kernel = lambda d: np.exp(-(d ** 2) / (2 * self.h ** 2))

    # -------------------------------------------------------
    # Classification (posterior probabilities)
    # -------------------------------------------------------
    def predict_class(self, X, y, x):
        classes = np.unique(y)
        probs = []

        for c in classes:
            mask = (y == c)

            distances = np.array([self.total_distance(X[i], x) for i in range(len(X))])
            K = self.kernel(distances)

            prob = np.sum(K[mask]) / np.sum(K)
            probs.append(prob)

        probs = np.array(probs)
        return classes[np.argmax(probs)], probs

    # -------------------------------------------------------
    # Predict on test set
    # -------------------------------------------------------
    def predict(self, X_train, y_train, X_test):
        preds = []
        for x in X_test:
            pred, _ = self.predict_class(X_train, y_train, x)
            preds.append(pred)
        return np.array(preds)
    

