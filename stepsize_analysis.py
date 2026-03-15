"""
As much as poosible to avoid data dependence and identically distributed, we need to take step size to make sure the dataset is IID.
"""

## data has 50% overlap - Strong correlation violate the assumption of IID
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


# loading data
def load_raw_inertial_data(base_path, split="train", signal="total_acc_x"):
    """
    load the original signal data to analysis sample similarity
    """
    file_path = f"{base_path}/{split}/Inertial Signals/{signal}_{split}.txt"
    data = pd.read_csv(file_path, header=None, delim_whitespace=True).values
    return data


# Similarity Analysis by cosine similarity
# cosine similarity
def analyze_step_similarity(data, max_step=10):
    """
    Calculate the change in cosine similarity between adjacent samples as the step size increases
    """
    mean_similarities = []

    for s in range(1, max_step + 1):
        #(Row i) and (Row i+s)
        current_samples = data[:-s]
        next_samples = data[s:]


        sims = [cosine_similarity(current_samples[i].reshape(1, -1),
                                  next_samples[i].reshape(1, -1))[0, 0]
                for i in range(len(current_samples))]

        mean_similarities.append(np.mean(sims))
        print(f"  Step {s}: Average Cosine Similarity = {mean_similarities[-1]:.4f}")

    return mean_similarities


# Generalization Gap
# avoid data leakage caused be 50% data overlap.
def evaluate_generalization_gap(X_full, y_full, steps=[1, 2, 3, 5, 8]):
    """
    Compare the gap between training accuracy and test accuracy under different synchronization lengths.
    """
    results = []

    for s in steps:
        # Sample by step size
        indices = np.arange(0, len(y_full), s)
        X_s = X_full[indices]
        y_s = y_full[indices]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.3, random_state=42)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        gap = train_acc - test_acc

        results.append({
            'step': s,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': gap
        })
        print(f"  Step {s}: Train-Test Gap = {gap:.4f}")

    return pd.DataFrame(results)


# Graph and conclusion
def plot_results(similarities, gap_df):
    plt.figure(figsize=(12, 5))

    # Graph 1: Physical similarity attenuation
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(similarities) + 1), similarities, 'o-', color='teal')
    plt.axvline(x=2, color='red', linestyle='--', label='50% Overlap Limit')
    plt.title("Sample Similarity vs. Step Size")
    plt.xlabel("Step Size")
    plt.ylabel("Average Cosine Similarity")
    plt.legend()

    # Graph 2: Generalize the gap changes
    plt.subplot(1, 2, 2)
    plt.plot(gap_df['step'], gap_df['gap'], 's-', color='orange')
    plt.title("Generalization Gap (Train - Test)")
    plt.xlabel("Step Size")
    plt.ylabel("Accuracy Gap")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    DATA_PATH = "UCI HAR Dataset"

    print("--- Phase 1: Analyzing Physical Signal Similarity ---")
    raw_acc = load_raw_inertial_data(DATA_PATH)
    sims = analyze_step_similarity(raw_acc)

    print("\n--- Phase 2: Analyzing Generalization Gap ---")
    # Only the train file was used for splitting
    X_all = pd.read_csv(f"{DATA_PATH}/train/X_train.txt", header=None, delim_whitespace=True).values
    y_all = pd.read_csv(f"{DATA_PATH}/train/y_train.txt", header=None, delim_whitespace=True).values.ravel()

    gap_data = evaluate_generalization_gap(X_all, y_all)

    print("\n--- Phase 3: Visualizing Results ---")
    plot_results(sims, gap_data)
