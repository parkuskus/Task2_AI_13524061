"""Bayesian Optimization for CART hyperparameter tuning (from scratch, numpy only).
Gaussian Process surrogate + Expected Improvement acquisition.
"""
import os, sys, csv, numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils.loader import load_csv, build_feature_matrix


class Node:
    def __init__(self, value, feature=None, threshold=None, left=None, right=None):
        self.value = value; self.feature = feature; self.threshold = threshold
        self.left = left; self.right = right
    def is_leaf(self): return self.feature is None

def gini(y):
    if len(y) == 0: return 0.0
    _, c = np.unique(y, return_counts=True); p = c / len(y)
    return 1.0 - np.sum(p ** 2)

def best_split(X, y):
    n = len(y); parent = gini(y); best_result = (None, None, 0.0)
    for f in range(X.shape[1]):
        vals = np.unique(X[:, f])
        if len(vals) > 50: vals = np.percentile(X[:, f], np.linspace(0, 100, 50))
        for t in vals:
            L = X[:, f] <= t; R = ~L; nl, nr = L.sum(), R.sum()
            if nl == 0 or nr == 0: continue
            gain = parent - ((nl/n)*gini(y[L]) + (nr/n)*gini(y[R]))
            if gain > best_result[2]: best_result = (f, t, gain)
    return best_result if best_result[0] is not None else None

def build(X, y, depth=0, max_depth=None, min_leaf=1, min_split=2):
    counts = np.bincount(y, minlength=2)
    if len(np.unique(y)) == 1: return Node(value=counts)
    if max_depth is not None and depth >= max_depth: return Node(value=counts)
    if len(y) < min_split: return Node(value=counts)
    split = best_split(X, y)
    if split is None: return Node(value=counts)
    f, t, _ = split; L = X[:, f] <= t; R = ~L
    if L.sum() < min_leaf or R.sum() < min_leaf: return Node(value=counts)
    return Node(value=counts, feature=f, threshold=t,
                left=build(X[L], y[L], depth+1, max_depth, min_leaf, min_split),
                right=build(X[R], y[R], depth+1, max_depth, min_leaf, min_split))

def predict_one(node, x):
    if node.is_leaf(): return node.value
    return predict_one(node.left if x[node.feature] <= node.threshold else node.right, x)

def predict_proba(tree_node, X):
    out = np.zeros((X.shape[0], 2))
    for i in range(X.shape[0]):
        d = predict_one(tree_node, X[i]); s = d.sum()
        out[i] = [1.0, 0.0] if s == 0 else d / s
    return out

def macro_f1(y_true, y_pred):
    f1s = []
    for cls in [0, 1]:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


# ── Gaussian Process ──

def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
    sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return variance * np.exp(-0.5 * sqdist / length_scale**2)

def gp_posterior(X_train, y_train, X_test, length_scale=1.0, variance=1.0, noise=1e-6):
    K = rbf_kernel(X_train, X_train, length_scale, variance)
    K += noise * np.eye(len(X_train))
    K_inv = np.linalg.inv(K)
    K_s = rbf_kernel(X_train, X_test, length_scale, variance)
    K_ss = rbf_kernel(X_test, X_test, length_scale, variance)
    mu = K_s.T @ K_inv @ y_train
    sigma = np.diag(K_ss) - np.sum((K_s.T @ K_inv) * K_s.T, axis=1)
    sigma = np.maximum(sigma, 1e-10)
    return mu, sigma

def expected_improvement(mu, sigma, y_best, xi=0.01):
    with np.errstate(divide='ignore'):
        z = (mu - y_best - xi) / np.sqrt(sigma)
        ei = (mu - y_best - xi) * 0.5 * (1 + np.sign(z))  # approximate CDF
        ei += np.sqrt(sigma) * np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)
        ei[sigma < 1e-10] = 0.0
    return ei


def cv_score(d, l, sp, X, y, n_folds=3):
    """3-fold stratified CV Macro F1 for speed."""
    rng = np.random.RandomState(42)
    idx = np.arange(len(y)); rng.shuffle(idx)
    fs = len(y) // n_folds; scores = []
    for fold in range(n_folds):
        s, e = fold * fs, (fold + 1) * fs
        vi = idx[s:e]; ti = np.setdiff1d(idx, vi)
        tree = build(X[ti], y[ti], max_depth=int(d), min_leaf=int(l), min_split=int(sp))
        preds = (predict_proba(tree, X[vi])[:, 1] >= 0.5).astype(np.int64)
        scores.append(macro_f1(y[vi], preds))
    return np.mean(scores)


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    t_path = os.path.join(base_dir, "train.csv")
    cat_cols = ["person_gender", "person_home_ownership", "previous_loan_defaults_on_file"]

    th, tr = load_csv(t_path)
    for row in tr:
        idx = th.index("person_age")
        row[idx] = str(np.clip(float(row[idx]), 0, 100))

    X, y, fn, cms = build_feature_matrix(th, tr, cat_cols, "loan_status")
    mean = X.mean(axis=0); std = X.std(axis=0); std[std == 0] = 1.0
    X = (X - mean) / std

    # Search space (log-scale for min_split)
    bounds = np.array([[5, 30], [2, 20], [1.7, 3.0]])  # depth, leaf, log10(split)

    n_init = 5
    n_iter = 20
    rng = np.random.RandomState(42)

    # Initial random samples
    X_init = rng.uniform(0, 1, (n_init, 3))
    X_params = bounds[:, 0] + X_init * (bounds[:, 1] - bounds[:, 0])
    y_obs = np.array([cv_score(X_params[i, 0], X_params[i, 1], 10**X_params[i, 2], X, y)
                      for i in range(n_init)])

    X_norm = X_init  # normalized to [0,1]
    best_idx = np.argmax(y_obs)
    best_f1 = y_obs[best_idx]
    best_params = X_params[best_idx]

    print(f"Init best: d={int(best_params[0])} l={int(best_params[1])} sp={int(10**best_params[2])} F1={best_f1:.4f}")

    # BO iterations
    for i in range(n_iter):
        mu, sigma = gp_posterior(X_norm, y_obs, X_norm)
        # Generate candidates for acquisition optimization
        n_cand = 1000
        X_cand = rng.uniform(0, 1, (n_cand, 3))
        mu_cand, sigma_cand = gp_posterior(X_norm, y_obs, X_cand)
        ei = expected_improvement(mu_cand, sigma_cand, best_f1)
        next_idx = np.argmax(ei)
        X_next = X_cand[next_idx]

        params = bounds[:, 0] + X_next * (bounds[:, 1] - bounds[:, 0])
        f1 = cv_score(params[0], params[1], 10**params[2], X, y)

        X_norm = np.vstack([X_norm, X_next])
        y_obs = np.append(y_obs, f1)
        X_params = np.vstack([X_params, params])

        if f1 > best_f1:
            best_f1 = f1; best_params = params
            print(f"  iter {i+1}: d={int(params[0])} l={int(params[1])} sp={int(10**params[2])} F1={f1:.4f} *** NEW BEST")

    print(f"\n=== Bayesian Optimization Result ===")
    print(f"Best: d={int(best_params[0])} l={int(best_params[1])} sp={int(10**best_params[2])} F1={best_f1:.4f}")
    print(f"Total evaluations: {n_init + n_iter}")
    print(f"Grid search equivalent: ~{len(range(5,31)) * len(range(2,21)) * len([2,10,50,100,200,500])} combos")

    # Compare with our best known config
    known_best_f1 = cv_score(19, 8, 100, X, y)
    print(f"Known best (d=19, l=8, sp=100): {known_best_f1:.4f}")
    print(f"BO improvement: {best_f1 - known_best_f1:+.4f}")
