"""CART Decision Tree — from scratch, numpy only.

Best config: max_depth=19, min_samples_leaf=8, min_samples_split=100, all 11 features.
person_age kept but capped at 100 (removes outliers age=144, 116).
Previous: dropped person_age (noisy). Deep analysis showed age is useful when capped.
"""

import csv, os, sys, numpy as np

sys.path.insert(0, os.path.dirname(__file__))


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

class CART:
    def __init__(self, max_depth=None, min_samples_leaf=1, min_samples_split=2):
        self.max_depth = max_depth; self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split; self.tree = None
    def fit(self, X, y):
        self.tree = build(X, y, max_depth=self.max_depth, min_leaf=self.min_samples_leaf,
                          min_split=self.min_samples_split)
        return self
    def predict_proba(self, X):
        out = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            d = predict_one(self.tree, X[i]); s = d.sum()
            out[i] = [1.0, 0.0] if s == 0 else d / s
        return out
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int64)


def load_with_age_capped(dataset_dir):
    """Load dataset with person_age capped at 100 to remove outliers."""
    from dataset_loader import load_csv, build_feature_matrix

    train_path = os.path.join(dataset_dir, "train.csv")
    test_path = os.path.join(dataset_dir, "test.csv")

    th, tr = load_csv(train_path)
    teh, ter = load_csv(test_path)

    # Cap age at 100 (outliers: 116, 144)
    for row in tr:
        idx = th.index("person_age")
        row[idx] = str(np.clip(float(row[idx]), 0, 100))
    for row in ter:
        idx = teh.index("person_age")
        row[idx] = str(np.clip(float(row[idx]), 0, 100))

    cat_cols = ["person_gender", "person_home_ownership", "previous_loan_defaults_on_file"]

    X_train, y_train, feature_names, cat_maps = build_feature_matrix(
        th, tr, cat_cols, "loan_status"
    )

    n_test = len(ter)
    num_cols = [c for c in teh if c not in cat_cols and c != "person_id"]
    X_test = np.zeros((n_test, len(feature_names)), dtype=np.float64)
    for i, row in enumerate(ter):
        for j, col in enumerate(num_cols):
            idx = teh.index(col)
            X_test[i, j] = float(row[idx]) if row[idx] != "" else 0.0
        for j, col in enumerate(cat_cols):
            idx = teh.index(col)
            X_test[i, len(num_cols) + j] = float(cat_maps[col].get(row[idx], 0))

    # Standardize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, feature_names


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    X, y, X_test, feat_names = load_with_age_capped(base)
    print(f"Features ({len(feat_names)}): {feat_names}")

    with open(os.path.join(base, "test.csv")) as f:
        r = csv.reader(f); next(r)
        test_ids = [int(row[0]) for row in r]

    model = CART(max_depth=19, min_samples_leaf=8, min_samples_split=100)
    model.fit(X, y)
    acc = (model.predict(X) == y).mean()
    print(f"CART d=19 l=8 min_split=100 (age capped): train_acc={acc:.4f}")

    test_preds = model.predict(X_test)
    out = os.path.join(os.path.dirname(__file__), "..", "..", "extra", "submission_cart.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["person_id", "loan_status"])
        for pid, p in zip(test_ids, test_preds):
            w.writerow([pid, int(p)])
    print(f"Saved: {out}")
