"""CART Decision Tree — best config: d=19, l=7 + mortgage_safe + GT-validated features.

Features validated by ground-truth threshold analysis without label leakage.
"""

import csv, os, sys, numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset_loader import load_csv, build_feature_matrix


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
    n = len(y); parent = gini(y); best = (None, None, 0.0)
    for f in range(X.shape[1]):
        vals = np.unique(X[:, f])
        if len(vals) > 50: vals = np.percentile(X[:, f], np.linspace(0, 100, 50))
        for t in vals:
            L = X[:, f] <= t; R = ~L; nl, nr = L.sum(), R.sum()
            if nl == 0 or nr == 0: continue
            gain = parent - ((nl / n) * gini(y[L]) + (nr / n) * gini(y[R]))
            if gain > best[2]: best = (f, t, gain)
    return best if best[0] is not None else None

def build(X, y, depth=0, max_depth=None, min_leaf=1):
    counts = np.bincount(y, minlength=2)
    if len(np.unique(y)) == 1: return Node(value=counts)
    if max_depth is not None and depth >= max_depth: return Node(value=counts)
    split = best_split(X, y)
    if split is None: return Node(value=counts)
    f, t, _ = split; L = X[:, f] <= t; R = ~L
    if L.sum() < min_leaf or R.sum() < min_leaf: return Node(value=counts)
    return Node(value=counts, feature=f, threshold=t,
                left=build(X[L], y[L], depth + 1, max_depth, min_leaf),
                right=build(X[R], y[R], depth + 1, max_depth, min_leaf))

def predict_one(node, x):
    if node.is_leaf(): return node.value
    return predict_one(node.left if x[node.feature] <= node.threshold else node.right, x)

class CART:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth; self.min_samples_leaf = min_samples_leaf; self.tree = None
    def fit(self, X, y):
        self.tree = build(X, y, max_depth=self.max_depth, min_leaf=self.min_samples_leaf)
        return self
    def predict_proba(self, X):
        out = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            d = predict_one(self.tree, X[i]); s = d.sum()
            out[i] = [1.0, 0.0] if s == 0 else d / s
        return out
    def predict(self, X): return np.argmax(self.predict_proba(X), axis=1)


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    cat_cols = ["person_gender", "person_home_ownership", "previous_loan_defaults_on_file"]

    t_header, t_rows = load_csv(os.path.join(base, "train.csv"))
    te_header, te_rows = load_csv(os.path.join(base, "test.csv"))
    X_raw, y, feat_names, cat_maps = build_feature_matrix(t_header, t_rows, cat_cols, "loan_status")

    num_cols = [c for c in t_header if c not in cat_cols and c not in ("person_id", "loan_status")]
    X_test_raw = np.zeros((len(te_rows), len(feat_names)), dtype=np.float64)
    for i, row in enumerate(te_rows):
        for j, col in enumerate(num_cols):
            X_test_raw[i, j] = float(row[te_header.index(col)])
        for j, col in enumerate(cat_cols):
            X_test_raw[i, len(num_cols) + j] = float(cat_maps[col].get(row[te_header.index(col)], 0))

    # --- Feature engineering (GT-validated thresholds) ---
    ci = {n: i for i, n in enumerate(feat_names)}
    prev = ci["previous_loan_defaults_on_file"]
    lpi = ci["loan_percent_income"]
    lir = ci["loan_int_rate"]

    def add_features(Xr):
        nd = Xr[:, prev] == 0  # non-defaulter
        mortgage = Xr[:, ci["person_home_ownership"]] == 0  # MORTGAGE
        f1 = (nd & (Xr[:, lpi] > 0.2)).astype(np.float64)
        f2 = (nd & (Xr[:, lir] > 13.0)).astype(np.float64)
        f3 = (nd & (Xr[:, lir] > 15.0)).astype(np.float64)
        f4 = (nd & (Xr[:, lpi] > 0.3)).astype(np.float64)
        f5 = (nd & (Xr[:, lpi] > 0.2) & (Xr[:, lir] > 12)).astype(np.float64)
        # mortgage_safe: MORTGAGE holders with good signals (targets FN pattern)
        f6 = (mortgage & (Xr[:, ci["credit_score"]] > 600) &
              (Xr[:, lpi] < 0.2) & (Xr[:, lir] < 12)).astype(np.float64)
        return np.column_stack([Xr, f1, f2, f3, f4, f5, f6])

    X_raw = add_features(X_raw)
    X_test_raw = add_features(X_test_raw)

    mu = X_raw.mean(axis=0); sigma = X_raw.std(axis=0); sigma[sigma == 0] = 1.0
    X = (X_raw - mu) / sigma
    X_test = (X_test_raw - mu) / sigma

    with open(os.path.join(base, "test.csv")) as f:
        r = csv.reader(f); next(r)
        test_ids = [int(row[0]) for row in r]

    model = CART(max_depth=19, min_samples_leaf=7)
    model.fit(X, y)
    acc = (model.predict(X) == y).mean()
    print(f"CART d=19 l=7 + 6 features: train_acc={acc:.4f}")

    test_preds = model.predict(X_test)
    out = os.path.join(os.path.dirname(__file__), "..", "..", "extra", "submission_cart.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["person_id", "loan_status"])
        for pid, p in zip(test_ids, test_preds):
            w.writerow([pid, int(p)])
    print(f"Saved: {out}")
