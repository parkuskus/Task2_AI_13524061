"""CART Decision Tree implemented from scratch (numpy only).

Binary splits, weighted Gini impurity, cost-complexity pruning,
min_samples_leaf, min_impurity_decrease, threshold optimization.
"""

import numpy as np


class Node:
    def __init__(self, value, feature=None, threshold=None, left=None, right=None,
                 n_samples=0, impurity=0.0):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.n_samples = n_samples
        self.impurity = impurity

    def is_leaf(self):
        return self.feature is None


def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def weighted_gini_impurity(y, sample_weights):
    """Gini impurity with per-sample weights."""
    total_w = sample_weights.sum()
    if total_w == 0:
        return 0.0
    classes = np.unique(y)
    weighted_counts = np.array([sample_weights[y == c].sum() for c in classes])
    probs = weighted_counts / total_w
    return 1.0 - np.sum(probs ** 2)


def macro_f1_score(y_true, y_pred):
    f1s = []
    for cls in [0, 1]:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def find_optimal_threshold(y_true, probas, search_range=None, n_steps=100):
    """Find threshold that maximizes Macro F1 on probas."""
    if search_range is None:
        search_range = np.linspace(0.1, 0.9, n_steps)
    best_f1 = -1.0
    best_t = 0.5
    for t in search_range:
        preds = (probas[:, 1] >= t).astype(int)
        f1 = macro_f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def best_split(X, y, sample_weights=None, min_impurity_decrease=0.0,
               max_features=None, rng=None):
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return None

    if max_features is not None and rng is not None:
        feat_indices = rng.choice(n_features, min(max_features, n_features), replace=False)
    else:
        feat_indices = range(n_features)

    if sample_weights is None:
        parent_gini = gini_impurity(y)
    else:
        parent_gini = weighted_gini_impurity(y, sample_weights)

    best_gain = min_impurity_decrease
    best_feat = None
    best_thresh = None

    for feat in feat_indices:
        thresholds = np.unique(X[:, feat])
        if len(thresholds) > 50:
            thresholds = np.percentile(X[:, feat], np.linspace(0, 100, 50))

        for thresh in thresholds:
            left_mask = X[:, feat] <= thresh
            right_mask = ~left_mask
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)

            if n_left == 0 or n_right == 0:
                continue

            if sample_weights is None:
                child_gini = (n_left / n_samples) * gini_impurity(y[left_mask]) + \
                             (n_right / n_samples) * gini_impurity(y[right_mask])
            else:
                w_left = sample_weights[left_mask]
                w_right = sample_weights[right_mask]
                total_w = sample_weights.sum()
                child_gini = (w_left.sum() / total_w) * weighted_gini_impurity(y[left_mask], w_left) + \
                             (w_right.sum() / total_w) * weighted_gini_impurity(y[right_mask], w_right)

            gain = parent_gini - child_gini

            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh

    if best_gain <= min_impurity_decrease:
        return None
    return best_feat, best_thresh, best_gain


def build_tree(X, y, sample_weights=None, depth=0, max_depth=None,
               min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0,
               max_features=None, rng=None):
    n_samples = len(y)
    n_classes = len(np.unique(y))
    class_counts = np.bincount(y, minlength=2)

    if sample_weights is None:
        impurity = gini_impurity(y)
    else:
        impurity = weighted_gini_impurity(y, sample_weights)

    if n_classes == 1:
        return Node(value=class_counts, n_samples=n_samples, impurity=impurity)
    if max_depth is not None and depth >= max_depth:
        return Node(value=class_counts, n_samples=n_samples, impurity=impurity)
    if n_samples < min_samples_split:
        return Node(value=class_counts, n_samples=n_samples, impurity=impurity)

    split = best_split(X, y, sample_weights, min_impurity_decrease, max_features, rng)
    if split is None:
        return Node(value=class_counts, n_samples=n_samples, impurity=impurity)

    feat, thresh, _ = split
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)

    if n_left < min_samples_leaf or n_right < min_samples_leaf:
        return Node(value=class_counts, n_samples=n_samples, impurity=impurity)

    sw_left = sample_weights[left_mask] if sample_weights is not None else None
    sw_right = sample_weights[right_mask] if sample_weights is not None else None

    left = build_tree(X[left_mask], y[left_mask], sw_left, depth + 1,
                      max_depth, min_samples_split, min_samples_leaf,
                      min_impurity_decrease, max_features, rng)
    right = build_tree(X[right_mask], y[right_mask], sw_right, depth + 1,
                       max_depth, min_samples_split, min_samples_leaf,
                       min_impurity_decrease, max_features, rng)

    return Node(value=class_counts, feature=feat, threshold=thresh, left=left, right=right,
                n_samples=n_samples, impurity=impurity)


def predict_single(node, x):
    if node.is_leaf():
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_single(node.left, x)
    else:
        return predict_single(node.right, x)


def count_leaves(node):
    if node.is_leaf():
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


def node_misclassification(node):
    total = node.value.sum()
    if total == 0:
        return 0
    majority = node.value.max()
    return total - majority


def _subtree_errors(node):
    if node.is_leaf():
        return node_misclassification(node)
    return _subtree_errors(node.left) + _subtree_errors(node.right)


def _prune_node(node, X, y, ccp_alpha):
    if node.is_leaf():
        return node

    node.left = _prune_node(node.left, X, y, ccp_alpha)
    node.right = _prune_node(node.right, X, y, ccp_alpha)

    n_total = X.shape[0]
    r_t = node_misclassification(node) / n_total
    r_Tt = _subtree_errors(node) / n_total
    n_leaves = count_leaves(node)

    if n_leaves > 1 and r_Tt > 0:
        per_leaf_gain = (r_t - r_Tt) / (n_leaves - 1)
        if per_leaf_gain <= ccp_alpha:
            return Node(value=node.value, n_samples=node.n_samples, impurity=node.impurity)

    return node


class CARTDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, ccp_alpha=0.0, max_features=None,
                 class_weights=None, random_seed=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.max_features = max_features
        self.class_weights = class_weights
        self.random_seed = random_seed
        self.tree = None
        self.rng = np.random.RandomState(random_seed)
        self.threshold = 0.5

    def _compute_sample_weights(self, y):
        if self.class_weights is None:
            return None
        weights = np.ones(len(y), dtype=np.float64)
        for cls, w in self.class_weights.items():
            weights[y == cls] = w
        return weights

    def fit(self, X, y):
        sample_weights = self._compute_sample_weights(y)
        self.tree = build_tree(
            X, y, sample_weights,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            rng=self.rng,
        )

        if self.ccp_alpha > 0:
            self.tree = _prune_node(self.tree, X, y, self.ccp_alpha)

        return self

    def optimize_threshold(self, X, y, search_range=None):
        """Find threshold that maximizes Macro F1 on given data."""
        probas = self.predict_proba(X)
        self.threshold, f1 = find_optimal_threshold(y, probas, search_range)
        return self.threshold, f1

    def predict_proba(self, X):
        probas = []
        for i in range(X.shape[0]):
            dist = predict_single(self.tree, X[i])
            total = dist.sum()
            if total == 0:
                probas.append([1.0, 0.0])
            else:
                probas.append(dist / total)
        return np.array(probas)

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas[:, 1] >= self.threshold).astype(np.int64)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dataset_loader import load_dataset

    base = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    data = load_dataset(base)
    X, y = data["X_train"], data["y_train"]

    print("--- Baseline (no weights, threshold=0.5) ---")
    tree = CARTDecisionTree(max_depth=10, min_samples_leaf=5, random_seed=42)
    tree.fit(X, y)
    preds = tree.predict(X)
    acc = tree.score(X, y)
    f1 = macro_f1_score(y, preds)
    print(f"Acc={acc:.4f}, F1={f1:.4f}, leaves={count_leaves(tree.tree)}")

    print("\n--- Weighted Gini (class_weights={0:1, 1:3.5}) ---")
    tree_w = CARTDecisionTree(max_depth=10, min_samples_leaf=5,
                              class_weights={0: 1.0, 1: 3.5}, random_seed=42)
    tree_w.fit(X, y)
    preds_w = tree_w.predict(X)
    acc_w = tree_w.score(X, y)
    f1_w = macro_f1_score(y, preds_w)
    print(f"Acc={acc_w:.4f}, F1={f1_w:.4f}, leaves={count_leaves(tree_w.tree)}")

    print("\n--- Weighted Gini + Threshold Optimization ---")
    tree_wt = CARTDecisionTree(max_depth=10, min_samples_leaf=5,
                               class_weights={0: 1.0, 1: 3.5}, random_seed=42)
    tree_wt.fit(X, y)
    opt_t, opt_f1 = tree_wt.optimize_threshold(X, y)
    preds_wt = tree_wt.predict(X)
    acc_wt = tree_wt.score(X, y)
    f1_wt = macro_f1_score(y, preds_wt)
    print(f"Threshold={opt_t:.3f}, Acc={acc_wt:.4f}, F1={f1_wt:.4f}, leaves={count_leaves(tree_wt.tree)}")
