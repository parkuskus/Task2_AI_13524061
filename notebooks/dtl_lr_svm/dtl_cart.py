"""CART Decision Tree implemented from scratch (numpy only).

Binary splits, Gini/Twoing criterion, weighted Gini, cost-complexity pruning
with Macro F1 cost, min_samples_leaf per-class, threshold optimization.
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


def twoing_split_score(y_left, y_right, n_total):
    """Twoing criterion for binary classification: PL * PR * |p(0|L) - p(0|R)|^2."""
    n_L, n_R = len(y_left), len(y_right)
    if n_L == 0 or n_R == 0:
        return 0.0
    p_L = n_L / n_total
    p_R = n_R / n_total
    p0_L = (y_left == 0).mean()
    p0_R = (y_right == 0).mean()
    return p_L * p_R * (p0_L - p0_R) ** 2


def _child_impurity(y_left, y_right, n_total, sample_weights=None,
                    sw_left=None, sw_right=None, criterion="gini"):
    if criterion == "twoing":
        return twoing_split_score(y_left, y_right, n_total)

    n_L, n_R = len(y_left), len(y_right)
    if sample_weights is None:
        return (n_L / n_total) * gini_impurity(y_left) + \
               (n_R / n_total) * gini_impurity(y_right)
    else:
        total_w = sample_weights.sum()
        return (sw_left.sum() / total_w) * weighted_gini_impurity(y_left, sw_left) + \
               (sw_right.sum() / total_w) * weighted_gini_impurity(y_right, sw_right)


def best_split(X, y, sample_weights=None, min_impurity_decrease=0.0,
               max_features=None, rng=None, criterion="gini",
               min_leaf_class_1=0):
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return None

    if max_features is not None and rng is not None:
        feat_indices = rng.choice(n_features, min(max_features, n_features), replace=False)
    else:
        feat_indices = range(n_features)

    if criterion == "twoing":
        parent_score = 0.0
    elif sample_weights is None:
        parent_score = gini_impurity(y)
    else:
        parent_score = weighted_gini_impurity(y, sample_weights)

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
            n_L, n_R = np.sum(left_mask), np.sum(right_mask)

            if n_L == 0 or n_R == 0:
                continue

            # Per-class leaf check: ensure both children have enough class-1 samples
            if min_leaf_class_1 > 0:
                c1_left = np.sum(y[left_mask] == 1)
                c1_right = np.sum(y[right_mask] == 1)
                if c1_left < min_leaf_class_1 or c1_right < min_leaf_class_1:
                    continue

            sw_left = sample_weights[left_mask] if sample_weights is not None else None
            sw_right = sample_weights[right_mask] if sample_weights is not None else None

            child_score = _child_impurity(
                y[left_mask], y[right_mask], n_samples,
                sample_weights, sw_left, sw_right, criterion)

            gain = parent_score - child_score if criterion == "gini" else child_score

            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh

    if best_feat is None or best_gain <= min_impurity_decrease:
        return None
    return best_feat, best_thresh, best_gain


def build_tree(X, y, sample_weights=None, depth=0, max_depth=None,
               min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0,
               max_features=None, rng=None, criterion="gini",
               min_leaf_class_1=0):
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
    # Leaf if minority count already too small
    if min_leaf_class_1 > 0 and class_counts[1] < min_leaf_class_1 * 2:
        return Node(value=class_counts, n_samples=n_samples, impurity=impurity)

    split = best_split(X, y, sample_weights, min_impurity_decrease,
                       max_features, rng, criterion, min_leaf_class_1)
    if split is None:
        return Node(value=class_counts, n_samples=n_samples, impurity=impurity)

    feat, thresh, _ = split
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    n_L, n_R = np.sum(left_mask), np.sum(right_mask)

    if n_L < min_samples_leaf or n_R < min_samples_leaf:
        return Node(value=class_counts, n_samples=n_samples, impurity=impurity)

    # Final per-class check on resulting children
    if min_leaf_class_1 > 0:
        c1_left = np.sum(y[left_mask] == 1)
        c1_right = np.sum(y[right_mask] == 1)
        if c1_left < min_leaf_class_1 or c1_right < min_leaf_class_1:
            return Node(value=class_counts, n_samples=n_samples, impurity=impurity)

    sw_left = sample_weights[left_mask] if sample_weights is not None else None
    sw_right = sample_weights[right_mask] if sample_weights is not None else None

    left = build_tree(X[left_mask], y[left_mask], sw_left, depth + 1,
                      max_depth, min_samples_split, min_samples_leaf,
                      min_impurity_decrease, max_features, rng, criterion,
                      min_leaf_class_1)
    right = build_tree(X[right_mask], y[right_mask], sw_right, depth + 1,
                       max_depth, min_samples_split, min_samples_leaf,
                       min_impurity_decrease, max_features, rng, criterion,
                       min_leaf_class_1)

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


# ---------- Macro F1-based pruning ----------

def _node_per_class_errors(node):
    """Return per-class errors if this node were a leaf (majority vote)."""
    total = node.value.sum()
    if total == 0:
        return np.array([0, 0])
    counts = node.value
    majority = np.argmax(counts)
    errors = np.zeros(2)
    for c in [0, 1]:
        if c != majority:
            errors[c] = counts[c]
    return errors


def _node_balanced_error(node, total_per_class):
    """Balanced error: average per-class error rate (Macro F1 style)."""
    errors = _node_per_class_errors(node)
    balanced = 0.0
    for c in [0, 1]:
        if total_per_class[c] > 0:
            balanced += errors[c] / total_per_class[c]
    return balanced / 2.0


def _subtree_per_class_errors(node):
    """Total per-class errors across all leaves in this subtree."""
    if node.is_leaf():
        return _node_per_class_errors(node)
    left_errors = _subtree_per_class_errors(node.left)
    right_errors = _subtree_per_class_errors(node.right)
    return left_errors + right_errors


def _subtree_balanced_error(node, total_per_class):
    """Balanced error for the full subtree."""
    errors = _subtree_per_class_errors(node)
    balanced = 0.0
    for c in [0, 1]:
        if total_per_class[c] > 0:
            balanced += errors[c] / total_per_class[c]
    return balanced / 2.0


def _prune_node_f1(node, total_per_class, ccp_alpha):
    """Bottom-up cost-complexity pruning with Macro F1-based cost."""
    if node.is_leaf():
        return node

    node.left = _prune_node_f1(node.left, total_per_class, ccp_alpha)
    node.right = _prune_node_f1(node.right, total_per_class, ccp_alpha)

    n_leaves = count_leaves(node)
    if n_leaves <= 1:
        return node

    r_t = _node_balanced_error(node, total_per_class)
    r_Tt = _subtree_balanced_error(node, total_per_class)

    if r_Tt > 0:
        per_leaf_gain = (r_t - r_Tt) / (n_leaves - 1)
        if per_leaf_gain <= ccp_alpha:
            return Node(value=node.value, n_samples=node.n_samples, impurity=node.impurity)

    return node


class CARTDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, ccp_alpha=0.0, max_features=None,
                 class_weights=None, random_seed=42, criterion="gini",
                 min_leaf_class_1=0, f1_pruning=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.max_features = max_features
        self.class_weights = class_weights
        self.random_seed = random_seed
        self.criterion = criterion
        self.min_leaf_class_1 = min_leaf_class_1
        self.f1_pruning = f1_pruning
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

    def _total_per_class(self, y):
        counts = np.bincount(y, minlength=2)
        return counts.astype(np.float64)

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
            criterion=self.criterion,
            min_leaf_class_1=self.min_leaf_class_1,
        )

        if self.ccp_alpha > 0:
            if self.f1_pruning:
                total_pc = self._total_per_class(y)
                self.tree = _prune_node_f1(self.tree, total_pc, self.ccp_alpha)
            else:
                # Fallback: standard misclassification-rate pruning
                self.tree = _prune_node_std(self.tree, X, y, self.ccp_alpha)  # noqa: F821  ponytail: old fn, drop when f1_pruning proven

        return self

    def optimize_threshold(self, X, y, search_range=None):
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


# Keep old standard pruning as reference for compatibility
def _prune_node_std(node, X, y, ccp_alpha):
    """Standard CCP pruning with misclassification rate."""
    if node.is_leaf():
        return node
    node.left = _prune_node_std(node.left, X, y, ccp_alpha)
    node.right = _prune_node_std(node.right, X, y, ccp_alpha)

    n_total = X.shape[0]
    total = node.value.sum()
    r_t = (total - node.value.max()) / n_total if total > 0 else 0
    errors = _std_subtree_errors(node)
    r_Tt = errors / n_total
    n_leaves = count_leaves(node)

    if n_leaves > 1 and r_Tt > 0:
        per_leaf_gain = (r_t - r_Tt) / (n_leaves - 1)
        if per_leaf_gain <= ccp_alpha:
            return Node(value=node.value, n_samples=node.n_samples, impurity=node.impurity)
    return node


def _std_subtree_errors(node):
    if node.is_leaf():
        total = node.value.sum()
        return total - node.value.max() if total > 0 else 0
    return _std_subtree_errors(node.left) + _std_subtree_errors(node.right)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dataset_loader import load_dataset

    base = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    data = load_dataset(base)
    X, y = data["X_train"], data["y_train"]

    # Train/val split for fair comparison
    rng = np.random.RandomState(42)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    sp = int(0.85 * len(y))
    X_tr, y_tr = X[idx[:sp]], y[idx[:sp]]
    X_val, y_val = X[idx[sp:]], y[idx[sp:]]

    print("=== CART Variant Comparison (val F1) ===")
    variants = [
        ("Baseline (Gini)", CARTDecisionTree(max_depth=10, min_samples_leaf=5, random_seed=42)),

        ("Twoing criterion", CARTDecisionTree(max_depth=10, min_samples_leaf=5,
                                              criterion="twoing", random_seed=42)),

        ("min_leaf_class_1=5", CARTDecisionTree(max_depth=10, min_samples_leaf=5,
                                                min_leaf_class_1=5, random_seed=42)),

        ("F1 pruning (ccp=0.001)", CARTDecisionTree(max_depth=10, min_samples_leaf=5,
                                                     ccp_alpha=0.001, f1_pruning=True, random_seed=42)),

        ("Twoing + min_leaf_c1 + F1 prune", CARTDecisionTree(
            max_depth=10, min_samples_leaf=5, criterion="twoing",
            min_leaf_class_1=5, ccp_alpha=0.001, f1_pruning=True, random_seed=42)),
    ]

    best_f1 = -1
    best_name = ""
    for name, model in variants:
        model.fit(X_tr, y_tr)
        opt_t, val_f1 = model.optimize_threshold(X_val, y_val)
        leaves = count_leaves(model.tree)
        print(f"  {name:<38}: val_F1={val_f1:.4f}, opt_t={opt_t:.3f}, leaves={leaves}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_name = name

    print(f"\nBest: {best_name} (val_F1={best_f1:.4f})")
