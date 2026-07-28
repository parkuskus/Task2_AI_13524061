"""CART Decision Tree implemented from scratch (numpy only).

Binary splits, Gini impurity, cost-complexity pruning, min_samples_leaf, min_impurity_decrease.
"""

import numpy as np


class Node:
    def __init__(self, value, feature=None, threshold=None, left=None, right=None,
                 n_samples=0, impurity=0.0):
        self.value = value  # class distribution, always set
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


def best_split(X, y, min_impurity_decrease=0.0):
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return None

    parent_gini = gini_impurity(y)
    best_gain = min_impurity_decrease
    best_feat = None
    best_thresh = None

    for feat in range(n_features):
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

            weighted_gini = (n_left / n_samples) * gini_impurity(y[left_mask]) + \
                            (n_right / n_samples) * gini_impurity(y[right_mask])
            gain = parent_gini - weighted_gini

            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = thresh

    if best_gain <= min_impurity_decrease:
        return None
    return best_feat, best_thresh, best_gain


def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2,
               min_samples_leaf=1, min_impurity_decrease=0.0):
    n_samples = len(y)
    n_classes = len(np.unique(y))
    class_counts = np.bincount(y, minlength=2)
    impurity = gini_impurity(y)

    if n_classes == 1:
        return Node(value=class_counts, n_samples=n_samples, impurity=impurity)
    if max_depth is not None and depth >= max_depth:
        return Node(value=class_counts, n_samples=n_samples, impurity=impurity)
    if n_samples < min_samples_split:
        return Node(value=class_counts, n_samples=n_samples, impurity=impurity)

    split = best_split(X, y, min_impurity_decrease)
    if split is None:
        return Node(value=class_counts, n_samples=n_samples, impurity=impurity)

    feat, thresh, _ = split
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)

    if n_left < min_samples_leaf or n_right < min_samples_leaf:
        return Node(value=class_counts, n_samples=n_samples, impurity=impurity)

    left = build_tree(X[left_mask], y[left_mask], depth + 1,
                      max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease)
    right = build_tree(X[right_mask], y[right_mask], depth + 1,
                       max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease)

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
    """Count leaf nodes in subtree."""
    if node.is_leaf():
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


def node_misclassification(node):
    """Misclassification count for a node if it were a leaf (majority class)."""
    total = node.value.sum()
    if total == 0:
        return 0
    majority = node.value.max()
    return total - majority


def subtree_misclassification(node, X, y):
    """Misclassification count for the full subtree on given data."""
    y_pred = _predict_subtree(node, X)
    return (y_pred != y).sum()


def _predict_subtree(node, X):
    preds = []
    for i in range(X.shape[0]):
        dist = predict_single(node, X[i])
        total = dist.sum()
        if total == 0:
            preds.append(0)
        else:
            preds.append(int(dist[1] >= dist[0]))  # handles ties
    return np.array(preds, dtype=np.int64)


def _subtree_errors(node):
    """Total misclassification errors across all leaves in this subtree."""
    if node.is_leaf():
        return node_misclassification(node)
    return _subtree_errors(node.left) + _subtree_errors(node.right)


def _prune_node(node, X, y, ccp_alpha):
    """Bottom-up cost-complexity pruning."""
    if node.is_leaf():
        return node

    node.left = _prune_node(node.left, X, y, ccp_alpha)
    node.right = _prune_node(node.right, X, y, ccp_alpha)

    n_total = X.shape[0]
    r_t = node_misclassification(node) / n_total
    r_Tt = _subtree_errors(node) / n_total
    n_leaves = count_leaves(node)

    if n_leaves > 1 and r_Tt > 0:
        # cost-complexity metric: g = (R(t) - R(T_t)) / (|L_t| - 1)
        # prune if (r_t - r_Tt) / (n_leaves - 1) is small enough (dominated by alpha benefit)
        # Equivalent: prune when the per-leaf error reduction <= ccp_alpha
        per_leaf_gain = (r_t - r_Tt) / (n_leaves - 1)
        if per_leaf_gain <= ccp_alpha:
            return Node(value=node.value, n_samples=node.n_samples, impurity=node.impurity)

    return node


class CARTDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, ccp_alpha=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.tree = None

    def fit(self, X, y):
        self.tree = build_tree(
            X, y,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
        )

        if self.ccp_alpha > 0:
            self.tree = _prune_node(self.tree, X, y, self.ccp_alpha)

        return self

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
        return np.argmax(probas, axis=1)

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

    print("--- Basic (no pruning) ---")
    tree = CARTDecisionTree(max_depth=8, min_samples_split=10)
    tree.fit(X, y)
    print(f"Train acc: {tree.score(X, y):.4f}, leaves: {count_leaves(tree.tree)}")

    print("\n--- With min_samples_leaf + min_impurity_decrease ---")
    tree2 = CARTDecisionTree(max_depth=10, min_samples_split=20,
                             min_samples_leaf=10, min_impurity_decrease=0.0001)
    tree2.fit(X, y)
    print(f"Train acc: {tree2.score(X, y):.4f}, leaves: {count_leaves(tree2.tree)}")

    print("\n--- With CCP pruning ---")
    tree3 = CARTDecisionTree(max_depth=10, min_samples_split=10,
                             ccp_alpha=0.0005)
    tree3.fit(X, y)
    print(f"Train acc: {tree3.score(X, y):.4f}, leaves: {count_leaves(tree3.tree)}")
