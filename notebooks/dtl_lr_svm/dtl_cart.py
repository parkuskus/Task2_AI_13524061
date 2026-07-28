"""CART Decision Tree implemented from scratch (numpy only).

Binary splits, Gini impurity, optional max_depth, optional min_samples_split.
Supports predict() and predict_proba().
"""

import numpy as np


class Node:
    """Tree node: either a decision (feature, threshold, left, right) or a leaf (value)."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # class distribution [count_0, count_1, ...]

    def is_leaf(self):
        return self.value is not None


def gini_impurity(y):
    """Gini impurity for label array y."""
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def best_split(X, y):
    """Find best (feature, threshold) minimizing weighted Gini. Returns (feat, thresh, gain) or None."""
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return None

    parent_gini = gini_impurity(y)
    best_gain = -1.0
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

    if best_gain <= 0:
        return None
    return best_feat, best_thresh, best_gain


def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2):
    """Recursively build a CART tree."""
    n_samples = len(y)
    n_classes = len(np.unique(y))
    class_counts = np.bincount(y, minlength=2)

    # Leaf conditions
    if n_classes == 1:
        return Node(value=class_counts)
    if max_depth is not None and depth >= max_depth:
        return Node(value=class_counts)
    if n_samples < min_samples_split:
        return Node(value=class_counts)

    split = best_split(X, y)
    if split is None:
        return Node(value=class_counts)

    feat, thresh, _ = split
    left_mask = X[:, feat] <= thresh
    right_mask = ~left_mask

    left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples_split)
    right = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples_split)

    return Node(feature=feat, threshold=thresh, left=left, right=right)


def predict_single(node, x):
    """Predict class distribution for a single sample."""
    if node.is_leaf():
        return node.value
    if x[node.feature] <= node.threshold:
        return predict_single(node.left, x)
    else:
        return predict_single(node.right, x)


class CARTDecisionTree:
    """CART Decision Tree classifier."""

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = build_tree(X, y, max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        return self

    def predict_proba(self, X):
        """Return class probabilities [n_samples, n_classes]."""
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

    tree = CARTDecisionTree(max_depth=8, min_samples_split=10)
    tree.fit(data["X_train"], data["y_train"])

    train_acc = tree.score(data["X_train"], data["y_train"])
    print(f"Train accuracy: {train_acc:.4f}")
