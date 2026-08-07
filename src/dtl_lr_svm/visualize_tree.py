"""CART Decision Tree Visualization — bonus DTL (3).

Renders the tree structure with matplotlib: split conditions, Gini, samples, class distribution.
Saves as PDF and PNG in visualization/ folder.
"""
import os, sys, csv, numpy as np
import matplotlib.pyplot as plt

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
    impurity = gini(y)
    node = Node(value=counts)
    node.n_samples = len(y); node.impurity = impurity
    if len(np.unique(y)) == 1: return node
    if max_depth is not None and depth >= max_depth: return node
    if len(y) < min_split: return node
    split = best_split(X, y)
    if split is None: return node
    f, t, _ = split; L = X[:, f] <= t; R = ~L
    if L.sum() < min_leaf or R.sum() < min_leaf: return node
    node.feature = f; node.threshold = t
    node.left = build(X[L], y[L], depth+1, max_depth, min_leaf, min_split)
    node.right = build(X[R], y[R], depth+1, max_depth, min_leaf, min_split)
    return node


def draw_tree(node, ax, x, y, dx, dy, feature_names, max_depth=4, depth=0):
    if node.is_leaf() or depth >= max_depth:
        c0, c1 = int(node.value[0]), int(node.value[1])
        total = c0 + c1
        color = '#c8e6c9' if c1 > c0 else '#ffcdd2'
        edge = '#2e7d32' if c1 > c0 else '#c62828'
        text = f"Leaf\nn = {total}\n0: {c0}   1: {c1}"
        ax.text(x, y, text, ha='center', va='center', fontsize=7,
                family='monospace', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                          edgecolor=edge, linewidth=1.5))
        return

    c0, c1 = int(node.value[0]), int(node.value[1])
    total = c0 + c1
    imp = node.impurity
    feat_name = feature_names[node.feature].replace("_", "\n")

    color = '#c8e6c9' if c1 > c0 else '#ffcdd2'
    edge = '#1565c0'
    text = f"[ {feat_name} ]\n<= {node.threshold:.3f}\nGini = {imp:.3f}  |  n = {total}"
    ax.text(x, y, text, ha='center', va='center', fontsize=7.5,
            family='monospace', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=color,
                      edgecolor=edge, linewidth=1.8))

    xl = x - dx; xr = x + dx; y_child = y - dy
    # Lines drawn behind boxes — extend enough to be hidden by bbox
    ax.plot([x, xl], [y-0.35, y_child+0.35], '-', color='#555555', lw=1.5, alpha=0.8, zorder=0)
    ax.plot([x, xr], [y-0.35, y_child+0.35], '-', color='#555555', lw=1.5, alpha=0.8, zorder=0)

    draw_tree(node.left, ax, xl, y_child, dx*0.55, dy, feature_names, max_depth, depth+1)
    draw_tree(node.right, ax, xr, y_child, dx*0.55, dy, feature_names, max_depth, depth+1)


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    cat_cols = ["person_gender", "person_home_ownership", "previous_loan_defaults_on_file"]

    th, tr = load_csv(os.path.join(base_dir, "train.csv"))
    for row in tr:
        row[th.index("person_age")] = str(np.clip(float(row[th.index("person_age")]), 0, 100))

    X, y, fn, cms = build_feature_matrix(th, tr, cat_cols, "loan_status")
    m = X.mean(0); s = X.std(0); s[s == 0] = 1.0
    Xs = (X - m) / s

    tree = build(Xs, y, max_depth=19, min_leaf=8, min_split=100)

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.set_xlim(-17, 17); ax.set_ylim(-2, 7)
    ax.axis('off')

    draw_tree(tree, ax, 0, 6.5, 6.5, 1.0, fn, max_depth=4)

    ax.set_title("CART Decision Tree (depth 19, top 4 levels shown)\n"
                 "Blue border = internal node | Color = majority class (red=reject, green=approve)",
                 fontsize=12, family='monospace', pad=15)

    viz_dir = os.path.join(os.path.dirname(__file__), "visualization")
    os.makedirs(viz_dir, exist_ok=True)

    for ext in ["pdf", "png"]:
        out = os.path.join(viz_dir, f"cart_tree.{ext}")
        plt.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.05, facecolor='white')
        print(f"Saved: {out}")
    plt.close()
