"""CART variant experiments: Gini, Twoing, F1 pruning, min_leaf_class_1.

Runs each variant on 85/15 validation split, selects best, retrains on full data,
and saves submission.
"""

import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import load_dataset
from dtl_cart import CARTDecisionTree, count_leaves


def macro_f1(y_true, y_pred):
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


def save_submission(person_ids, predictions, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["person_id", "loan_status"])
        for pid, pred in zip(person_ids, predictions):
            writer.writerow([pid, pred])


def load_test_ids(data_dir, test_file="test.csv"):
    test_path = os.path.join(data_dir, test_file)
    ids = []
    with open(test_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ids.append(int(row[0]))
    return ids


def main():
    base = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    data = load_dataset(base)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]

    rng = np.random.RandomState(42)
    idx = np.arange(len(y_train))
    rng.shuffle(idx)
    split = int(0.85 * len(y_train))
    train_idx, val_idx = idx[:split], idx[split:]
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    print(f"Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Class dist (train): {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "submissions")
    os.makedirs(out_dir, exist_ok=True)
    test_ids = load_test_ids(base)

    print("\n" + "=" * 60)
    print("  CART VARIANTS (Validation)")
    print("=" * 60)

    cw = {0: 1, 1: 3}

    variants = [
        ("Baseline (depth=10, leaf=5)", CARTDecisionTree(
            max_depth=10, min_samples_leaf=5, random_seed=42)),

        ("Baseline + minority rules", CARTDecisionTree(
            max_depth=10, min_samples_leaf=5, minority_rules=True, random_seed=42)),

        ("Rules + deeper (depth=12)", CARTDecisionTree(
            max_depth=12, min_samples_leaf=5, minority_rules=True, random_seed=42)),

        ("Rules + leaf=3", CARTDecisionTree(
            max_depth=10, min_samples_leaf=3, minority_rules=True, random_seed=42)),

        ("Rules + CW (1:3)", CARTDecisionTree(
            max_depth=10, min_samples_leaf=5, class_weights=cw,
            minority_rules=True, random_seed=42)),

        ("Rules + CW + F1 prune", CARTDecisionTree(
            max_depth=10, min_samples_leaf=5, ccp_alpha=0.00005,
            f1_pruning=True, class_weights=cw,
            minority_rules=True, random_seed=42)),

        ("Rules + min_leaf_c1=2", CARTDecisionTree(
            max_depth=10, min_samples_leaf=5, min_leaf_class_1=2,
            minority_rules=True, random_seed=42)),
    ]

    best_val_f1 = -1
    best_name = None
    best_model = None

    for name, model in variants:
        model.fit(X_tr, y_tr)
        opt_t, val_f1 = model.optimize_threshold(X_val, y_val)
        leaves = count_leaves(model.tree)
        print(f"  {name:<36}: val_F1={val_f1:.4f}, opt_t={opt_t:.3f}, leaves={leaves}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_name = name
            best_model = model

    print(f"\n=== SUBMISSION: {best_name} (val_F1={best_val_f1:.4f}) ===")

    final_model = CARTDecisionTree(
        max_depth=best_model.max_depth,
        min_samples_split=best_model.min_samples_split,
        min_samples_leaf=best_model.min_samples_leaf,
        ccp_alpha=best_model.ccp_alpha,
        criterion=best_model.criterion,
        class_weights=best_model.class_weights,
        f1_pruning=best_model.f1_pruning,
        min_leaf_class_1=best_model.min_leaf_class_1,
        random_seed=42,
    )
    final_model.fit(X_train, y_train)
    final_model.threshold = best_model.threshold

    train_pred = final_model.predict(X_train)
    train_acc = (train_pred == y_train).mean()
    train_f1 = macro_f1(y_train, train_pred)
    print(f"Full train accuracy: {train_acc:.4f}")
    print(f"Full train macro F1: {train_f1:.4f}")
    print(f"Threshold: {final_model.threshold:.3f}")

    test_pred = final_model.predict(X_test)
    print(f"Test predictions: {dict(zip(*np.unique(test_pred, return_counts=True)))}")

    sub_path = os.path.join(out_dir, "cart_best_submission.csv")
    save_submission(test_ids, test_pred, sub_path)
    print(f"Saved: {sub_path}")


if __name__ == "__main__":
    main()
