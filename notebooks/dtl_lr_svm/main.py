"""Main runner for DTL (CART), Logistic Regression, SVM experiments."""

import os
import csv
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import load_dataset
from dtl_cart import CARTDecisionTree
from logreg import LogisticRegressionScratch
from svm import LinearSVMScratch


def macro_f1(y_true, y_pred):
    """Compute macro F1-score for binary classification."""
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
    """Save predictions in Kaggle submission format."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["person_id", "loan_status"])
        for pid, pred in zip(person_ids, predictions):
            writer.writerow([pid, pred])


def load_test_ids(data_dir, test_file="test.csv"):
    """Load person_id from test.csv for submission."""
    test_path = os.path.join(data_dir, test_file)
    ids = []
    with open(test_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ids.append(int(row[0]))
    return ids


def run_model(name, model, X_train, y_train, X_test, test_ids, out_dir,
              X_val=None, y_val=None):
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)

    if X_val is not None and y_val is not None and hasattr(model, 'optimize_threshold'):
        opt_t, opt_f1 = model.optimize_threshold(X_val, y_val)
        print(f"Optimized threshold: {opt_t:.3f} (val F1={opt_f1:.4f})")

    train_pred = model.predict(X_train)
    train_acc = (train_pred == y_train).mean()
    train_f1 = macro_f1(y_train, train_pred)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Train macro F1: {train_f1:.4f}")

    test_pred = model.predict(X_test)
    print(f"Test predictions: {dict(zip(*np.unique(test_pred, return_counts=True)))}")

    sub_path = os.path.join(out_dir, f"{name.lower().replace(' ', '_')}_submission.csv")
    save_submission(test_ids, test_pred, sub_path)
    print(f"Saved: {sub_path}")

    return train_acc, train_f1


def main():
    base = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    data = load_dataset(base)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]

    # Hold-out validation split for threshold optimization
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

    results = []

    # CART with weighted Gini + threshold optimization
    results.append(run_model("CART", CARTDecisionTree(
        max_depth=10, min_samples_split=10, min_samples_leaf=5,
        class_weights={0: 1.0, 1: 4.0}, random_seed=42),
        X_tr, y_tr, X_test, test_ids, out_dir,
        X_val=X_val, y_val=y_val))

    results.append(run_model("Logistic Regression",
                             LogisticRegressionScratch(learning_rate=0.1, n_iterations=5000),
                             X_tr, y_tr, X_test, test_ids, out_dir))

    results.append(run_model("SVM",
                             LinearSVMScratch(learning_rate=0.1, n_iterations=5000),
                             X_tr, y_tr, X_test, test_ids, out_dir))

    print("\n=== Summary ===")
    for i, (acc, f1) in enumerate(results):
        print(f"{['CART', 'LogReg', 'SVM'][i]:>20}: Acc={acc:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    import numpy as np
    main()
