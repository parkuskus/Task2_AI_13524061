"""Ablasi: perbandingan from-scratch vs scikit-learn (DTL, LogReg, SVM).
"""

import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.loader import load_dataset
from models.cart import CARTDecisionTree


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
from models.logreg import LogisticRegressionScratch
from models.svm import LinearSVMScratch


def print_report(name, y_true, y_pred):
    acc = np.mean(y_true == y_pred)
    f1 = macro_f1_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  {cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Rejected (0)", "Approved (1)"]))


def main():
    base = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset")
    data = load_dataset(base)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # --- From-scratch ---
    print("\n\n" + "="*60)
    print("  FROM-SCRATCH MODELS")
    print("="*60)

    scratch_cart = CARTDecisionTree(
        max_depth=10, min_samples_split=10, min_samples_leaf=5,
        random_seed=42)
    scratch_cart.fit(X_train, y_train)
    print_report("CART (from-scratch)", y_train, scratch_cart.predict(X_train))

    scratch_lr = LogisticRegressionScratch(learning_rate=0.01, n_iterations=5000,
                                          lambda_l2=0.01, class_weight="balanced",
                                          optimizer="adam")
    scratch_lr.fit(X_train, y_train)
    print_report("LogReg (from-scratch)", y_train, scratch_lr.predict(X_train))

    scratch_svm = LinearSVMScratch(C=1.0, learning_rate=0.01, n_iterations=5000,
                                   class_weight="balanced", optimizer="adam")
    scratch_svm.fit(X_train, y_train)
    print_report("SVM (from-scratch)", y_train, scratch_svm.predict(X_train))

    # --- Scikit-learn ---
    print("\n\n" + "="*60)
    print("  SCIKIT-LEARN MODELS")
    print("="*60)

    sk_cart = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
    sk_cart.fit(X_train, y_train)
    print_report("CART (sklearn)", y_train, sk_cart.predict(X_train))

    sk_lr = LogisticRegression(max_iter=5000, random_state=42)
    sk_lr.fit(X_train, y_train)
    print_report("LogReg (sklearn)", y_train, sk_lr.predict(X_train))

    sk_svm = LinearSVC(max_iter=5000, random_state=42)
    sk_svm.fit(X_train, y_train)
    print_report("SVM (sklearn)", y_train, sk_svm.predict(X_train))

    # --- Summary ---
    print("\n\n" + "="*60)
    print("  SUMMARY (Train)")
    print("="*60)
    print(f"  {'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*45}")
    models = [
        ("CART (scratch)", scratch_cart.predict(X_train)),
        ("CART (sklearn)", sk_cart.predict(X_train)),
        ("LogReg (scratch)", scratch_lr.predict(X_train)),
        ("LogReg (sklearn)", sk_lr.predict(X_train)),
        ("SVM (scratch)", scratch_svm.predict(X_train)),
        ("SVM (sklearn)", sk_svm.predict(X_train)),
    ]
    for name, preds in models:
        acc = np.mean(y_train == preds)
        f1 = macro_f1_score(y_train, preds)
        print(f"  {name:<25} {acc:>10.4f} {f1:>10.4f}")


if __name__ == "__main__":
    main()
