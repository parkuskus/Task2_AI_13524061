"""EDA: Explorasi distribusi dataset Loan Acceptance Prediction."""

import os
import sys
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(__file__))
from dataset_loader import load_dataset


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    base = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    data = load_dataset(base)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test = data["X_test"]
    feat_names = data["feature_names"]
    cat_maps = data["cat_maps"]

    print_section("DATASET OVERVIEW")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples:  {X_test.shape[0]}")
    print(f"Features:      {X_train.shape[1]}")
    print(f"Features list: {feat_names}")

    print_section("CLASS DISTRIBUTION (Train)")
    classes, counts = np.unique(y_train, return_counts=True)
    for c, n in zip(classes, counts):
        pct = n / len(y_train) * 100
        print(f"  Class {c}: {n:>6,} ({pct:.1f}%)")
    print(f"  Imbalance ratio: {counts[0]/counts[1]:.1f}:1")

    print_section("FEATURE STATISTICS (Train, raw scale)")
    # Load raw data for stats (before standardization)
    raw_path = os.path.join(base, "train.csv")
    with open(raw_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]

    num_cols = [c for c in feat_names if c not in cat_maps and c in header]
    print(f"  {'Feature':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}")
    print(f"  {'-'*80}")
    for col in num_cols:
        idx = header.index(col)
        vals = np.array([float(row[idx]) for row in rows if row[idx] != ""])
        print(f"  {col:<30} {vals.mean():>10.2f} {vals.std():>10.2f} {vals.min():>10.2f} {vals.max():>10.2f} {np.median(vals):>10.2f}")

    print_section("CATEGORICAL DISTRIBUTION")
    for col, mapping in cat_maps.items():
        idx = header.index(col)
        vals = [row[idx] for row in rows]
        unique, counts = np.unique(vals, return_counts=True)
        print(f"\n  {col}:")
        for u, n in zip(unique, counts):
            print(f"    {u:<15}: {n:>6,} ({n/len(vals)*100:.1f}%)")

    print_section("FEATURE-BY-CLASS MEAN (Standardized)")
    print(f"  {'Feature':<30} {'Class 0':>10} {'Class 1':>10} {'Diff':>10}")
    print(f"  {'-'*60}")
    for i, feat in enumerate(feat_names):
        mean_0 = X_train[y_train == 0, i].mean()
        mean_1 = X_train[y_train == 1, i].mean()
        diff = abs(mean_1 - mean_0)
        print(f"  {feat:<30} {mean_0:>10.3f} {mean_1:>10.3f} {diff:>10.3f}")

    print_section("CORRELATION WITH TARGET")
    print(f"  {'Feature':<30} {'Correlation':>12}")
    print(f"  {'-'*45}")
    corrs = np.corrcoef(X_train.T, y_train)[:-1, -1]
    sorted_idx = np.argsort(np.abs(corrs))[::-1]
    for i in sorted_idx:
        print(f"  {feat_names[i]:<30} {corrs[i]:>12.4f}")

    print_section("MISSING VALUES CHECK")
    missing_train = np.isnan(X_train).sum(axis=0)
    missing_test = np.isnan(X_test).sum(axis=0)
    total = missing_train.sum() + missing_test.sum()
    print(f"  Train missing: {missing_train.sum()}")
    print(f"  Test missing:  {missing_test.sum()}")
    if total == 0:
        print("  No missing values found.")


if __name__ == "__main__":
    main()
