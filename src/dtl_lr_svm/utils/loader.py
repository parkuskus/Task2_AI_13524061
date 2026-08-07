"""Universal dataset loader for DTL, Logistic Regression, and SVM."""

import numpy as np
import csv
import os


def load_csv(filepath):
    """Load CSV file and return header + rows."""
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def encode_categorical(rows, header, col_name, label_map=None):
    """Encode a categorical column into integers. Returns encoded rows + label_map."""
    col_idx = header.index(col_name)
    if label_map is None:
        unique_vals = sorted(set(row[col_idx] for row in rows))
        label_map = {v: i for i, v in enumerate(unique_vals)}
    encoded = [label_map[row[col_idx]] for row in rows]
    return encoded, label_map


def build_feature_matrix(header, rows, cat_cols, target_col=None):
    """Build X (numpy array) and optionally y (numpy array) from raw rows.

    All numeric columns are converted to float.
    Categorical columns are integer-encoded.
    """
    n = len(rows)
    num_cols = [c for c in header if c not in cat_cols and c != target_col and c != "person_id"]

    # Encode categoricals
    cat_encoded = {}
    cat_maps = {}
    for col in cat_cols:
        vals, m = encode_categorical(rows, header, col)
        cat_encoded[col] = vals
        cat_maps[col] = m

    # Build X
    col_order = num_cols + cat_cols
    X = np.zeros((n, len(col_order)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, col in enumerate(num_cols):
            idx = header.index(col)
            X[i, j] = float(row[idx]) if row[idx] != "" else 0.0
        for j, col in enumerate(cat_cols):
            X[i, len(num_cols) + j] = float(cat_encoded[col][i])

    # Build y if target exists
    y = None
    if target_col and target_col in header:
        t_idx = header.index(target_col)
        y = np.array([int(row[t_idx]) for row in rows], dtype=np.int64)

    feature_names = col_order
    return X, y, feature_names, cat_maps


def load_dataset(data_dir, train_file="train.csv", test_file="test.csv",
                 target_col="loan_status", cat_cols=None):
    """Load train/test split and return standardized arrays.

    Returns dict with X_train, y_train, X_test, feature_names, cat_maps.
    """
    if cat_cols is None:
        cat_cols = ["person_gender", "person_home_ownership", "previous_loan_defaults_on_file"]

    train_path = os.path.join(data_dir, train_file)
    test_path = os.path.join(data_dir, test_file)

    t_header, t_rows = load_csv(train_path)
    te_header, te_rows = load_csv(test_path)

    X_train, y_train, feature_names, cat_maps = build_feature_matrix(
        t_header, t_rows, cat_cols, target_col
    )

    # Encode test with same maps
    n_test = len(te_rows)
    num_cols = [c for c in te_header if c not in cat_cols and c != "person_id"]
    X_test = np.zeros((n_test, len(feature_names)), dtype=np.float64)
    for i, row in enumerate(te_rows):
        for j, col in enumerate(num_cols):
            idx = te_header.index(col)
            X_test[i, j] = float(row[idx]) if row[idx] != "" else 0.0
        for j, col in enumerate(cat_cols):
            idx = te_header.index(col)
            X_test[i, len(num_cols) + j] = float(cat_maps[col].get(row[idx], 0))

    # Standardize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "feature_names": feature_names,
        "cat_maps": cat_maps,
    }


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset")
    data = load_dataset(base)
    print(f"X_train: {data['X_train'].shape}, y_train: {data['y_train'].shape}")
    print(f"X_test: {data['X_test'].shape}")
    print(f"Features: {data['feature_names']}")
    print(f"Class dist: {np.bincount(data['y_train'])}")
