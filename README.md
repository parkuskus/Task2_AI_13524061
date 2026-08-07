# Task2_AI_13524061

Repository for Task #2 Seleksi Laboratorium Intelegensi Buatan.

## Struktur Repository

```
Task2_AI_13524061/
├── src/
│   ├── local_search/         # PoC Local Search
│   └── dtl_lr_svm/           # Implementasi DTL, LR, SVM (modular)
│       ├── main.py           # Runner utama ketiga algoritma
│       ├── best_cart.py      # Submission generator (CART final)
│       ├── models/           # Implementasi algoritma from-scratch
│       │   ├── cart.py       # CART (Gini, Twoing, CCP, F1 pruning)
│       │   ├── logreg.py     # Logistic Regression (BCE + Adam)
│       │   ├── svm.py        # Linear SVM (Hinge + Adam)
│       │   └── adaboost.py   # AdaBoost (bonus)
│       └── utils/            # Utility
│           ├── loader.py     # Data loading + preprocessing
│           ├── eda.py        # Exploratory Data Analysis
│           └── compare.py    # Perbandingan from-scratch vs sklearn
│           └── compare.py    # Perbandingan from-scratch vs sklearn
├── notebooks/                # Notebook eksperimen (local search)
├── docs/
│   └── Write-Up/
│       └── Kaggle_Writeup.tex
├── extra/                    # Data tambahan & submission
│   ├── cart_submission_hyperparam.csv
│   └── submission_cart.csv
└── README.md
```

## Cara Menjalankan

```bash
# Generate submission (CART best config)
python src/dtl_lr_svm/best_cart.py

# Runner ketiga algoritma (DTL, LR, SVM)
python src/dtl_lr_svm/main.py

# EDA
python src/dtl_lr_svm/utils/eda.py

# Perbandingan with scikit-learn
python src/dtl_lr_svm/utils/compare.py
```

## Kaggle Competition

**Loan Acceptance Prediction** — Macro F1-score evaluation.

- `train.csv`: 36.000 rows (28k class 0, 8k class 1)
- `test.csv`: 9.000 rows
- Features: demografis, finansial, riwayat kredit
- Target: `loan_status` (0=ditolak, 1=disetujui)

## Deliverables

1. **PDF Gabungan** (`docs/Task2_AI_13524061.pdf`)
   - Spesifikasi Local Search
   - Write-up DTL, LR, SVM (max 5 halaman)
2. **Repository GitHub** dengan struktur di atas
3. **Submission Kaggle** minimal 1× dengan model from-scratch
