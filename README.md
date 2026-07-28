# Task2_AI_13524061

Repository for Task #2 Seleksi Laboratorium Intelegensi Buatan.

## Struktur Repository

```
Task2_AI_13524061/
├── src/
│   ├── local_search/     # PoC Local Search
│   └── dtl_lr_svm/       # Implementasi DTL, LR, SVM
├── notebooks/
│   ├── local_search/     # Notebook eksperimen Local Search
│   └── dtl_lr_svm/       # Notebook eksperimen DTL, LR, SVM
├── docs/
│   └── Task2_AI_13524061.pdf  # PDF gabungan spesifikasi & write-up
├── .gitignore
├── LICENSE
└── README.md
```

## Cara Menjalankan

```bash
# Local Search PoC
cd src/local_search
python main.py

# DTL, LR, SVM
cd src/dtl_lr_svm
python main.py
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
