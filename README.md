# 🔐 Crypto Binary Detector — XGBoost Model

**Runtime detection of cryptographic algorithms in Linux ELF binaries** using an XGBoost classifier trained on 80 static features extracted from compiled binaries.

## Model Performance

| Metric | Test Set | 5-Fold CV (mean ± std) |
|--------|----------|------------------------|
| **Accuracy** | 95.74% | 94.05% ± 3.52% |
| **F1 Score** | 93.33% | 90.71% ± 5.23% |
| **ROC AUC** | 95.83% | 98.17% ± 1.50% |
| **MCC** | 90.21% | 87.05% ± 7.19% |

**Overfitting gap: 2.1%** (Train acc 97.8% vs Test acc 95.7%) ✅

## Top 10 Discriminative Features

Selected via ensemble ranking (Mutual Information + XGBoost importance + Pearson Correlation):

| Rank | Feature | What it Captures |
|------|---------|------------------|
| 1 | `crypto_string_ratio` | Ratio of crypto-related strings (aes, sha, encrypt, etc.) |
| 2 | `sec_entropy_max` | Maximum section entropy — crypto code has distinctive high-entropy sections |
| 3 | `crypto_constant_hits` | YARA-like scan for AES S-box, SHA init vectors, DES tables |
| 4 | `text_rotate_density` | ROL/ROR instruction density — crypto uses heavy bitwise rotation |
| 5 | `sec_rodata_entropy` | .rodata section entropy — lookup tables (S-boxes) are high entropy |
| 6 | `sec_text_entropy` | .text section entropy — crypto round functions are near-uniform |
| 7 | `text_xor_density` | XOR instruction density — core crypto operation |
| 8 | `sec_entropy_std` | Entropy variation across sections — crypto binaries show wider range |
| 9 | `n_crypto_strings` | Count of crypto-related strings in binary |
| 10 | `avg_string_len` | Average string length — structural indicator |

## Anti-Overfitting Measures

- L1 regularization (reg_alpha=0.5)
- L2 regularization (reg_lambda=2.0)
- Shallow trees (max_depth=5, tuned via GridSearch)
- Row subsampling (subsample=0.8)
- Column subsampling (colsample_bytree=0.8, colsample_bylevel=0.8)
- Minimum child weight = 3
- Gamma / min split loss = 0.1
- Stratified 5-Fold CV
- Class imbalance handling (scale_pos_weight=2.25)
- Learning curve analysis

## Dataset

- **232 Linux ELF binaries** compiled from 29 unique C programs
- **9 crypto programs**: Custom AES (S-box), SHA-256, SHA-1, MD5, RC4, DES, ChaCha20, Blowfish, XOR cipher
- **20 non-crypto programs**: Sorting algorithms, data structures, string ops, file I/O, math, signal processing, etc.
- **8 compile variants** per source: -O0, -O2, -Os, -O3, PIE, stripped, static
- **80 features** extracted per binary using LIEF + custom analysis

## Usage

```python
import xgboost as xgb
import json

# Load model
model = xgb.XGBClassifier()
model.load_model("crypto_detector_xgboost.json")

# Load feature list
with open("top10_features.json") as f:
    features = json.load(f)["features"]

# Extract features from a binary (see scripts/feature_extraction.py)
# features_dict = extract_features("/path/to/binary")
# X = [features_dict[f] for f in features]
# prediction = model.predict([X])
```

## Files

- `crypto_detector_xgboost.json` — Trained XGBoost model
- `top10_features.json` — Feature names and ranking details
- `results_summary.json` — Full evaluation metrics
- `figures/` — Visualizations (SHAP, ROC, learning curves, etc.)
- `scripts/` — Full pipeline: dataset creation, feature extraction, training

## References

- EMBER feature engineering (Anderson & Roth, 2018) adapted for ELF
- LIEF library for binary parsing
- SHAP for model interpretability
