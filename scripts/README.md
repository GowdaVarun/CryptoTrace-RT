# CryptoTrace-RT Scripts

This directory contains the Python scripts for generating the dataset, extracting features, and training the XGBoost model for cryptographic algorithm detection in Linux ELF binaries.

## Requirements

Install the dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## How to Run the Scripts

The scripts are designed to be run in sequence.

### 1. Build the Dataset

```bash
python3 build_dataset.py
```

**What it does:** Compiles various C programs (both cryptographic and non-cryptographic) with different compiler optimization flags to generate a labeled dataset of ELF binaries. The C source codes are embedded within the script. It saves the source files to `../dataset/sources/`, binaries to `../dataset/binaries`, and generates a metadata file.

### 2. Extract Features

```bash
python3 feature_extraction.py
```

**What it does:** Parses the compiled ELF binaries to extract various static features including byte histograms, byte-entropy histograms, structural features, and imported cryptographic functions/libraries. It uses the `lief` library to parse ELF files. It reads `../dataset/binary_metadata.json` and outputs the extracted features to a CSV file at `../dataset/binary_features.csv`.

### 3. Extract Dynamic Features

```bash
python3 dynamic_analysis.py
```

**What it does:** Runs each compiled binary in a sandboxed harness using `strace` and `perf` to capture dynamic runtime behaviors (e.g., number of branch misses, instructions per cycle, executed system calls like `getrandom`). It outputs these metrics to `../dataset/dynamic_features.csv`.

### 4. Train the Model

```bash
python3 train_model.py
```

**What it does:** Loads the extracted features (`../dataset/binary_features.csv`), performs exploratory data analysis, feature selection (selecting the top 10 most discriminative features), and trains an XGBoost classifier using Stratified 5-Fold cross-validation and hyperparameter tuning. It saves the trained model, feature rankings, evaluation metrics, and SHAP interpretability plots to the `../results/` directory.
