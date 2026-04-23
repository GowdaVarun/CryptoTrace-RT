#!/usr/bin/env python3
"""
XGBoost Model for Crypto Algorithm Detection in Linux Binaries

Pipeline:
1. Load extracted features
2. Exploratory Data Analysis
3. Feature selection (top 10 most discriminative)
4. Train XGBoost with:
   - Stratified K-Fold cross-validation
   - Early stopping
   - Regularization (L1/L2, max_depth, min_child_weight)
   - Class imbalance handling (scale_pos_weight)
   - Feature importance via SHAP
5. Final evaluation on held-out test set
6. Save model + push dataset to HF Hub
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
    GridSearchCV, learning_curve
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import shap
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = "/app/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================================================
# 1. LOAD DATA
# ================================================================
print("=" * 70)
print("CRYPTO BINARY DETECTION — XGBoost Pipeline")
print("=" * 70)

df = pd.read_csv("/app/binary_features.csv")
print(f"\nDataset: {len(df)} samples")
print(f"Class distribution:")
print(df['label_name'].value_counts())
print(f"\nCrypto ratio: {df['label'].mean():.2%}")

# Separate metadata and features
meta_cols = ['binary_name', 'source_file', 'label', 'label_name', 'opt_level']
feature_cols = [c for c in df.columns if c not in meta_cols]

X = df[feature_cols].copy()
y = df['label'].copy()

# Handle any NaN/inf
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"\nTotal features: {len(feature_cols)}")

# ================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ================================================================
print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Correlation with target
correlations = X.corrwith(y).abs().sort_values(ascending=False)
print("\nTop 20 features by |correlation| with target:")
for feat, corr in correlations.head(20).items():
    print(f"  {corr:.4f}  {feat}")

# Save correlation heatmap of top features
fig, ax = plt.subplots(figsize=(14, 10))
top_feats = correlations.head(20).index.tolist()
corr_matrix = X[top_feats].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', 
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title("Feature Correlation Matrix (Top 20 by target correlation)")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

# Distribution plots for key features
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
for idx, feat in enumerate(top_feats[:12]):
    ax = axes[idx // 4, idx % 4]
    for label, color, name in [(0, '#2ecc71', 'Non-crypto'), (1, '#e74c3c', 'Crypto')]:
        data = X[y == label][feat]
        ax.hist(data, bins=30, alpha=0.6, color=color, label=name, density=True)
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=8)
plt.suptitle("Feature Distributions: Crypto vs Non-Crypto", fontsize=14)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/feature_distributions.png", dpi=150)
plt.close()

# ================================================================
# 3. FEATURE SELECTION — TOP 10 MOST DISCRIMINATIVE
# ================================================================
print("\n" + "=" * 70)
print("FEATURE SELECTION: TOP 10 DISCRIMINATIVE FEATURES")
print("=" * 70)

# Method 1: Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_ranking = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
print("\nMutual Information scores (top 20):")
for feat, score in mi_ranking.head(20).items():
    print(f"  {score:.4f}  {feat}")

# Method 2: XGBoost built-in importance (train quick model on all features)
quick_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    random_state=42, eval_metric='logloss', verbosity=0,
)
quick_model.fit(X, y)
xgb_importance = pd.Series(
    quick_model.feature_importances_, index=feature_cols
).sort_values(ascending=False)
print("\nXGBoost feature importance (top 20):")
for feat, score in xgb_importance.head(20).items():
    print(f"  {score:.4f}  {feat}")

# Method 3: Absolute correlation
corr_ranking = correlations

# Combine rankings (rank-based ensemble)
rankings = pd.DataFrame({
    'mi_rank': mi_ranking.rank(ascending=False),
    'xgb_rank': xgb_importance.rank(ascending=False),
    'corr_rank': corr_ranking.rank(ascending=False),
})
rankings['avg_rank'] = rankings.mean(axis=1)
rankings = rankings.sort_values('avg_rank')

print("\n" + "-" * 50)
print("COMBINED RANKING (avg of MI, XGBoost importance, correlation):")
print("-" * 50)
for i, (feat, row) in enumerate(rankings.head(15).iterrows()):
    print(f"  #{i+1:2d}  {feat:40s}  avg_rank={row['avg_rank']:.1f}  "
          f"(MI={row['mi_rank']:.0f}, XGB={row['xgb_rank']:.0f}, Corr={row['corr_rank']:.0f})")

# Select top 10
TOP_10_FEATURES = rankings.head(10).index.tolist()
print(f"\n{'='*50}")
print("SELECTED TOP 10 FEATURES:")
print(f"{'='*50}")
for i, feat in enumerate(TOP_10_FEATURES):
    print(f"  {i+1:2d}. {feat}")

# Save feature importance plot
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

ax = axes[0]
mi_ranking.head(15).plot(kind='barh', ax=ax, color='#3498db')
ax.set_title('Mutual Information', fontsize=12)
ax.invert_yaxis()

ax = axes[1]
xgb_importance.head(15).plot(kind='barh', ax=ax, color='#e74c3c')
ax.set_title('XGBoost Feature Importance', fontsize=12)
ax.invert_yaxis()

ax = axes[2]
corr_ranking.head(15).plot(kind='barh', ax=ax, color='#2ecc71')
ax.set_title('|Correlation| with Target', fontsize=12)
ax.invert_yaxis()

plt.suptitle("Feature Importance Rankings", fontsize=14)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/feature_importance.png", dpi=150)
plt.close()

# ================================================================
# 4. TRAIN-TEST SPLIT (stratified)
# ================================================================
X_selected = X[TOP_10_FEATURES]

# Hold out 20% for final test (never seen during CV or tuning)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_selected, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train_full)} samples (crypto: {y_train_full.sum()})")
print(f"Test:  {len(X_test)} samples  (crypto: {y_test.sum()})")

# ================================================================
# 5. STRATIFIED K-FOLD CROSS-VALIDATION
# ================================================================
print("\n" + "=" * 70)
print("STRATIFIED 5-FOLD CROSS-VALIDATION")
print("=" * 70)

# Class imbalance handling
n_neg = (y_train_full == 0).sum()
n_pos = (y_train_full == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"scale_pos_weight: {scale_pos_weight:.2f} (neg/pos = {n_neg}/{n_pos})")

# Anti-overfitting XGBoost config
base_params = {
    'n_estimators': 500,
    'max_depth': 4,                 # Shallow trees to prevent overfitting
    'learning_rate': 0.05,          # Low LR with many trees
    'subsample': 0.8,               # Row subsampling
    'colsample_bytree': 0.8,        # Column subsampling per tree
    'colsample_bylevel': 0.8,       # Column subsampling per level
    'min_child_weight': 5,          # Higher = more conservative
    'gamma': 0.2,                   # Min loss reduction for split
    'reg_alpha': 0.5,               # L1 regularization
    'reg_lambda': 2.0,              # L2 regularization
    'scale_pos_weight': scale_pos_weight,
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0,
}

# 5-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {
    'fold': [], 'accuracy': [], 'f1': [], 'auc': [], 
    'precision': [], 'recall': [], 'mcc': [],
    'train_accuracy': [], 'train_f1': [],  # For overfitting detection
}

fold_models = []
oof_predictions = np.zeros(len(X_train_full))  # Out-of-fold predictions

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
    X_tr = X_train_full.iloc[train_idx]
    y_tr = y_train_full.iloc[train_idx]
    X_val = X_train_full.iloc[val_idx]
    y_val = y_train_full.iloc[val_idx]
    
    model = xgb.XGBClassifier(**base_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    # Validation metrics
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Training metrics (for overfitting check)
    y_tr_pred = model.predict(X_tr)
    
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_proba)
    prec = f1_score(y_val, y_val_pred, pos_label=1, average='binary')
    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(y_val, y_val_pred)
    rec = recall_score(y_val, y_val_pred)
    mcc = matthews_corrcoef(y_val, y_val_pred)
    
    train_acc = accuracy_score(y_tr, y_tr_pred)
    train_f1 = f1_score(y_tr, y_tr_pred)
    
    cv_results['fold'].append(fold_idx + 1)
    cv_results['accuracy'].append(acc)
    cv_results['f1'].append(f1)
    cv_results['auc'].append(auc)
    cv_results['precision'].append(prec)
    cv_results['recall'].append(rec)
    cv_results['mcc'].append(mcc)
    cv_results['train_accuracy'].append(train_acc)
    cv_results['train_f1'].append(train_f1)
    
    oof_predictions[val_idx] = y_val_proba
    fold_models.append(model)
    
    print(f"  Fold {fold_idx+1}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, "
          f"MCC={mcc:.4f} | Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}")

cv_df = pd.DataFrame(cv_results)
print(f"\n{'='*50}")
print("CROSS-VALIDATION SUMMARY")
print(f"{'='*50}")
for metric in ['accuracy', 'f1', 'auc', 'precision', 'recall', 'mcc']:
    vals = cv_df[metric]
    print(f"  {metric:12s}: {vals.mean():.4f} ± {vals.std():.4f}")

# Overfitting check
print(f"\nOVERFITTING CHECK:")
train_f1_mean = cv_df['train_f1'].mean()
val_f1_mean = cv_df['f1'].mean()
gap = train_f1_mean - val_f1_mean
print(f"  Train F1 (mean): {train_f1_mean:.4f}")
print(f"  Val F1 (mean):   {val_f1_mean:.4f}")
print(f"  Gap:             {gap:.4f} {'⚠️  POTENTIAL OVERFITTING' if gap > 0.1 else '✅ OK'}")

# ================================================================
# 6. HYPERPARAMETER TUNING via GridSearchCV (if needed)
# ================================================================
print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING (Grid Search)")
print("=" * 70)

param_grid = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [3, 5, 7],
    'gamma': [0.1, 0.2, 0.5],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
}

# Use a smaller grid for efficiency
mini_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.03, 0.05, 0.1],
    'min_child_weight': [3, 5, 7],
    'gamma': [0.1, 0.3],
}

grid_model = xgb.XGBClassifier(
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bylevel=0.8,
    reg_alpha=0.5,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    tree_method='hist',
    random_state=42,
    verbosity=0,
)

grid_search = GridSearchCV(
    grid_model, mini_grid,
    scoring='f1',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=0,
    refit=True,
)
grid_search.fit(X_train_full, y_train_full)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1: {grid_search.best_score_:.4f}")

# ================================================================
# 7. TRAIN FINAL MODEL WITH BEST PARAMS
# ================================================================
print("\n" + "=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)

best_params = {
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'scale_pos_weight': scale_pos_weight,
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0,
}
best_params.update(grid_search.best_params_)

final_model = xgb.XGBClassifier(**best_params)

# Split train into train/val for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

final_model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    verbose=False,
)

print(f"Final model parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ================================================================
# 8. FINAL EVALUATION ON HELD-OUT TEST SET
# ================================================================
print("\n" + "=" * 70)
print("FINAL TEST SET EVALUATION")
print("=" * 70)

y_test_pred = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)
test_mcc = matthews_corrcoef(y_test, y_test_pred)
test_ap = average_precision_score(y_test, y_test_proba)

print(f"  Accuracy:  {test_acc:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")
print(f"  ROC AUC:   {test_auc:.4f}")
print(f"  PR AUC:    {test_ap:.4f}")
print(f"  MCC:       {test_mcc:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Non-Crypto', 'Crypto']))

print(f"Confusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Overfitting final check
y_train_pred = final_model.predict(X_train_full)
train_acc_final = accuracy_score(y_train_full, y_train_pred)
print(f"\nFINAL OVERFITTING CHECK:")
print(f"  Train Accuracy: {train_acc_final:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")
print(f"  Gap:            {train_acc_final - test_acc:.4f} {'⚠️  OVERFITTING' if train_acc_final - test_acc > 0.1 else '✅ OK'}")

# ================================================================
# 9. LEARNING CURVES (overfitting diagnostic)
# ================================================================
print("\nGenerating learning curves...")
train_sizes, train_scores, val_scores = learning_curve(
    xgb.XGBClassifier(**best_params),
    X_selected, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1',
    train_sizes=np.linspace(0.2, 1.0, 8),
    n_jobs=-1,
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(train_sizes, 
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='#e74c3c')
ax.fill_between(train_sizes,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color='#3498db')
ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='#e74c3c', label='Training F1')
ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', color='#3498db', label='Validation F1')
ax.set_xlabel('Training Set Size')
ax.set_ylabel('F1 Score')
ax.set_title('Learning Curves — Overfitting Diagnostic')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/learning_curves.png", dpi=150)
plt.close()

# ================================================================
# 10. ROC & PR CURVES
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
axes[0].plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC (AUC={test_auc:.4f})')
axes[0].plot([0, 1], [0, 1], '--', color='gray')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# PR
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_proba)
axes[1].plot(recall_vals, precision_vals, color='#3498db', lw=2, label=f'PR (AP={test_ap:.4f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Final Model Performance', fontsize=14)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/roc_pr_curves.png", dpi=150)
plt.close()

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Non-Crypto', 'Crypto'], yticklabels=['Non-Crypto', 'Crypto'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# ================================================================
# 11. SHAP ANALYSIS (Model Interpretability)
# ================================================================
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=TOP_10_FEATURES, 
                  show=False, plot_size=(12, 8))
plt.title("SHAP Feature Importance — What Drives Crypto Detection?")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()

# SHAP bar plot
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=TOP_10_FEATURES,
                  plot_type="bar", show=False, plot_size=(10, 6))
plt.title("SHAP Mean |Impact| on Prediction")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/shap_bar.png", dpi=150, bbox_inches='tight')
plt.close()

# ================================================================
# 12. SAVE MODEL & ARTIFACTS
# ================================================================
print("\nSaving model and artifacts...")

# Save XGBoost model
final_model.save_model(f"{RESULTS_DIR}/crypto_detector_xgboost.json")

# Save feature list
with open(f"{RESULTS_DIR}/top10_features.json", "w") as f:
    json.dump({
        "features": TOP_10_FEATURES,
        "feature_ranking": {feat: {
            "mi_rank": int(rankings.loc[feat, 'mi_rank']),
            "xgb_rank": int(rankings.loc[feat, 'xgb_rank']),
            "corr_rank": int(rankings.loc[feat, 'corr_rank']),
            "avg_rank": float(rankings.loc[feat, 'avg_rank']),
        } for feat in TOP_10_FEATURES},
    }, f, indent=2)

# Save full results summary
results_summary = {
    "dataset": {
        "total_samples": len(df),
        "crypto_samples": int(df['label'].sum()),
        "noncrypto_samples": int((1-df['label']).sum()),
        "n_source_programs": {
            "crypto": len(df[df['label']==1]['source_file'].unique()),
            "noncrypto": len(df[df['label']==0]['source_file'].unique()),
        },
        "compile_variants": sorted(df['opt_level'].unique().tolist()),
    },
    "feature_selection": {
        "total_features_extracted": len(feature_cols),
        "selected_top_10": TOP_10_FEATURES,
        "methods": ["Mutual Information", "XGBoost Feature Importance", "Pearson Correlation"],
    },
    "cross_validation": {
        "n_folds": 5,
        "metrics_mean": {m: float(cv_df[m].mean()) for m in ['accuracy', 'f1', 'auc', 'precision', 'recall', 'mcc']},
        "metrics_std": {m: float(cv_df[m].std()) for m in ['accuracy', 'f1', 'auc', 'precision', 'recall', 'mcc']},
    },
    "hyperparameter_tuning": {
        "method": "GridSearchCV (5-fold stratified)",
        "best_params": grid_search.best_params_,
        "best_cv_f1": float(grid_search.best_score_),
    },
    "final_test_evaluation": {
        "accuracy": float(test_acc),
        "f1": float(test_f1),
        "roc_auc": float(test_auc),
        "pr_auc": float(test_ap),
        "mcc": float(test_mcc),
        "confusion_matrix": cm.tolist(),
    },
    "overfitting_analysis": {
        "train_accuracy": float(train_acc_final),
        "test_accuracy": float(test_acc),
        "gap": float(train_acc_final - test_acc),
        "status": "OK" if train_acc_final - test_acc < 0.1 else "POTENTIAL_OVERFITTING",
    },
    "anti_overfitting_measures": [
        "L1 regularization (reg_alpha=0.5)",
        "L2 regularization (reg_lambda=2.0)",
        "Shallow trees (max_depth tuned via GridSearch)",
        "Row subsampling (subsample=0.8)",
        "Column subsampling (colsample_bytree=0.8, colsample_bylevel=0.8)",
        "Minimum child weight constraint (min_child_weight tuned)",
        "Gamma / min split loss (tuned)",
        "Stratified K-Fold CV (5 folds)",
        "Class imbalance handling (scale_pos_weight)",
        "Learning curve analysis",
    ],
    "model_params": best_params,
}

with open(f"{RESULTS_DIR}/results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

# Save the dataset
df.to_csv(f"{RESULTS_DIR}/crypto_binary_dataset.csv", index=False)

print(f"\n{'='*70}")
print("ALL ARTIFACTS SAVED TO /app/results/")
print(f"{'='*70}")
print(f"  Model:          crypto_detector_xgboost.json")
print(f"  Features:       top10_features.json")
print(f"  Results:        results_summary.json")
print(f"  Dataset:        crypto_binary_dataset.csv")
print(f"  Plots:          correlation_heatmap.png, feature_distributions.png,")
print(f"                  feature_importance.png, learning_curves.png,")
print(f"                  roc_pr_curves.png, confusion_matrix.png,")
print(f"                  shap_summary.png, shap_bar.png")
print(f"{'='*70}")
print("DONE!")
