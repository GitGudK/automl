"""
Debug script to reproduce exact ROC AUC from optimization
"""
import pandas as pd
import numpy as np
import random
import json
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
from feature_pipeline import FeatureReproducer

# Set seeds
np.random.seed(42)
random.seed(42)

print("="*80)
print("DEBUGGING ROC AUC DISCREPANCY")
print("="*80)

# Load config
with open('data/models/best_model_config.json', 'r') as f:
    config = json.load(f)

print(f"\nConfig:")
print(f"  ROC AUC (stored): {config['roc_auc']}")
print(f"  Model: {config['model_name']}")
print(f"  Feature sets: {config['feature_sets']}")
print(f"  Params: {config['params']}")

# Load data
print("\nLoading data...")
df_raw = pd.read_csv('data/data_science_project_data.csv')
print(f"  Raw data: {df_raw.shape}")

# Reproduce features
print("\nReproducing features...")
reproducer = FeatureReproducer('data/models/best_model_config.json')
df_features = reproducer.transform(df_raw, return_all_features=False)
print(f"  Feature data: {df_features.shape}")

# Split
split_idx = int(len(df_features) * 0.8)
print(f"\nSplit at index: {split_idx}")
print(f"  Train: {split_idx} samples")
print(f"  Test: {len(df_features) - split_idx} samples")

df_train = df_features.iloc[:split_idx]
df_test = df_features.iloc[split_idx:]

# Prepare data
X_train = df_train.drop(columns=['session_id', 'entity', 'flag_positive'], errors='ignore')
y_train = df_train['flag_positive']
X_test = df_test.drop(columns=['session_id', 'entity', 'flag_positive'], errors='ignore')
y_test = df_test['flag_positive']

print(f"\nData prepared:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"  Train positive rate: {y_train.mean():.4f}")
print(f"  Test positive rate: {y_test.mean():.4f}")

# Check for NaN
print(f"\nNaN check:")
print(f"  X_train has NaN: {X_train.isnull().any().any()}")
print(f"  X_test has NaN: {X_test.isnull().any().any()}")

# Train model
print(f"\nTraining XGBoost...")
sample_weights = compute_sample_weight('balanced', y_train)
print(f"  Sample weights shape: {sample_weights.shape}")
print(f"  Sample weights range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

model = xgb.XGBClassifier(**config['params'])
model.fit(X_train, y_train, sample_weight=sample_weights, verbose=0)
print("  Training complete")

# Predict
print(f"\nPredicting...")
y_pred_proba = model.predict_proba(X_test)[:, 1]
print(f"  Predictions shape: {y_pred_proba.shape}")
print(f"  Prediction range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")

# Calculate ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Stored ROC AUC:     {config['roc_auc']:.10f}")
print(f"Calculated ROC AUC: {roc_auc:.10f}")
print(f"Difference:         {abs(config['roc_auc'] - roc_auc):.10f}")
print(f"Match:              {'✓ YES' if abs(config['roc_auc'] - roc_auc) < 0.0001 else '✗ NO'}")
print("="*80)
