"""
Save Best Model for Production Use

This script retrains the best model configuration found during optimization
and saves it for production use with the FeatureReproducer pipeline.

Run this AFTER running generate_features.py to train the optimization.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from feature_pipeline import FeatureReproducer


def load_best_configuration():
    """Load the best model configuration from optimization results"""
    config_path = "data/models/best_model_config.json"

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "Best model configuration not found. Please run 'python generate_features.py' first."
        )

    with open(config_path, 'r') as f:
        config = json.load(f)

    print("Loaded best model configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  ROC AUC: {config['roc_auc']:.4f}")
    print(f"  Average Precision: {config['avg_precision']:.4f}")
    print(f"  Combined Score: {config['combined_score']:.4f}")

    return config


def get_model_class(model_name):
    """Get the model class based on name"""
    models = {
        'RandomForest': RandomForestClassifier,
        'ExtraTrees': ExtraTreesClassifier,
        'XGBoost': xgb.XGBClassifier,
        'GradientBoosting': GradientBoostingClassifier
    }
    return models.get(model_name)


def train_and_save_best_model(use_full_data=True):
    """
    Train the best model and save it for production use.

    Args:
        use_full_data: If True, train on all available data.
                      If False, use the same train/test split as optimization.
    """
    print("="*80)
    print("TRAINING AND SAVING BEST MODEL FOR PRODUCTION")
    print("="*80)

    # Load configuration
    config = load_best_configuration()

    # Load raw data
    data_path = "data/data_science_project_data.csv"
    print(f"\nLoading data from {data_path}...")
    df_raw = pd.read_csv(data_path)
    print(f"  - {len(df_raw)} observations")
    print(f"  - {df_raw['session_id'].nunique()} visits")

    # Use FeatureReproducer to create features with the same pipeline
    print("\nCreating features using FeatureReproducer...")
    reproducer = FeatureReproducer("data/models/best_model_config.json")

    df_features = reproducer.transform(df_raw, return_all_features=False)
    print(f"  - Created {df_features.shape[1]} columns")

    # Prepare data
    X = df_features.drop(columns=['session_id', 'entity', 'flag_positive'], errors='ignore')
    y = df_features['flag_positive']

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:")
    print(f"  - Negative cases: {(y == 0).sum()}")
    print(f"  - Positive cases: {(y == 1).sum()}")
    print(f"  - Class ratio: {(y == 1).sum() / len(y):.2%}")

    # Determine training data
    if use_full_data:
        print("\nTraining on full dataset...")
        X_train, y_train = X, y
    else:
        print("\nUsing 80/20 train/test split (same as optimization)...")
        split_idx = int(len(df_features) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        print(f"  - Training set: {len(X_train)} samples")
        print(f"  - Test set: {len(X_test)} samples")

    # Initialize model with best parameters
    model_class = get_model_class(config['model_name'])
    model = model_class(**config['params'])

    print(f"\nTraining {config['model_name']} model...")

    # Train model (handle XGBoost separately for sample weights)
    if config['model_name'] == 'XGBoost':
        sample_weights = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=0)
    else:
        model.fit(X_train, y_train)

    print("✓ Model training complete")

    # Evaluate on test set if not using full data
    if not use_full_data:
        from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        combined_score = roc_auc + avg_precision

        print("\nTest set performance:")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Combined Score: {combined_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    # Save model
    model_path = "data/models/best_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n✓ Model saved to {model_path}")

    # Save feature names for reference
    feature_names_path = "data/models/feature_names.json"
    feature_info = {
        'feature_names': list(X.columns),
        'num_features': len(X.columns),
        'feature_sets_used': reproducer.feature_sets
    }
    with open(feature_names_path, 'w') as f:
        json.dump(feature_info, f, indent=2)

    print(f"✓ Feature names saved to {feature_names_path}")

    # Create model info file
    model_info_path = "data/models/production_model_info.txt"
    with open(model_info_path, 'w') as f:
        f.write("PRODUCTION MODEL INFORMATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model Type: {config['model_name']}\n")
        f.write(f"Training Date: {pd.Timestamp.now()}\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Number of Features: {len(X.columns)}\n\n")

        f.write("Performance Metrics (from optimization):\n")
        f.write(f"  ROC AUC: {config['roc_auc']:.4f}\n")
        f.write(f"  Average Precision: {config['avg_precision']:.4f}\n")
        f.write(f"  Combined Score: {config['combined_score']:.4f}\n\n")

        f.write("Model Parameters:\n")
        for param, value in config['params'].items():
            f.write(f"  {param}: {value}\n")

        f.write("\nFeature Sets Used:\n")
        for fs in reproducer.feature_sets:
            f.write(f"  - {fs}\n")

        f.write(f"\nFiles Generated:\n")
        f.write(f"  - {model_path}\n")
        f.write(f"  - {feature_names_path}\n")
        f.write(f"  - {model_info_path}\n")

    print(f"✓ Model info saved to {model_info_path}")

    print("\n" + "="*80)
    print("MODEL READY FOR PRODUCTION!")
    print("="*80)
    print("\nYou can now use the model with ProductionPipeline:")
    print("\n  from feature_pipeline import ProductionPipeline")
    print("  pipeline = ProductionPipeline(")
    print("      config_path='data/models/best_model_config.json',")
    print("      model_path='data/models/best_model.pkl'")
    print("  )")
    print("  predictions = pipeline.predict(new_data)")
    print("="*80)

    return model


def test_saved_model():
    """Test that the saved model can be loaded and used"""
    print("\n" + "="*80)
    print("TESTING SAVED MODEL")
    print("="*80)

    from feature_pipeline import ProductionPipeline

    # Load pipeline
    print("\nLoading production pipeline...")
    pipeline = ProductionPipeline(
        config_path='data/models/best_model_config.json',
        model_path='data/models/best_model.pkl'
    )

    # Load test data
    print("\nLoading test data...")
    df_raw = pd.read_csv("data/data_science_project_data.csv")

    # Take a small subset for testing
    session_ids = df_raw['session_id'].unique()[:10]
    df_test = df_raw[df_raw['session_id'].isin(session_ids)].copy()

    # Make predictions
    print("\nMaking predictions on 10 test visits...")
    predictions = pipeline.predict(df_test, return_probabilities=True)

    print("\nPredictions:")
    print(predictions)

    print("\n✓ Model loaded and working correctly!")

    return predictions


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    use_full_data = '--full' in sys.argv

    if use_full_data:
        print("Training on FULL dataset (recommended for production)")
    else:
        print("Training on 80% of data (will evaluate on remaining 20%)")
        print("Use --full flag to train on all data")

    try:
        # Train and save model
        model = train_and_save_best_model(use_full_data=use_full_data)

        # Test saved model
        test_saved_model()

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run the optimization first:")
        print("  python generate_features.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
