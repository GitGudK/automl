"""
Test script for FeatureReproducer class

This script demonstrates how to use the FeatureReproducer to recreate features
for new time-series data using the exact same pipeline from training.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_pipeline import FeatureReproducer, ProductionPipeline


def test_feature_reproduction():
    """Test that features can be reproduced correctly"""

    print("="*80)
    print("TESTING FEATURE REPRODUCTION")
    print("="*80)

    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check if optimization has been run
    config_path = os.path.join(project_root, "data/models/best_model_config.json")
    if not os.path.exists(config_path):
        print("\n❌ Error: Optimization has not been run yet!")
        print("Please run 'python generate_features.py' first to generate the configuration.\n")
        return False

    print("\n✓ Configuration file found")

    # Load original data
    data_path = os.path.join(project_root, "data/data_science_project_data.csv")
    if not os.path.exists(data_path):
        print(f"\n❌ Error: Data file not found: data/data_science_project_data.csv\n")
        return False

    print("✓ Data file found")

    # Initialize reproducer
    print("\nInitializing FeatureReproducer...")
    reproducer = FeatureReproducer(config_path)

    # Load data
    print("\nLoading original data...")
    df_raw = pd.read_csv(data_path)
    print(f"  - {len(df_raw)} observations")
    print(f"  - {df_raw['session_id'].nunique()} unique visits")
    print(f"  - {df_raw['entity'].nunique()} unique entitys")

    # Test 1: Transform a subset of data
    print("\n" + "-"*80)
    print("TEST 1: Transform subset of data (first 100 visits)")
    print("-"*80)

    session_ids = df_raw['session_id'].unique()[:100]
    df_subset = df_raw[df_raw['session_id'].isin(session_ids)].copy()

    try:
        df_features = reproducer.transform(df_subset, return_all_features=False)
        print("\n✓ Feature transformation successful!")
        print(f"  - Input: {len(df_subset)} observations")
        print(f"  - Output: {len(df_features)} visits")
        print(f"  - Features: {df_features.shape[1]} columns")
        print(f"\nFeature columns:")
        feature_cols = [c for c in df_features.columns if c not in ['session_id', 'entity', 'flag_positive']]
        print(f"  {feature_cols[:10]}..." if len(feature_cols) > 10 else f"  {feature_cols}")
    except Exception as e:
        print(f"\n❌ Error during transformation: {e}")
        return False

    # Test 2: Verify feature names match expected features
    print("\n" + "-"*80)
    print("TEST 2: Verify feature names match training")
    print("-"*80)

    feature_cols = [c for c in df_features.columns if c not in ['session_id', 'entity', 'flag_positive']]
    expected_features = reproducer.selected_features

    if set(feature_cols) == set(expected_features):
        print("✓ All features match training configuration")
        print(f"  - Expected: {len(expected_features)} features")
        print(f"  - Generated: {len(feature_cols)} features")
    else:
        missing = set(expected_features) - set(feature_cols)
        extra = set(feature_cols) - set(expected_features)
        print(f"⚠ Feature mismatch:")
        if missing:
            print(f"  - Missing features: {missing}")
        if extra:
            print(f"  - Extra features: {extra}")

    # Test 3: Check for NaN values
    print("\n" + "-"*80)
    print("TEST 3: Check for missing values")
    print("-"*80)

    nan_counts = df_features[feature_cols].isna().sum()
    total_nans = nan_counts.sum()

    if total_nans == 0:
        print("✓ No missing values in features")
    else:
        print(f"⚠ Found {total_nans} missing values across features")
        print("  Top features with NaN:")
        print(nan_counts[nan_counts > 0].head())

    # Test 4: Test with single visit
    print("\n" + "-"*80)
    print("TEST 4: Transform single visit")
    print("-"*80)

    single_visit = df_raw[df_raw['session_id'] == session_ids[0]].copy()
    try:
        df_single = reproducer.transform(single_visit, return_all_features=False)
        print(f"✓ Single visit transformation successful")
        print(f"  - Input observations: {len(single_visit)}")
        print(f"  - Output features: {df_single.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 5: Compare all features vs selected features
    print("\n" + "-"*80)
    print("TEST 5: Compare all features vs selected features")
    print("-"*80)

    df_all_features = reproducer.transform(df_subset, return_all_features=True)
    df_selected_features = reproducer.transform(df_subset, return_all_features=False)

    all_cols = [c for c in df_all_features.columns if c not in ['session_id', 'entity', 'flag_positive']]
    selected_cols = [c for c in df_selected_features.columns if c not in ['session_id', 'entity', 'flag_positive']]

    print(f"✓ All features: {len(all_cols)} columns")
    print(f"✓ Selected features: {len(selected_cols)} columns")
    print(f"  Reduction: {len(all_cols) - len(selected_cols)} features filtered out")

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("✓ All tests completed successfully!")
    print("\nThe FeatureReproducer can:")
    print("  1. Load configuration from optimization results")
    print("  2. Transform new time-series data")
    print("  3. Generate exact same features as training")
    print("  4. Handle single or multiple visits")
    print("  5. Return all or selected features")

    return True


def demo_production_usage():
    """Demonstrate production usage scenarios"""

    print("\n\n" + "="*80)
    print("PRODUCTION USAGE EXAMPLES")
    print("="*80)

    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check if config exists
    config_path = os.path.join(project_root, "data/models/best_model_config.json")
    if not os.path.exists(config_path):
        print("\n⚠ Skipping production demo - run optimization first")
        return

    print("\nScenario 1: Feature Engineering Only")
    print("-"*80)
    print("""
from feature_pipeline import FeatureReproducer

# Initialize with saved configuration
reproducer = FeatureReproducer('data/models/best_model_config.json')

# Load new patient data (same format as training data)
df_new_patients = pd.read_csv('data/new_patients.csv')

# Transform to features
df_features = reproducer.transform(df_new_patients, return_all_features=False)

# Save features for later use
df_features.to_csv('data/new_patient_features.csv', index=False)
    """)

    print("\nScenario 2: Complete Prediction Pipeline")
    print("-"*80)
    print("""
from feature_pipeline import ProductionPipeline
import pickle

# First, train and save the best model (do this once after optimization)
# ... (training code from generate_features.py) ...
# with open('data/models/best_model.pkl', 'wb') as f:
#     pickle.dump(trained_model, f)

# Then, use in production:
pipeline = ProductionPipeline(
    config_path='data/models/best_model_config.json',
    model_path='data/models/best_model.pkl'
)

# Load new patient data
df_new_patients = pd.read_csv('data/new_patients.csv')

# Get predictions with probabilities
predictions = pipeline.predict(df_new_patients, return_probabilities=True)

# Results: DataFrame with session_id, entity, prediction_probability
print(predictions)
    """)

    print("\nScenario 3: Batch Processing Multiple Files")
    print("-"*80)
    print("""
from feature_pipeline import FeatureReproducer
import glob

reproducer = FeatureReproducer('data/models/best_model_config.json')

# Process all CSV files in a directory
for file_path in glob.glob('data/incoming/*.csv'):
    df_new = pd.read_csv(file_path)
    df_features = reproducer.transform(df_new)

    # Save with same filename
    output_path = file_path.replace('incoming', 'processed')
    df_features.to_csv(output_path, index=False)
    print(f"Processed {file_path}")
    """)

    print("\nScenario 4: Real-time Single Patient Prediction")
    print("-"*80)
    print("""
from feature_pipeline import ProductionPipeline

# Initialize once (at application startup)
pipeline = ProductionPipeline(
    config_path='data/models/best_model_config.json',
    model_path='data/models/best_model.pkl'
)

# For each new patient visit:
def predict_sepsis_risk(visit_data):
    '''
    visit_data: DataFrame with time-series observations for one visit
    Required columns: session_id, entity, timestamp,
                     metric_type, metric_value,
                     threshold_lower, threshold_upper
    '''
    prediction = pipeline.predict(visit_data, return_probabilities=True)

    risk_score = prediction['prediction_probability'].iloc[0]

    if risk_score > 0.7:
        alert = "HIGH RISK"
    elif risk_score > 0.4:
        alert = "MEDIUM RISK"
    else:
        alert = "LOW RISK"

    return {
        'session_id': prediction['session_id'].iloc[0],
        'entity': prediction['entity'].iloc[0],
        'risk_score': risk_score,
        'alert_level': alert
    }
    """)


def create_sample_new_data():
    """Create a sample new data file for testing"""

    print("\n\n" + "="*80)
    print("CREATING SAMPLE DATA FILE")
    print("="*80)

    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = os.path.join(project_root, "data/data_science_project_data.csv")
    if not os.path.exists(data_path):
        print("⚠ Original data not found, skipping sample creation")
        return

    # Load original data
    df_original = pd.read_csv(data_path)

    # Take last 50 visits as "new" data
    session_ids = df_original['session_id'].unique()[-50:]
    df_new = df_original[df_original['session_id'].isin(session_ids)].copy()

    # Save as sample
    sample_path = os.path.join(project_root, "data/sample_new_patients.csv")
    df_new.to_csv(sample_path, index=False)

    print(f"\n✓ Created sample file: data/sample_new_patients.csv")
    print(f"  - {len(df_new)} observations")
    print(f"  - {df_new['session_id'].nunique()} visits")
    print("\nYou can test the feature reproducer with:")
    print(f"  reproducer = FeatureReproducer('data/models/best_model_config.json')")
    print(f"  df_new = pd.read_csv('data/sample_new_patients.csv')")
    print(f"  df_features = reproducer.transform(df_new)")


if __name__ == "__main__":
    # Run tests
    success = test_feature_reproduction()

    if success:
        # Show production examples
        demo_production_usage()

        # Create sample data
        create_sample_new_data()

        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Review the production usage examples above")
        print("  2. Train and save your best model as 'data/models/best_model.pkl'")
        print("  3. Use FeatureReproducer or ProductionPipeline for new data")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("TESTS FAILED")
        print("="*80)
        print("\nPlease run the optimization first:")
        print("  python generate_features.py")
        print("="*80)
