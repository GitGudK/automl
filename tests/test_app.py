"""
Quick test to verify app structure without running full optimization
"""

import sys
import os

# Add parent directory to path to import app module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import xgboost as xgb
        print("✓ All core dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install dependencies with:")
        print("  pip install -r requirements.txt")
        return False


def test_app_structure():
    """Test that app classes are properly structured"""
    try:
        import generate_features as app
        print("✓ App module imported successfully")

        # Check classes exist
        assert hasattr(app, 'FeatureEngineering'), "FeatureEngineering class not found"
        assert hasattr(app, 'ModelOptimizer'), "ModelOptimizer class not found"
        assert hasattr(app, 'AutomatedFeatureOptimizer'), "AutomatedFeatureOptimizer class not found"
        print("✓ All required classes found")

        # Check methods exist
        fe_methods = ['create_temporal_features', 'create_statistical_features',
                      'create_interaction_features', 'create_aggregated_features']
        for method in fe_methods:
            assert hasattr(app.FeatureEngineering, method), f"{method} not found"
        print("✓ All feature engineering methods found")

        mo_methods = ['get_model_configs', 'evaluate_model', 'train_and_evaluate_all']
        for method in mo_methods:
            assert hasattr(app.ModelOptimizer, method), f"{method} not found"
        print("✓ All model optimizer methods found")

        afo_methods = ['load_data', 'create_base_features', 'run_feature_iteration',
                       'run_optimization', 'report_best_results', 'save_results']
        for method in afo_methods:
            assert hasattr(app.AutomatedFeatureOptimizer, method), f"{method} not found"
        print("✓ All orchestrator methods found")

        return True
    except Exception as e:
        print(f"✗ Structure test failed: {e}")
        return False


def test_data_exists():
    """Test that required data file exists"""
    # Get path relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data/data_science_project_data.csv")
    if os.path.exists(data_path):
        print(f"✓ Data file found: data/data_science_project_data.csv")
        return True
    else:
        print(f"✗ Data file not found: data/data_science_project_data.csv")
        return False


if __name__ == "__main__":
    print("="*60)
    print("AUTOMATED FEATURE ENGINEERING APP - TESTS")
    print("="*60)
    print()

    results = []

    print("Test 1: Checking dependencies...")
    results.append(test_imports())
    print()

    if results[0]:  # Only continue if imports work
        print("Test 2: Checking app structure...")
        results.append(test_app_structure())
        print()

        print("Test 3: Checking data file...")
        results.append(test_data_exists())
        print()

    print("="*60)
    if all(results):
        print("✓ ALL TESTS PASSED - App is ready to run!")
        print("\nTo run the optimization:")
        print("  python generate_features.py")
    else:
        print("✗ SOME TESTS FAILED - Please fix issues above")
    print("="*60)
