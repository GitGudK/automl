# Tests Directory

This directory contains all test files for the AutoML automated feature engineering and model optimization project.

## Test Files

### [test_app.py](test_app.py)
Tests the main application structure and dependencies.

**What it tests:**
- All required Python packages are installed
- App module can be imported
- All required classes exist (FeatureEngineering, ModelOptimizer, AutomatedFeatureOptimizer)
- All required methods exist in each class
- Data file is present

**Run:**
```bash
python tests/test_app.py
# or
python run_tests.py app
```

**Expected output:**
```
============================================================
AUTOMATED FEATURE ENGINEERING APP - TESTS
============================================================

Test 1: Checking dependencies...
✓ All core dependencies available

Test 2: Checking app structure...
✓ App module imported successfully
✓ All required classes found
✓ All feature engineering methods found
✓ All model optimizer methods found
✓ All orchestrator methods found

Test 3: Checking data file...
✓ Data file found: data/data_science_project_data.csv

============================================================
✓ ALL TESTS PASSED - App is ready to run!
============================================================
```

---

### [test_feature_pipeline.py](test_feature_pipeline.py)
Tests the feature reproduction pipeline for production use.

**What it tests:**
- Configuration files exist (requires optimization to be run first)
- FeatureReproducer can load configuration
- Features can be reproduced on new data
- Feature names match training configuration
- No missing values in output features
- Single visit transformation works
- All features vs selected features comparison

**Additional functions:**
- `demo_production_usage()` - Shows example production code
- `create_sample_new_data()` - Creates sample data file for testing

**Run:**
```bash
python tests/test_feature_pipeline.py
# or
python run_tests.py pipeline
```

**Prerequisites:**
- Must run `python generate_features.py` first to generate configuration files

**Expected output:**
```
================================================================================
TESTING FEATURE REPRODUCTION
================================================================================

✓ Configuration file found
✓ Data file found

Initializing FeatureReproducer...
Loaded configuration from iteration 9
Feature sets: ['temporal', 'statistical', 'aggregated']
Number of selected features: 30

[... test results ...]

================================================================================
TEST SUMMARY
================================================================================
✓ All tests completed successfully!

[... production usage examples ...]

✓ Created sample file: data/sample_new_patients.csv
```

---

## Running Tests

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Test Suite
```bash
# App tests only
python run_tests.py app

# Feature pipeline tests only
python run_tests.py pipeline
```

### Run Individual Test File
```bash
# From project root
python tests/test_app.py
python tests/test_feature_pipeline.py
```

## Test Workflow

### Before Optimization

Run `test_app.py` to verify setup:
```bash
python tests/test_app.py
```

This ensures:
- Dependencies are installed
- Data file exists
- App structure is correct

### After Optimization

Run `test_feature_pipeline.py` to verify production pipeline:
```bash
python tests/test_feature_pipeline.py
```

This ensures:
- Best model configuration was saved
- Features can be reproduced
- Production pipeline works correctly

## Adding New Tests

To add a new test file:

1. Create test file in `tests/` directory
2. Add path handling at the top:
   ```python
   import sys
   import os
   sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```
3. Use project root for file paths:
   ```python
   project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   data_path = os.path.join(project_root, "data/file.csv")
   ```
4. Update `run_tests.py` to include new test

## Troubleshooting

### Import Errors
If you get import errors, ensure you're running tests from the project root:
```bash
cd /path/to/automl
python tests/test_app.py
```

### Path Errors
Tests use relative paths from the project root. Always run from the project root directory.

### Missing Configuration
If `test_feature_pipeline.py` fails with "Configuration not found":
1. Run optimization first: `python generate_features.py`
2. This generates `models/best_model_config.json`
3. Then run the test again

### Missing Dependencies
If dependency tests fail:
```bash
pip install -r requirements.txt
```

## Test Coverage

Current test coverage:

**App Structure:**
- ✅ Dependency imports
- ✅ Class definitions
- ✅ Method definitions
- ✅ Data file existence

**Feature Pipeline:**
- ✅ Configuration loading
- ✅ Feature transformation
- ✅ Feature name validation
- ✅ Missing value handling
- ✅ Single vs batch processing
- ✅ All features vs selected features

**Not Yet Covered:**
- Model training accuracy
- End-to-end prediction validation
- Performance benchmarking
- Edge cases (empty data, single observation, etc.)

## Future Improvements

Potential test additions:
- Unit tests for individual feature engineering functions
- Integration tests for complete workflow
- Performance tests for large datasets
- Mock data tests that don't require real data
- Regression tests for model performance
- API endpoint tests (if REST API is implemented)

## CI/CD Integration

To integrate with CI/CD:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python run_tests.py
```

## Contact

For questions or issues with tests, refer to:
- [QUICK_START.md](../QUICK_START.md) for general usage
- [FEATURE_PIPELINE_GUIDE.md](../FEATURE_PIPELINE_GUIDE.md) for pipeline details
- [APP_USAGE.md](../APP_USAGE.md) for app documentation
