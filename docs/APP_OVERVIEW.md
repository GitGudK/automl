# Automated Feature Engineering and Model Optimization App

## Summary

A comprehensive automated machine learning app that creates new features and optimizes models to maximize the sum of **ROC AUC** and **Average Precision (AP)** scores for sepsis prediction.

## What It Does

The app automatically:

1. **Generates 4 types of features:**
   - **Temporal**: Tracks changes over time (trends, volatility, first/last values, max jumps)
   - **Statistical**: Advanced statistics (median, quartiles, IQR, skewness, kurtosis, CV)
   - **Interactions**: Pairwise feature combinations (multiplications and divisions)
   - **Aggregated**: Cross-observation summary statistics

2. **Tests 10 different feature combinations:**
   - Baseline (original features only)
   - Each feature type individually
   - Various combinations
   - All features together

3. **Trains 4 model types with multiple configurations:**
   - Random Forest (3 configurations)
   - Extra Trees (2 configurations)
   - XGBoost (3 configurations)
   - Gradient Boosting (2 configurations)
   - **Total: 8 models per iteration**

4. **Evaluates and optimizes:**
   - Calculates ROC AUC and Average Precision for each model
   - Maximizes their sum (combined score)
   - Tracks the best performing configuration

5. **Saves comprehensive results:**
   - Complete experiment logs (JSON)
   - Best model configuration (JSON)
   - Human-readable summary (TXT)

## Key Features

### Intelligent Feature Engineering

- **Temporal analysis**: Captures how observations change over time
- **Statistical depth**: Goes beyond mean/std to include distribution shape
- **Feature interactions**: Discovers relationships between variables
- **Smart aggregation**: Summarizes patterns across all observations

### Automated Model Selection

Instead of manually testing models, the app:

- Tests multiple model families simultaneously
- Tries different hyperparameter combinations
- Uses class balancing for imbalanced data
- Maintains time-series integrity in train/test split

### Comprehensive Evaluation

Optimizes for the **sum of ROC AUC + AP**:
- **ROC AUC**: Measures ranking ability (how well the model separates classes)
- **Average Precision**: Measures precision-recall tradeoff (especially important for imbalanced data)
- **Combined**: Balances both metrics for robust model selection

## File Structure

```
automl/
├── generate_features.py                      # Main automated optimization app
├── src/
│   └── fe.py                   # Original feature engineering script
├── data/
│   └── data_science_project_data.csv
├── models/
│   ├── optimization_log.json           # Complete experiment history
│   ├── best_model_config.json          # Best configuration found
│   └── optimization_summary.txt        # Human-readable summary
├── requirements.txt            # Python dependencies
├── APP_OVERVIEW.md            # This file
└── APP_USAGE.md               # Detailed usage guide
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Tests (Optional)

```bash
python tests/test_app.py
```

### 3. Run Workflow

```bash
python generate_features.py
```

This will:
- Run 10 iterations testing different feature combinations
- Train 80 models total (8 models × 10 iterations)
- Take approximately 5-15 minutes depending on data size
- Output results to `models/` directory

## Example Output

```
================================================================================
AUTOMATED FEATURE ENGINEERING AND MODEL OPTIMIZATION
================================================================================

Loading raw data...
Loaded 50000 observations for 1000 visits

================================================================================
ITERATION 1: Testing feature combinations
================================================================================

Creating base features...
Created 245 base features
Total features before selection: 245

Performing feature selection...
Selected 30 features

Training set: 800 samples | Test set: 200 samples

Training models...
  RandomForest         | ROC AUC: 0.8512 | AP: 0.8334 | Combined: 1.6846
  ExtraTrees           | ROC AUC: 0.8445 | AP: 0.8298 | Combined: 1.6743
  XGBoost              | ROC AUC: 0.8723 | AP: 0.8611 | Combined: 1.7334
  ...

[... 9 more iterations ...]

================================================================================
OPTIMIZATION COMPLETE - BEST RESULTS
================================================================================

Best Combined Score: 1.8234
  ROC AUC: 0.9123
  Average Precision: 0.9111
  Model: XGBoost
  Iteration: 9

Best Parameters:
  n_estimators: 150
  max_depth: 8
  learning_rate: 0.05
  random_state: 42
  eval_metric: logloss

Feature Sets Used: ['temporal', 'statistical', 'aggregated']
Number of Features: 30

Results saved to models/
  - optimization_log.json
  - best_model_config.json
  - optimization_summary.txt
```

## Architecture

### Class Structure

**FeatureEngineering**
- Creates new feature sets
- Transforms raw data into engineered features
- Returns DataFrames ready for merging

**ModelOptimizer**
- Manages model training and evaluation
- Tracks best configurations
- Maintains complete results history

**AutomatedFeatureOptimizer**
- Main orchestrator
- Coordinates feature engineering and model training
- Runs iterations and saves results

### Workflow

```
┌─────────────────────────────────────────┐
│         Load Raw Data                   │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│    For Each Feature Combination:        │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │ 1. Create Base Features            │ │
│  │ 2. Add Selected Feature Sets       │ │
│  │ 3. Perform Feature Selection       │ │
│  │ 4. Time-series Train/Test Split    │ │
│  └────────────────────────────────────┘ │
│              │                           │
│              ▼                           │
│  ┌────────────────────────────────────┐ │
│  │ Train 8 Model Configurations:      │ │
│  │   - RandomForest (3 configs)       │ │
│  │   - ExtraTrees (2 configs)         │ │
│  │   - XGBoost (3 configs)            │ │
│  │   - GradientBoosting (2 configs)   │ │
│  └────────────────────────────────────┘ │
│              │                           │
│              ▼                           │
│  ┌────────────────────────────────────┐ │
│  │ Evaluate Each Model:               │ │
│  │   - Calculate ROC AUC              │ │
│  │   - Calculate Average Precision    │ │
│  │   - Compute Combined Score         │ │
│  │   - Track Best Configuration       │ │
│  └────────────────────────────────────┘ │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│    Save Results & Generate Reports      │
└─────────────────────────────────────────┘
```

## Key Improvements Over Original Script

### Original Script ([fe.py](src/fe.py:1))
- Manual feature engineering
- Tests 3 fixed models (RF, ET, XGBoost)
- Single hyperparameter configuration per model
- No systematic optimization
- Manual result comparison

### New App ([generate_features.py](generate_features.py:1))
- **Automated feature generation** (4 strategies, 10 combinations)
- **Multiple model families** (4 types)
- **Hyperparameter tuning** (8 different configurations)
- **Systematic optimization** (80 models tested)
- **Comprehensive tracking** (all results logged)
- **Best configuration selection** (automatic)
- **Reproducible results** (saved configurations)

## Metrics Explained

### ROC AUC (Area Under ROC Curve)
- Measures how well the model ranks predictions
- 0.5 = random guessing, 1.0 = perfect
- Good performance: > 0.8
- Robust to class imbalance in ranking ability

### Average Precision (AP)
- Weighted mean of precision at each threshold
- Focuses on precision-recall tradeoff
- Range: 0 to 1.0
- Particularly important for imbalanced datasets
- Good performance: > 0.7

### Combined Score
- **Sum of ROC AUC + AP**
- Range: 0 to 2.0
- Balances ranking ability with precision
- Higher values indicate better overall performance
- Optimization target for this app

## Customization

The app is designed to be easily customizable:

1. **Add new feature types**: Extend `FeatureEngineering` class
2. **Add new models**: Update `get_model_configs()` method
3. **Change hyperparameters**: Modify params in model configs
4. **Adjust iterations**: Change `n_iterations` parameter
5. **Modify feature selection**: Adjust variance threshold or K value

See [APP_USAGE.md](APP_USAGE.md) for detailed customization examples.

## Performance Considerations

- **Runtime**: ~5-15 minutes for 10 iterations (depends on data size)
- **Memory**: Manages features efficiently with variance thresholding
- **Parallelization**: Models run sequentially but iterations can be run independently
- **Scalability**: Feature selection prevents dimensionality explosion

## Next Steps

After running the optimization:

1. **Review results** in `models/optimization_summary.txt`
2. **Examine best configuration** in `models/best_model_config.json`
3. **Analyze feature importance** from the best model
4. **Retrain on full dataset** using best configuration
5. **Save production model** for deployment
6. **Create prediction pipeline** with same feature engineering

Example production code:

```python
import pickle
import json

# Load best configuration
with open('models/best_model_config.json', 'r') as f:
    best_config = json.load(f)

# Recreate and train on full data
# (implement based on best_config)

# Save for production
with open('models/production_model.pkl', 'wb') as f:
    pickle.dump(trained_model, f)
```

## Troubleshooting

### Installation Issues
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Memory Issues
- Reduce `n_interactions` parameter (default: 5)
- Increase variance threshold (default: 0.01)
- Reduce number of top features (default: 30)

### Long Runtime
- Reduce `n_iterations` (default: 10)
- Use fewer model configurations
- Reduce number of features selected

### Poor Performance
- Check data quality and class balance
- Verify temporal ordering is preserved
- Try different feature combinations
- Adjust hyperparameter ranges

## Technical Details

### Dependencies
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: ML models and feature selection
- xgboost: Gradient boosting implementation
- matplotlib: Visualization (for future enhancements)
- imbalanced-learn: For SMOTE (if enabled)

### Time Complexity
- Feature generation: O(n × m) where n=visits, m=observation types
- Model training: O(k × t × f) where k=models, t=trees, f=features
- Total: O(iterations × models × training_time)

### Space Complexity
- Features: O(n × f) where n=samples, f=features
- Results history: O(iterations × models)
- Manageable through feature selection and variance thresholding

## License & Usage

This app is designed for the AutoML sepsis prediction project. Feel free to modify and extend it for your specific needs.

## Support

For questions or issues:
1. Check [APP_USAGE.md](APP_USAGE.md) for detailed usage instructions
2. Review [tests/test_app.py](tests/test_app.py) for validation
3. Examine the original [fe.py](src/fe.py:1) for baseline comparison

---

**Created**: 2025-12-16
**Purpose**: Automated feature engineering and model optimization
**Goal**: Maximize ROC AUC + Average Precision for sepsis prediction
