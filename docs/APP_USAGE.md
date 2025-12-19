# Automated Feature Engineering App - Usage Guide

## Overview

This app automatically creates new features and optimizes machine learning models to maximize the sum of ROC AUC and Average Precision (AP) scores for sepsis prediction.

## Features

The app provides automated:

1. **Feature Engineering**
   - Temporal features: trends, volatility, first/last values, changes over time
   - Statistical features: median, quartiles, IQR, skewness, kurtosis, coefficient of variation
   - Interaction features: pairwise multiplications and divisions
   - Aggregated features: cross-observation statistics

2. **Model Optimization**
   - Tests multiple model types: RandomForest, ExtraTrees, XGBoost, GradientBoosting
   - Tries different hyperparameter configurations
   - Evaluates all combinations of features and models
   - Maximizes ROC AUC + Average Precision combined score

3. **Results Tracking**
   - Saves all experiment logs
   - Tracks best performing configurations
   - Generates summary reports

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the app with default settings (10 iterations):

```bash
python generate_features.py
```

Run with custom number of iterations:

```bash
python generate_features.py --iterations 15
```

Run continuously for a specified duration (time-based mode):

```bash
# Run for 30 minutes
python generate_features.py --duration 30

# Run for 2 hours
python generate_features.py --duration 120
```

**Time-Based Mode**: When using `--duration`, the app will cycle through the standard 10 feature combinations repeatedly for the specified time. Each cycle uses random sampling in interaction features, ensuring unique features in each iteration and increasing chances of finding better performance.

### What Happens

The app will:

1. Load the raw data from `data/data_science_project_data.csv`
2. Run 10 iterations, each testing a different combination of feature sets:
   - Iteration 1: Baseline (only base features)
   - Iteration 2: Base + Temporal features
   - Iteration 3: Base + Statistical features
   - Iteration 4: Base + Aggregated features
   - Iteration 5: Base + Interaction features
   - Iteration 6: Base + Temporal + Statistical
   - Iteration 7: Base + Temporal + Aggregated
   - Iteration 8: Base + Statistical + Aggregated
   - Iteration 9: Base + Temporal + Statistical + Aggregated
   - Iteration 10: All features combined

3. For each iteration:
   - Creates the specified features
   - Performs feature selection (variance threshold + SelectKBest)
   - Trains 8 different model configurations
   - Evaluates each on ROC AUC and Average Precision
   - Tracks the best performing configuration

4. Outputs:
   - Real-time progress to console
   - Best model configuration
   - Detailed logs in `models/` directory

## Output Files

The app generates three files in the `models/` directory:

### 1. `optimization_log.json`
Complete log of all experiments including:
- Feature sets used in each iteration
- Selected features for each iteration
- All model results (ROC AUC, AP, combined score)
- Timestamps

### 2. `best_model_config.json`
Configuration of the best performing model:
```json
{
  "combined_score": 1.8234,
  "roc_auc": 0.9123,
  "avg_precision": 0.9111,
  "model_name": "XGBoost",
  "params": {
    "n_estimators": 150,
    "max_depth": 8,
    "learning_rate": 0.05,
    ...
  },
  "iteration": 9
}
```

### 3. `optimization_summary.txt`
Human-readable summary with:
- Total iterations and models trained
- Best scores achieved
- Top 10 model configurations

## Customization

### Modifying Feature Generation

To add new feature types, edit the `FeatureEngineering` class in [generate_features.py](generate_features.py):

```python
def create_custom_features(self, df_raw):
    # Your custom feature logic here
    pass
```

Then add it to the feature combinations in `run_optimization()`:

```python
feature_combinations = [
    ['custom'],  # your new feature type
    ['temporal', 'custom'],
    # ...
]
```

### Modifying Model Configurations

Edit the `get_model_configs()` method in the `ModelOptimizer` class:

```python
configs = [
    {
        'name': 'YourModel',
        'model': YourModelClass,
        'params': [
            {'param1': value1, 'param2': value2},
            # Add more parameter combinations
        ]
    },
    # ...
]
```

### Changing Number of Iterations

Use command-line arguments:

```bash
# Run with 20 iterations
python generate_features.py --iterations 20
```

### Using Continuous Mode for Maximum Exploration

The time-based continuous mode allows you to explore many more feature combinations by running for a specified duration:

```bash
# Run for 1 hour
python generate_features.py --duration 60

# Run for 4 hours overnight
python generate_features.py --duration 240
```

**Benefits of Continuous Mode:**
- **More Feature Exploration**: Tests the same proven combinations multiple times with different random samples
- **Random Interaction Sampling**: Each iteration generates unique interaction features due to random sampling
- **Performance Optimization**: Increases chances of finding better model performance through repeated sampling
- **Flexible Time Budget**: Run as long as you have compute time available

The app will:
1. Cycle through the 10 standard feature combinations repeatedly
2. Use random sampling for interaction features in each cycle (ensuring unique features)
3. Track and save the best model found across all iterations
4. Save all results incrementally (safe to stop anytime)

**Example**: Running for 30 minutes might complete 3-4 cycles (30-40 iterations total), each with different random interaction features.

### Adjusting Feature Selection

In the `run_feature_iteration()` method:

```python
# Change variance threshold
selector = VarianceThreshold(threshold=0.05)  # default: 0.01

# Change number of top features
n_features = min(50, X.shape[1])  # default: 30
```

## Architecture

### Classes

1. **FeatureEngineering**
   - Handles all feature generation strategies
   - Creates temporal, statistical, interaction, and aggregated features
   - Returns DataFrames that can be merged with base features

2. **ModelOptimizer**
   - Manages model training and evaluation
   - Tracks best performing configurations
   - Maintains results history

3. **AutomatedFeatureOptimizer**
   - Main orchestrator
   - Loads data and runs iterations
   - Combines features and models
   - Saves results

### Workflow

```
Load Data
    ↓
For each iteration:
    Create Base Features
    ↓
    Add Specified Feature Sets
    ↓
    Feature Selection
    ↓
    Train/Test Split (80/20, time-aware)
    ↓
    Train Multiple Models
    ↓
    Evaluate (ROC AUC + AP)
    ↓
    Track Best Configuration
    ↓
Save Results
```

## Performance Tips

1. **Reduce iterations** for faster initial testing
2. **Limit interaction features** to avoid feature explosion (controlled by `n_interactions` parameter)
3. **Adjust feature selection threshold** to reduce dimensionality
4. **Use fewer model configurations** for each model type

## Interpreting Results

### Combined Score
- Sum of ROC AUC + Average Precision
- Range: 0 to 2.0
- Higher is better
- Balances model's ranking ability (ROC AUC) with precision-recall tradeoff (AP)

### ROC AUC
- Measures model's ability to rank positive cases higher than negative cases
- Range: 0.5 (random) to 1.0 (perfect)
- Good: > 0.8

### Average Precision (AP)
- Weighted mean of precisions at each threshold
- Range: 0 to 1.0
- Especially important for imbalanced datasets
- Good: > 0.7

## Example Output

```
================================================================================
ITERATION 9: Testing feature combinations
================================================================================

Creating base features...
Created 245 base features
Adding temporal features...
Adding statistical features...
Adding aggregated features...
Total features before selection: 512

Performing feature selection...
Selected 30 features

Training set: 800 samples | Test set: 200 samples

Training models...
  RandomForest         | ROC AUC: 0.8912 | AP: 0.8734 | Combined: 1.7646
  ExtraTrees          | ROC AUC: 0.8845 | AP: 0.8698 | Combined: 1.7543
  XGBoost             | ROC AUC: 0.9123 | AP: 0.9111 | Combined: 1.8234
  GradientBoosting    | ROC AUC: 0.8956 | AP: 0.8876 | Combined: 1.7832

================================================================================
OPTIMIZATION COMPLETE - BEST RESULTS
================================================================================

Best Combined Score: 1.8234
  ROC AUC: 0.9123
  Average Precision: 0.9111
  Model: XGBoost
  Iteration: 9

Feature Sets Used: ['temporal', 'statistical', 'aggregated']
Number of Features: 30
```

## Troubleshooting

### Memory Issues
- Reduce `n_interactions` parameter
- Reduce number of feature combinations
- Increase variance threshold for feature selection

### Long Runtime
- Reduce `n_iterations`
- Use fewer hyperparameter configurations per model
- Reduce number of top features selected

### Poor Performance
- Check class imbalance in data
- Verify temporal ordering is preserved
- Increase number of features selected
- Try different feature combinations

## Next Steps

After finding the best configuration:

1. Retrain the best model on full dataset
2. Save the trained model using pickle or joblib
3. Create a prediction pipeline using the same feature engineering steps
4. Deploy the model for inference

Example:
```python
import pickle

# After optimization, retrain best model
best_model = optimizer.model_optimizer.best_config['model']
best_features = optimizer.experiment_log[best_iteration]['selected_features']

# Save for production
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
```
