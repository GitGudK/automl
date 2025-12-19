# Configuration File Format

## best_model_config.json

The `best_model_config.json` file contains all information needed to reproduce the best model, including the feature engineering pipeline configuration.

### New Format (Self-Contained)

Starting from the latest version, the configuration file is **self-contained** and includes feature information directly:

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
    "random_state": 42,
    "eval_metric": "logloss"
  },
  "iteration": 9,
  "feature_sets": ["temporal", "statistical", "aggregated"],
  "selected_features": [
    "TEMP_mean_value",
    "HR_mean_value",
    "TEMP_temporal_value_trend",
    "...30 features total..."
  ],
  "n_features": 30
}
```

### Fields Explained

| Field | Type | Description |
|-------|------|-------------|
| `combined_score` | float | Sum of ROC AUC + Average Precision |
| `roc_auc` | float | ROC AUC score on test set |
| `avg_precision` | float | Average Precision score on test set |
| `model_name` | string | Model type (RandomForest, ExtraTrees, XGBoost, GradientBoosting) |
| `params` | object | Model hyperparameters |
| `iteration` | integer | Which optimization iteration produced this config |
| `feature_sets` | array | Which feature sets to create (temporal, statistical, aggregated, interactions) |
| `selected_features` | array | Exact list of feature names selected during training |
| `n_features` | integer | Number of selected features (convenience field) |

### Benefits of New Format

1. **Self-Contained**: No need to read `optimization_log.json` separately
2. **Portable**: Can copy just this one file for deployment
3. **Faster Loading**: Only reads one file instead of two
4. **More Robust**: No iteration indexing lookup required
5. **Backward Compatible**: Code still supports legacy format

### Legacy Format (Pre-Update)

Older configurations only had these fields:

```json
{
  "combined_score": 1.8234,
  "roc_auc": 0.9123,
  "avg_precision": 0.9111,
  "model_name": "XGBoost",
  "params": { ... },
  "iteration": 9
}
```

Feature information had to be looked up from `optimization_log.json` by iteration number.

### Backward Compatibility

The `FeatureReproducer` class automatically detects the format:

```python
from feature_pipeline import FeatureReproducer

reproducer = FeatureReproducer('models/best_model_config.json')

# If new format: Reads feature_sets and selected_features directly
# If legacy format: Falls back to reading optimization_log.json
```

Console output for legacy format:
```
Warning: Using legacy config format. Consider re-running optimization.
```

### Migration

To migrate from legacy to new format:

1. **Re-run optimization:**
   ```bash
   python generate_features.py
   ```

2. **Or manually update:** Add `feature_sets`, `selected_features`, and `n_features` fields from `optimization_log.json`

### Example: Finding Features in Legacy Format

If you have a legacy config and need to find the features:

```python
import json

# Load both files
with open('models/best_model_config.json') as f:
    config = json.load(f)

with open('models/optimization_log.json') as f:
    log = json.load(f)

# Find matching iteration
iteration = config['iteration']
for exp in log:
    if exp['iteration'] == iteration:
        print(f"Feature sets: {exp['feature_sets']}")
        print(f"Selected features: {exp['selected_features']}")
        break
```

## optimization_log.json

Complete log of all iterations, used for analysis and debugging:

```json
[
  {
    "iteration": 1,
    "feature_sets": [],
    "n_features": 30,
    "selected_features": ["feature1", "feature2", ...],
    "results": [
      {
        "model_name": "RandomForest",
        "params": {...},
        "roc_auc": 0.8512,
        "avg_precision": 0.8334,
        "combined_score": 1.6846
      },
      ...
    ],
    "timestamp": "2024-01-15T10:30:00"
  },
  ...
]
```

**Note:** The `optimization_log.json` is still generated for reference, but is no longer required for production deployment when using the new config format.

## Summary

✅ **New format**: Self-contained, includes feature information
✅ **Backward compatible**: Legacy format still supported
✅ **Automatic detection**: No code changes needed
✅ **Faster**: Reads one file instead of two
✅ **Portable**: Easier deployment

**Recommendation:** Re-run optimization to generate the new format for best performance and simplicity.
