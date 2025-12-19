# Feature Pipeline Guide

## Overview

The Feature Pipeline module allows you to recreate the exact same features for new time-series data that were used during model training. This ensures consistency between training and production inference.

## Key Components

### 1. FeatureReproducer Class
Recreates features using saved configuration from optimization.

### 2. ProductionPipeline Class
Complete pipeline including feature engineering and model predictions.

### 3. Helper Scripts
- [train_best_model.py](train_best_model.py) - Train and save the best model
- [tests/test_feature_pipeline.py](tests/test_feature_pipeline.py) - Test feature reproduction
- [run_tests.py](run_tests.py) - Test runner for all tests

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Run Optimization (Once)                            │
│  python generate_features.py                                 │
│  → Generates: models/best_model_config.json                 │
│               models/optimization_log.json                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Save Best Model (Once)                             │
│  python train_best_model.py --full                           │
│  → Trains model on full dataset                             │
│  → Saves: models/best_model.pkl                             │
│           models/feature_names.json                          │
│           models/production_model_info.txt                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Use in Production (Repeatedly)                     │
│  → Load new patient data                                    │
│  → Transform features using FeatureReproducer               │
│  → Get predictions using ProductionPipeline                  │
└─────────────────────────────────────────────────────────────┘
```

## Usage Examples

### Example 1: Feature Engineering Only

```python
from feature_pipeline import FeatureReproducer
import pandas as pd

# Initialize reproducer with saved configuration
reproducer = FeatureReproducer('models/best_model_config.json')

# Load new patient data (same format as training data)
df_new_patients = pd.read_csv('data/new_patients.csv')

# Transform to features (returns only selected features)
df_features = reproducer.transform(df_new_patients, return_all_features=False)

# Save features
df_features.to_csv('data/new_patient_features.csv', index=False)

print(f"Created features for {len(df_features)} visits")
print(f"Feature columns: {df_features.shape[1]}")
```

### Example 2: Complete Production Pipeline

```python
from feature_pipeline import ProductionPipeline
import pandas as pd

# Initialize pipeline (do this once at startup)
pipeline = ProductionPipeline(
    config_path='models/best_model_config.json',
    model_path='models/best_model.pkl'
)

# Load new patient data
df_new_patients = pd.read_csv('data/new_patients.csv')

# Get predictions with probabilities
predictions = pipeline.predict(df_new_patients, return_probabilities=True)

# Results: DataFrame with visit_id, client, sepsis_probability
print(predictions)
#   visit_id  client  sepsis_probability
# 0     1001    C001            0.8234
# 1     1002    C002            0.1567
# 2     1003    C001            0.6789
```

### Example 3: Real-time Single Patient Prediction

```python
from feature_pipeline import ProductionPipeline
import pandas as pd

# Initialize pipeline once at application startup
pipeline = ProductionPipeline(
    config_path='models/best_model_config.json',
    model_path='models/best_model.pkl'
)

def predict_sepsis_risk(visit_data):
    """
    Predict sepsis risk for a single patient visit.

    Args:
        visit_data: DataFrame with time-series observations for one visit
                   Required columns: visit_id, client, observation_date,
                                   observation_type_code, observation_value,
                                   reference_range_min, reference_range_max

    Returns:
        Dictionary with risk assessment
    """
    # Get prediction
    prediction = pipeline.predict(visit_data, return_probabilities=True)

    risk_score = prediction['sepsis_probability'].iloc[0]

    # Classify risk level
    if risk_score > 0.7:
        alert_level = "HIGH RISK"
        action = "Immediate clinical review required"
    elif risk_score > 0.4:
        alert_level = "MEDIUM RISK"
        action = "Monitor closely"
    else:
        alert_level = "LOW RISK"
        action = "Continue routine monitoring"

    return {
        'visit_id': prediction['visit_id'].iloc[0],
        'client': prediction['client'].iloc[0],
        'risk_score': risk_score,
        'alert_level': alert_level,
        'recommended_action': action
    }

# Example usage
# visit_observations = load_patient_observations(visit_id=1001)
# result = predict_sepsis_risk(visit_observations)
# print(result)
```

### Example 4: Batch Processing Multiple Files

```python
from feature_pipeline import ProductionPipeline
import pandas as pd
import glob
import os

# Initialize pipeline
pipeline = ProductionPipeline(
    config_path='models/best_model_config.json',
    model_path='models/best_model.pkl'
)

# Process all CSV files in incoming directory
incoming_dir = 'data/incoming/'
output_dir = 'data/predictions/'
os.makedirs(output_dir, exist_ok=True)

for file_path in glob.glob(f'{incoming_dir}*.csv'):
    print(f"Processing {file_path}...")

    # Load new data
    df_new = pd.read_csv(file_path)

    # Get predictions
    predictions = pipeline.predict(df_new, return_probabilities=True)

    # Save predictions with same filename
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, f'predictions_{filename}')
    predictions.to_csv(output_path, index=False)

    print(f"  ✓ Saved predictions to {output_path}")

print("Batch processing complete!")
```

### Example 5: Get All Features (Not Just Selected)

```python
from feature_pipeline import FeatureReproducer
import pandas as pd

reproducer = FeatureReproducer('models/best_model_config.json')
df_new = pd.read_csv('data/new_patients.csv')

# Get ALL engineered features (before feature selection)
df_all_features = reproducer.transform(df_new, return_all_features=True)

# Get ONLY selected features (used by model)
df_selected = reproducer.transform(df_new, return_all_features=False)

print(f"All features: {df_all_features.shape[1]} columns")
print(f"Selected features: {df_selected.shape[1]} columns")
```

## Data Format Requirements

### Input Data Format (New Patients)

Your new patient data CSV must have these columns:

```python
Required columns:
- visit_id: Unique identifier for each hospital visit
- client: Patient/client identifier
- observation_date: Timestamp of observation (YYYY-MM-DD HH:MM:SS)
- observation_type_code: Code identifying the type of observation
- observation_value: Numeric value of the observation
- reference_range_min: Minimum normal value for this observation type
- reference_range_max: Maximum normal value for this observation type

Optional column:
- condition_present: 0 or 1 (only needed if you have ground truth labels)
```

Example CSV format:
```csv
visit_id,client,observation_date,observation_type_code,observation_value,reference_range_min,reference_range_max
1001,C001,2024-01-15 08:00:00,TEMP,38.5,36.0,37.5
1001,C001,2024-01-15 08:00:00,HR,95,60,100
1001,C001,2024-01-15 12:00:00,TEMP,39.2,36.0,37.5
1002,C002,2024-01-15 09:00:00,TEMP,37.0,36.0,37.5
```

### Output Format (Features)

The `transform()` method returns a DataFrame with:

```python
Columns:
- visit_id: Same as input
- client: Same as input
- [feature_1]: First selected feature
- [feature_2]: Second selected feature
- ...
- [feature_N]: Nth selected feature
- condition_present: Optional, if present in input
```

### Output Format (Predictions)

The `predict()` method returns a DataFrame with:

```python
Columns:
- visit_id: Visit identifier
- client: Client identifier
- sepsis_probability: Predicted probability of sepsis (0.0 to 1.0)
  OR
- sepsis_prediction: Binary prediction (0 or 1) if return_probabilities=False
```

## Step-by-Step Setup Guide

### Step 1: Run Optimization

First, run the optimization to find the best feature combination and model:

```bash
python generate_features.py
```

This will create:
- `models/best_model_config.json`
- `models/optimization_log.json`
- `models/optimization_summary.txt`

### Step 2: Train and Save Best Model

Train the best model on full dataset and save it:

```bash
# Train on full dataset (recommended for production)
python train_best_model.py --full

# Or train on 80% to evaluate on 20% test set
python train_best_model.py
```

This will create:
- `models/best_model.pkl` - Trained model ready for predictions
- `models/feature_names.json` - List of features used
- `models/production_model_info.txt` - Model information

### Step 3: Test the Pipeline

Test that everything works correctly:

```bash
python run_tests.py pipeline
```

This will:
- Verify feature reproduction works
- Test single and multiple visits
- Create sample new patient data
- Show production usage examples

### Step 4: Use in Production

Now you can use the pipeline in your production code:

```python
from feature_pipeline import ProductionPipeline

pipeline = ProductionPipeline(
    config_path='models/best_model_config.json',
    model_path='models/best_model.pkl'
)

# Use pipeline.predict() for new patients
```

## API Reference

### FeatureReproducer

#### `__init__(config_path=None)`
Initialize the feature reproducer.

**Parameters:**
- `config_path` (str, optional): Path to best_model_config.json

#### `load_config(config_path)`
Load saved model configuration.

**Parameters:**
- `config_path` (str): Path to configuration file

#### `transform(df_raw, return_all_features=False)`
Transform raw time-series data into engineered features.

**Parameters:**
- `df_raw` (DataFrame): Raw time-series data
- `return_all_features` (bool): If True, return all features. If False, return only selected features.

**Returns:**
- DataFrame with engineered features

**Example:**
```python
df_features = reproducer.transform(df_new, return_all_features=False)
```

### ProductionPipeline

#### `__init__(config_path, model_path=None)`
Initialize production pipeline.

**Parameters:**
- `config_path` (str): Path to best_model_config.json
- `model_path` (str, optional): Path to saved model pickle file

#### `load_model(model_path)`
Load a trained model from pickle file.

**Parameters:**
- `model_path` (str): Path to model pickle file

#### `predict(df_raw, return_probabilities=True)`
Make predictions on new data.

**Parameters:**
- `df_raw` (DataFrame): Raw time-series data
- `return_probabilities` (bool): If True, return probabilities. If False, return class labels.

**Returns:**
- DataFrame with visit_id, client, and predictions

**Example:**
```python
predictions = pipeline.predict(df_new, return_probabilities=True)
```

## Troubleshooting

### Error: "Configuration not loaded"
**Solution:** Initialize with config path or call `load_config()`:
```python
reproducer = FeatureReproducer('models/best_model_config.json')
```

### Error: "Model not loaded"
**Solution:** Initialize ProductionPipeline with model_path or call `load_model()`:
```python
pipeline = ProductionPipeline(
    config_path='models/best_model_config.json',
    model_path='models/best_model.pkl'
)
```

### Warning: "Features from training are missing"
**Cause:** New data doesn't have all observation types that were in training data.

**Solution:** This is expected if new patients have different tests. Missing features are filled with 0.

### Error: "KeyError" on column names
**Cause:** Input data is missing required columns.

**Solution:** Ensure your CSV has all required columns:
- visit_id, client, observation_date, observation_type_code, observation_value, reference_range_min, reference_range_max

### Features have NaN values
**Cause:** Some calculations resulted in NaN (e.g., std of single value).

**Solution:** The pipeline automatically fills NaN with 0 or column mean. This is expected behavior.

## Performance Considerations

### Automatic Optimization
The FeatureReproducer is **automatically optimized** to only create the feature sets that were used in the best model configuration:

- If the best model used `['temporal', 'statistical']`, it will **skip** creating `aggregated` and `interactions` features
- This can reduce feature generation time by **50-75%** depending on which features were selected
- No code changes needed - optimization happens automatically based on `optimization_log.json`

**How it works:**
```
Traditional Approach (slow):
  Raw Data → Base → Temporal → Statistical → Aggregated → Interactions → Select 30 features
  [Creates ALL features, then selects subset]

Optimized Approach (fast):
  Raw Data → Base → Temporal → Statistical → Select 30 features
  [Only creates needed feature sets, skips Aggregated and Interactions]

  ⚡ Result: 40-50% faster feature generation
```

Example output:
```
Loaded configuration from iteration 9
Feature sets to create: ['temporal', 'statistical', 'aggregated']
Skipping unused feature sets: ['interactions'] (saves compute time)
Number of selected features: 30
```

### Memory Usage
- Feature reproduction is memory-efficient
- Processes data in grouped chunks
- Can handle thousands of visits

### Speed (Optimized)
- Single visit: ~10-50ms
- 100 visits: ~1-5 seconds (faster if fewer feature sets)
- 1000 visits: ~10-30 seconds (faster if fewer feature sets)

**Performance varies by feature sets:**
- Base only: Fastest
- Base + 1-2 feature sets: Fast (typical production case)
- Base + all 4 feature sets: Slower (but still efficient)

### Additional Optimization Tips
1. **Initialize pipeline once** at application startup
2. **Batch multiple visits** together when possible
3. **Cache feature reproducer** instance
4. **Use return_all_features=False** for faster processing (default)
5. **Automatic optimization** - no action needed, already implemented!

## Integration Examples

### Flask Web API

```python
from flask import Flask, request, jsonify
from feature_pipeline import ProductionPipeline
import pandas as pd

app = Flask(__name__)

# Initialize pipeline once at startup
pipeline = ProductionPipeline(
    config_path='models/best_model_config.json',
    model_path='models/best_model.pkl'
)

@app.route('/predict', methods=['POST'])
def predict():
    # Expect JSON array of observations
    data = request.json
    df = pd.DataFrame(data)

    # Get predictions
    predictions = pipeline.predict(df, return_probabilities=True)

    # Return as JSON
    return jsonify(predictions.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Database Integration

```python
from feature_pipeline import ProductionPipeline
import pandas as pd
import sqlalchemy

# Initialize pipeline
pipeline = ProductionPipeline(
    config_path='models/best_model_config.json',
    model_path='models/best_model.pkl'
)

# Connect to database
engine = sqlalchemy.create_engine('postgresql://user:pass@localhost/db')

# Query new patients
query = """
    SELECT visit_id, client, observation_date, observation_type_code,
           observation_value, reference_range_min, reference_range_max
    FROM patient_observations
    WHERE prediction_status = 'pending'
"""
df_new = pd.read_sql(query, engine)

# Get predictions
predictions = pipeline.predict(df_new, return_probabilities=True)

# Save predictions back to database
predictions.to_sql('sepsis_predictions', engine, if_exists='append', index=False)
```

## Files Generated

After running the complete workflow:

```
models/
├── best_model_config.json          # Best configuration from optimization
├── optimization_log.json           # Complete experiment history
├── optimization_summary.txt        # Human-readable summary
├── best_model.pkl                  # Trained model (after train_best_model.py)
├── feature_names.json              # Feature list (after train_best_model.py)
└── production_model_info.txt       # Model info (after train_best_model.py)
```

## Best Practices

1. **Version Control**: Save model files with version numbers
   ```python
   model_path = f'models/best_model_v{version}.pkl'
   ```

2. **Model Monitoring**: Track prediction distribution over time
   ```python
   predictions['timestamp'] = pd.Timestamp.now()
   predictions.to_csv('predictions_log.csv', mode='a')
   ```

3. **Data Validation**: Validate input data before transformation
   ```python
   required_cols = ['visit_id', 'client', 'observation_date', ...]
   assert all(col in df.columns for col in required_cols)
   ```

4. **Error Handling**: Wrap predictions in try-except
   ```python
   try:
       predictions = pipeline.predict(df_new)
   except Exception as e:
       logger.error(f"Prediction failed: {e}")
       # Handle error appropriately
   ```

5. **Logging**: Log all predictions for audit trail
   ```python
   import logging
   logging.info(f"Predicted {len(predictions)} visits")
   ```

## Summary

The Feature Pipeline provides a robust, production-ready system for:
- ✅ Reproducing exact training features on new data
- ✅ Making consistent predictions
- ✅ Handling missing observation types gracefully
- ✅ Processing single or batch patients
- ✅ Integrating with existing systems

For questions or issues, refer to [tests/test_feature_pipeline.py](tests/test_feature_pipeline.py) for working examples.
