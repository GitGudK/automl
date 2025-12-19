# Inference Guide

Complete guide for making predictions with the trained sepsis prediction model.

## Quick Start

### 1. Train the Model First

Before making predictions, you need to train the model:

```bash
# Run optimization to find best model
python generate_features.py

# Train and save the best model for production
python train_best_model.py --full
```

This creates:
- `data/models/best_model_config.json` - Model configuration
- `data/models/best_model.pkl` - Trained model
- `data/models/feature_names.json` - Feature list

### 2. Make Predictions

```bash
# Basic prediction
python predict.py --input data/new_patients.csv --output predictions.csv

# Include probability scores
python predict.py --input data/new_patients.csv --output predictions.csv --probabilities
```

## Usage Modes

### Mode 1: File-Based Prediction

Predict on a CSV file containing patient data:

```bash
python predict.py --input data/new_patients.csv --output predictions.csv --probabilities
```

**Input Format:**

Your CSV must have these columns:
- `visit_id` - Unique identifier for each patient visit
- `client` - Client identifier
- `observation_code` - Medical observation code
- `observation_value` - Numeric value of the observation

Example:
```csv
visit_id,client,observation_code,observation_value
V001,C1,HR,85
V001,C1,TEMP,98.6
V001,C1,BP_SYS,120
V002,C1,HR,92
...
```

**Output Format:**

The output CSV contains:
- `visit_id` - Patient visit ID
- `prediction` - Binary prediction (0=No sepsis, 1=Sepsis)
- `probability` - Risk score 0-1 (if `--probabilities` flag used)

Example:
```csv
visit_id,prediction,probability
V001,0,0.1234
V002,1,0.8567
V003,0,0.2345
...
```

### Mode 2: Test Set Evaluation

Evaluate model performance on the test portion of your dataset:

```bash
python predict.py --test
```

This will:
- Use the last 20% of visits as test set
- Generate predictions
- Calculate performance metrics (accuracy, precision, recall, ROC AUC, etc.)
- Display confusion matrix and classification report
- Save predictions to `predictions_test_set.csv`

**Example Output:**
```
================================================================================
TEST SET PERFORMANCE
================================================================================

Accuracy: 0.9123
Precision: 0.8567
Recall: 0.8234
F1 Score: 0.8398
ROC AUC: 0.9456
Average Precision: 0.9123

Confusion Matrix:
                 Predicted
              Negative  Positive
Actual Negative    145        8
       Positive     12       35

Classification Report:
              precision    recall  f1-score   support

    Negative       0.92      0.95      0.93       153
    Positive       0.81      0.74      0.77        47

    accuracy                           0.90       200
   macro avg       0.87      0.84      0.85       200
weighted avg       0.90      0.90      0.90       200

================================================================================
```

### Mode 3: Interactive Mode

Start an interactive Python session with the model loaded:

```bash
python predict.py --interactive
```

This gives you a Python REPL with:
- `pipeline` - ProductionPipeline instance
- `pd` - pandas module
- `np` - numpy module

**Example Session:**
```python
# Load your data
df = pd.read_csv('data/new_patients.csv')

# Make predictions
predictions = pipeline.predict(df, return_probabilities=True)

# View results
print(predictions)

# Filter high-risk cases
high_risk = predictions[predictions['probability'] > 0.7]
print(high_risk)

# Get statistics
print(f"Mean risk: {predictions['probability'].mean():.4f}")
print(f"Positive rate: {predictions['prediction'].mean():.2%}")
```

## Advanced Usage

### Custom Model Paths

Specify custom paths for model files:

```bash
python predict.py \
    --input data/new_patients.csv \
    --output predictions.csv \
    --config path/to/config.json \
    --model path/to/model.pkl \
    --probabilities
```

### Programmatic Usage

Use the inference pipeline in your own Python code:

```python
from feature_pipeline import ProductionPipeline
import pandas as pd

# Load pipeline
pipeline = ProductionPipeline(
    config_path='data/models/best_model_config.json',
    model_path='data/models/best_model.pkl'
)

# Load data
df = pd.read_csv('data/new_patients.csv')

# Make predictions
predictions = pipeline.predict(df, return_probabilities=True)

# Use predictions
for idx, row in predictions.iterrows():
    visit_id = row['visit_id']
    prediction = row['prediction']
    probability = row['probability']

    if prediction == 1:
        print(f"⚠️ Alert: Visit {visit_id} at high risk ({probability:.2%})")
```

### Batch Processing

Process multiple files:

```bash
# Process all CSV files in a directory
for file in data/batch/*.csv; do
    python predict.py \
        --input "$file" \
        --output "predictions/$(basename $file)" \
        --probabilities
done
```

### Real-Time Prediction

For a single visit in real-time:

```python
from feature_pipeline import ProductionPipeline
import pandas as pd

# Load pipeline once
pipeline = ProductionPipeline(
    config_path='data/models/best_model_config.json',
    model_path='data/models/best_model.pkl'
)

# Incoming patient data
new_visit = pd.DataFrame({
    'visit_id': ['V999'],
    'client': ['C1'],
    'observation_code': ['HR', 'TEMP', 'BP_SYS'],
    'observation_value': [105, 101.2, 150]
})

# Predict
result = pipeline.predict(new_visit, return_probabilities=True)

if result['prediction'].iloc[0] == 1:
    prob = result['probability'].iloc[0]
    print(f"⚠️ SEPSIS ALERT: {prob:.1%} risk")
```

## Output Interpretation

### Prediction Values

- **0** - No sepsis predicted (negative case)
- **1** - Sepsis predicted (positive case)

### Probability Scores

When using `--probabilities`, you get a risk score between 0 and 1:

- **0.0 - 0.3** - Low risk (unlikely to have sepsis)
- **0.3 - 0.7** - Medium risk (monitor closely)
- **0.7 - 1.0** - High risk (likely sepsis, immediate attention)

**Note:** The exact threshold depends on your use case. Default is 0.5, but you might adjust based on:
- Cost of false positives vs false negatives
- Clinical protocols
- Available resources

### Custom Thresholds

You can apply custom thresholds in your code:

```python
# Conservative (fewer false negatives)
predictions['high_sensitivity'] = (predictions['probability'] > 0.3).astype(int)

# Aggressive (fewer false positives)
predictions['high_specificity'] = (predictions['probability'] > 0.7).astype(int)
```

## Performance Expectations

Based on typical model performance:

| Metric | Expected Range |
|--------|----------------|
| Accuracy | 85-92% |
| Precision | 75-88% |
| Recall | 70-85% |
| ROC AUC | 88-94% |
| Avg Precision | 85-92% |

**Note:** Actual performance depends on:
- Quality of training data
- Data distribution
- Number of observations per visit
- Feature engineering choices

## Troubleshooting

### Error: "Model configuration not found"

**Solution:**
```bash
# Train the model first
python generate_features.py
python train_best_model.py --full
```

### Error: "column X not found"

**Cause:** Input data is missing required columns

**Solution:** Ensure your CSV has these columns:
- `visit_id`
- `client`
- `observation_code`
- `observation_value`

### Low Prediction Accuracy

**Possible causes:**
1. **Data drift** - New data differs from training data
   - Solution: Retrain model with recent data

2. **Insufficient observations** - Visits have too few observations
   - Solution: Collect more data per visit

3. **Different data distribution** - Test data differs from training
   - Solution: Check data quality and preprocessing

### Memory Issues

**For large datasets:**

```python
# Process in chunks
chunk_size = 1000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    predictions = pipeline.predict(chunk, return_probabilities=True)
    predictions.to_csv('output.csv', mode='a', header=False, index=False)
```

## Integration Examples

### REST API

Create a simple Flask API:

```python
from flask import Flask, request, jsonify
from feature_pipeline import ProductionPipeline
import pandas as pd

app = Flask(__name__)

# Load model once at startup
pipeline = ProductionPipeline(
    config_path='data/models/best_model_config.json',
    model_path='data/models/best_model.pkl'
)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    df = pd.DataFrame(data)

    # Make prediction
    result = pipeline.predict(df, return_probabilities=True)

    return jsonify(result.to_dict('records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Scheduled Batch Job

Create a cron job to process new data:

```bash
#!/bin/bash
# predict_daily.sh

# Get today's date
DATE=$(date +%Y%m%d)

# Process today's data
python predict.py \
    --input data/incoming/patients_$DATE.csv \
    --output data/predictions/predictions_$DATE.csv \
    --probabilities

# Send high-risk alerts
python send_alerts.py --input data/predictions/predictions_$DATE.csv
```

Add to crontab:
```bash
# Run daily at 2 AM
0 2 * * * /path/to/predict_daily.sh
```

### Database Integration

Query from database and predict:

```python
import sqlalchemy
from feature_pipeline import ProductionPipeline
import pandas as pd

# Connect to database
engine = sqlalchemy.create_engine('postgresql://user:pass@localhost/db')

# Query recent visits
query = """
    SELECT visit_id, client, observation_code, observation_value
    FROM patient_observations
    WHERE visit_date >= CURRENT_DATE
"""
df = pd.read_sql(query, engine)

# Load pipeline and predict
pipeline = ProductionPipeline(
    config_path='data/models/best_model_config.json',
    model_path='data/models/best_model.pkl'
)

predictions = pipeline.predict(df, return_probabilities=True)

# Write back to database
predictions.to_sql('sepsis_predictions', engine, if_exists='append', index=False)
```

## Best Practices

### 1. Data Quality

**Always validate input data:**
```python
def validate_data(df):
    required_cols = ['visit_id', 'client', 'observation_code', 'observation_value']

    # Check columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Check for nulls
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        print(f"Warning: Found null values:\n{null_counts}")

    # Check data types
    if not pd.api.types.is_numeric_dtype(df['observation_value']):
        raise ValueError("observation_value must be numeric")

    return True
```

### 2. Logging

**Log predictions for audit trail:**
```python
import logging

logging.basicConfig(
    filename='predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# After prediction
for idx, row in predictions.iterrows():
    logging.info(
        f"Visit {row['visit_id']}: "
        f"Prediction={row['prediction']}, "
        f"Probability={row['probability']:.4f}"
    )
```

### 3. Monitoring

**Track prediction distribution:**
```python
import matplotlib.pyplot as plt

# Monitor probability distribution over time
predictions['date'] = pd.to_datetime(predictions['timestamp'])
daily_avg = predictions.groupby(predictions['date'].dt.date)['probability'].mean()

plt.plot(daily_avg)
plt.xlabel('Date')
plt.ylabel('Average Probability')
plt.title('Daily Average Sepsis Risk')
plt.savefig('monitoring/risk_trend.png')
```

### 4. Error Handling

**Robust error handling:**
```python
def safe_predict(pipeline, df):
    try:
        validate_data(df)
        predictions = pipeline.predict(df, return_probabilities=True)
        return predictions, None
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return None, str(e)

# Use it
predictions, error = safe_predict(pipeline, df)
if error:
    print(f"Error: {error}")
else:
    print(f"Success: {len(predictions)} predictions made")
```

## Model Retraining

When to retrain:
- **Data drift detected** - Prediction accuracy degrades
- **New data available** - More training examples
- **Domain changes** - New protocols or observation types

Retraining process:
```bash
# 1. Add new data to training set
cat new_data.csv >> data/data_science_project_data.csv

# 2. Re-run optimization
python generate_features.py

# 3. Train new model
python train_best_model.py --full

# 4. Test new model
python predict.py --test

# 5. If performance is good, deploy
mv data/models/best_model.pkl data/models/best_model_$(date +%Y%m%d).pkl.backup
# New model is already at data/models/best_model.pkl
```

## Support

For issues or questions:
- Check [QUICK_START.md](QUICK_START.md) for setup
- See [FEATURE_PIPELINE_GUIDE.md](FEATURE_PIPELINE_GUIDE.md) for pipeline details
- Review [APP_USAGE.md](APP_USAGE.md) for training
- Run tests: `python run_tests.py`

---

**Created:** December 2024
**Purpose:** Production inference for sepsis prediction model
**Related Files:**
- [predict.py](../predict.py) - Main inference script
- [feature_pipeline.py](../feature_pipeline.py) - Pipeline implementation
- [train_best_model.py](../train_best_model.py) - Model training
