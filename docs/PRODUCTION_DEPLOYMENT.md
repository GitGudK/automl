# Production Deployment Guide

## Complete Workflow: Training to Production

### Phase 1: Development & Optimization

#### Step 1: Run Optimization
```bash
python generate_features.py
```

**What it does:**
- Tests 10 different feature combinations
- Trains 80 models with various configurations
- Finds best combination of features and model hyperparameters
- Maximizes ROC AUC + Average Precision

**Outputs:**
- `models/best_model_config.json` - Best configuration found
- `models/optimization_log.json` - Complete experiment history
- `models/optimization_summary.txt` - Human-readable summary

**Duration:** 5-15 minutes depending on data size

---

### Phase 2: Model Preparation

#### Step 2: Save Production Model
```bash
python train_best_model.py --full
```

**What it does:**
- Loads best configuration from optimization
- Recreates exact feature pipeline
- Trains model on FULL dataset (no holdout)
- Saves model and metadata for production

**Outputs:**
- `models/best_model.pkl` - Trained model ready for inference
- `models/feature_names.json` - List of features
- `models/production_model_info.txt` - Model documentation

**Duration:** 1-3 minutes

---

### Phase 3: Testing & Validation

#### Step 3: Test Feature Reproduction
```bash
python run_tests.py pipeline
```

**What it does:**
- Verifies feature pipeline works correctly
- Tests on subset of data
- Validates feature names match training
- Creates sample new patient data

**Outputs:**
- Console output showing test results
- `data/sample_new_patients.csv` - Sample data for testing

---

### Phase 4: Production Deployment

#### Step 4: Deploy to Production

Two deployment options:

##### Option A: Feature Engineering Only

Use when you want to create features separately from predictions (e.g., batch processing):

```python
from feature_pipeline import FeatureReproducer
import pandas as pd

# Initialize once
reproducer = FeatureReproducer('models/best_model_config.json')

# Process new patients
df_new = pd.read_csv('new_patients.csv')
df_features = reproducer.transform(df_new, return_all_features=False)

# Save features
df_features.to_csv('patient_features.csv', index=False)

# Later, use features with your own prediction logic
```

##### Option B: Complete Pipeline (Recommended)

Use for end-to-end predictions:

```python
from feature_pipeline import ProductionPipeline
import pandas as pd

# Initialize once at application startup
pipeline = ProductionPipeline(
    config_path='models/best_model_config.json',
    model_path='models/best_model.pkl'
)

# For each batch of new patients
df_new = pd.read_csv('new_patients.csv')
predictions = pipeline.predict(df_new, return_probabilities=True)

# predictions contains: visit_id, client, sepsis_probability
```

---

## Production Integration Patterns

### Pattern 1: REST API Service

```python
# api_server.py
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
def predict_sepsis():
    """
    Expects JSON with patient observations:
    [
        {
            "visit_id": 1001,
            "client": "C001",
            "observation_date": "2024-01-15 08:00:00",
            "observation_type_code": "TEMP",
            "observation_value": 38.5,
            "reference_range_min": 36.0,
            "reference_range_max": 37.5
        },
        ...
    ]
    """
    try:
        # Parse request
        data = request.json
        df = pd.DataFrame(data)

        # Get predictions
        predictions = pipeline.predict(df, return_probabilities=True)

        # Return results
        return jsonify({
            'status': 'success',
            'predictions': predictions.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Deploy:**
```bash
# Development
python api_server.py

# Production with gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

**Test:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @new_patient_data.json
```

---

### Pattern 2: Batch Processing Service

```python
# batch_processor.py
from feature_pipeline import ProductionPipeline
import pandas as pd
import glob
import os
from datetime import datetime

class BatchProcessor:
    def __init__(self, config_path, model_path):
        self.pipeline = ProductionPipeline(config_path, model_path)

    def process_directory(self, input_dir, output_dir):
        """Process all CSV files in input directory"""
        os.makedirs(output_dir, exist_ok=True)

        for file_path in glob.glob(f'{input_dir}/*.csv'):
            print(f"Processing {file_path}...")

            try:
                # Load data
                df = pd.read_csv(file_path)

                # Get predictions
                predictions = self.pipeline.predict(
                    df,
                    return_probabilities=True
                )

                # Add metadata
                predictions['processed_at'] = datetime.now()
                predictions['model_version'] = '1.0'

                # Save results
                filename = os.path.basename(file_path)
                output_path = os.path.join(
                    output_dir,
                    f'predictions_{filename}'
                )
                predictions.to_csv(output_path, index=False)

                print(f"  ✓ Saved to {output_path}")

            except Exception as e:
                print(f"  ✗ Error: {e}")

if __name__ == '__main__':
    processor = BatchProcessor(
        config_path='models/best_model_config.json',
        model_path='models/best_model.pkl'
    )

    processor.process_directory(
        input_dir='data/incoming',
        output_dir='data/predictions'
    )
```

**Run:**
```bash
# Process once
python batch_processor.py

# Schedule with cron (every hour)
0 * * * * cd /path/to/automl && python batch_processor.py
```

---

### Pattern 3: Database Integration

```python
# db_predictor.py
from feature_pipeline import ProductionPipeline
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

class DatabasePredictor:
    def __init__(self, db_url, config_path, model_path):
        self.engine = create_engine(db_url)
        self.pipeline = ProductionPipeline(config_path, model_path)

    def process_pending_visits(self):
        """Process visits that haven't been scored yet"""

        # Query pending visits
        query = """
            SELECT
                visit_id, client, observation_date,
                observation_type_code, observation_value,
                reference_range_min, reference_range_max
            FROM patient_observations
            WHERE visit_id IN (
                SELECT visit_id FROM visits
                WHERE sepsis_score IS NULL
            )
        """

        df = pd.read_sql(query, self.engine)

        if len(df) == 0:
            print("No pending visits to process")
            return

        print(f"Processing {df['visit_id'].nunique()} visits...")

        # Get predictions
        predictions = self.pipeline.predict(
            df,
            return_probabilities=True
        )

        # Add timestamp
        predictions['scored_at'] = datetime.now()

        # Update database
        predictions.to_sql(
            'sepsis_predictions',
            self.engine,
            if_exists='append',
            index=False
        )

        # Update visits table
        for _, row in predictions.iterrows():
            update_query = """
                UPDATE visits
                SET
                    sepsis_score = :score,
                    risk_level = :risk_level,
                    scored_at = :scored_at
                WHERE visit_id = :visit_id
            """

            risk_level = 'HIGH' if row['sepsis_probability'] > 0.7 else \
                        'MEDIUM' if row['sepsis_probability'] > 0.4 else 'LOW'

            self.engine.execute(
                update_query,
                score=row['sepsis_probability'],
                risk_level=risk_level,
                scored_at=row['scored_at'],
                visit_id=row['visit_id']
            )

        print(f"✓ Updated {len(predictions)} visits")

if __name__ == '__main__':
    predictor = DatabasePredictor(
        db_url='postgresql://user:pass@localhost/hospital_db',
        config_path='models/best_model_config.json',
        model_path='models/best_model.pkl'
    )

    predictor.process_pending_visits()
```

---

### Pattern 4: Real-time Stream Processing

```python
# stream_processor.py
from feature_pipeline import ProductionPipeline
import pandas as pd
from collections import defaultdict
import time

class StreamProcessor:
    def __init__(self, config_path, model_path):
        self.pipeline = ProductionPipeline(config_path, model_path)
        self.observation_buffer = defaultdict(list)
        self.min_observations = 5  # Minimum observations before prediction

    def add_observation(self, observation):
        """
        Add a single observation to the buffer.

        observation: dict with keys:
            visit_id, client, observation_date, observation_type_code,
            observation_value, reference_range_min, reference_range_max
        """
        visit_id = observation['visit_id']
        self.observation_buffer[visit_id].append(observation)

    def should_predict(self, visit_id):
        """Check if we have enough data to make a prediction"""
        return len(self.observation_buffer[visit_id]) >= self.min_observations

    def predict_visit(self, visit_id):
        """Make prediction for a specific visit"""

        if not self.should_predict(visit_id):
            return None

        # Convert observations to DataFrame
        df = pd.DataFrame(self.observation_buffer[visit_id])

        # Get prediction
        prediction = self.pipeline.predict(
            df,
            return_probabilities=True
        )

        result = {
            'visit_id': visit_id,
            'client': prediction['client'].iloc[0],
            'sepsis_probability': prediction['sepsis_probability'].iloc[0],
            'num_observations': len(self.observation_buffer[visit_id]),
            'timestamp': time.time()
        }

        return result

    def clear_buffer(self, visit_id):
        """Clear observations for a visit after processing"""
        if visit_id in self.observation_buffer:
            del self.observation_buffer[visit_id]

# Example usage
processor = StreamProcessor(
    config_path='models/best_model_config.json',
    model_path='models/best_model.pkl'
)

# As observations stream in:
# processor.add_observation({...})
#
# if processor.should_predict(visit_id):
#     result = processor.predict_visit(visit_id)
#     send_alert_if_high_risk(result)
#     processor.clear_buffer(visit_id)
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] Run optimization (`python generate_features.py`)
- [ ] Review results in `models/optimization_summary.txt`
- [ ] Save production model (`python train_best_model.py --full`)
- [ ] Test feature pipeline (`python test_feature_pipeline.py`)
- [ ] Validate predictions on known data
- [ ] Document model version and performance metrics

### Model Files to Deploy

Required files:
- [ ] `models/best_model.pkl`
- [ ] `models/best_model_config.json`
- [ ] `models/optimization_log.json`
- [ ] `feature_pipeline.py`

Optional but recommended:
- [ ] `models/feature_names.json`
- [ ] `models/production_model_info.txt`

### Environment Setup

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Set up proper Python version (3.9+)
- [ ] Configure file paths for your environment
- [ ] Set up logging
- [ ] Configure monitoring/alerting

### Testing

- [ ] Test with sample data
- [ ] Verify feature counts match training
- [ ] Check prediction latency
- [ ] Test error handling
- [ ] Load test with expected volume

### Monitoring

- [ ] Log all predictions
- [ ] Track prediction distribution
- [ ] Monitor for data drift
- [ ] Alert on errors
- [ ] Track model performance over time

---

## Performance Optimization

### Caching

```python
from functools import lru_cache
from feature_pipeline import ProductionPipeline

@lru_cache(maxsize=1)
def get_pipeline():
    """Cache pipeline initialization"""
    return ProductionPipeline(
        config_path='models/best_model_config.json',
        model_path='models/best_model.pkl'
    )

# Use cached pipeline
pipeline = get_pipeline()
predictions = pipeline.predict(df_new)
```

### Batch Processing

```python
# Process in batches instead of one at a time
batch_size = 100
for i in range(0, len(all_visits), batch_size):
    batch = all_visits[i:i+batch_size]
    predictions = pipeline.predict(batch)
    save_predictions(predictions)
```

### Parallel Processing

```python
from multiprocessing import Pool
from feature_pipeline import ProductionPipeline

def process_file(file_path):
    pipeline = ProductionPipeline(
        config_path='models/best_model_config.json',
        model_path='models/best_model.pkl'
    )
    df = pd.read_csv(file_path)
    return pipeline.predict(df)

# Process multiple files in parallel
with Pool(processes=4) as pool:
    results = pool.map(process_file, file_list)
```

---

## Model Versioning

```python
# Version your models
import shutil
from datetime import datetime

version = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save with version
shutil.copy(
    'models/best_model.pkl',
    f'models/archive/best_model_v{version}.pkl'
)

# Track in metadata
metadata = {
    'version': version,
    'roc_auc': 0.9123,
    'avg_precision': 0.9111,
    'training_date': datetime.now().isoformat()
}
```

---

## Troubleshooting Production Issues

### Issue: Predictions are slow

**Solutions:**
1. Initialize pipeline once, not per request
2. Use batch processing
3. Cache pipeline instance
4. Consider using async processing

### Issue: Memory usage is high

**Solutions:**
1. Process in smaller batches
2. Clear observation buffers regularly
3. Use generators for large datasets
4. Monitor and set memory limits

### Issue: Feature mismatch errors

**Causes:**
- New data has different observation types
- Column names don't match

**Solutions:**
- Validate input data format
- Use `return_all_features=False` (default)
- Missing features are automatically filled with 0

### Issue: Model performance degrading

**Monitoring:**
```python
# Track predictions over time
predictions['date'] = pd.to_datetime('today')
predictions.to_csv('predictions_log.csv', mode='a')

# Periodically check distribution
recent = pd.read_csv('predictions_log.csv')
recent['sepsis_probability'].hist()
```

**Actions:**
- Review recent prediction distribution
- Check for data drift
- Retrain model if needed
- Update feature pipeline

---

## Security Considerations

1. **Model File Security**
   - Store model files in secure location
   - Limit access permissions
   - Version control model files
   - Encrypt sensitive data

2. **API Security**
   - Use authentication/authorization
   - Validate all inputs
   - Rate limit requests
   - Log all access

3. **Data Privacy**
   - Anonymize patient data when possible
   - Comply with HIPAA/GDPR
   - Secure data transmission
   - Audit data access

---

## Complete Production Example

See `examples/production_system.py` for a complete, production-ready implementation combining all patterns above.

---

## Support & Maintenance

### Regular Tasks
- Monitor prediction accuracy
- Review error logs
- Update model quarterly
- Test with new data
- Document changes

### When to Retrain
- Performance drops below threshold
- Significant data drift detected
- New observation types added
- Quarterly scheduled retraining

### Retraining Process
1. Collect new labeled data
2. Run optimization with new data
3. Compare to existing model
4. Test thoroughly
5. Deploy new version
6. Monitor closely

---

For detailed API documentation, see [FEATURE_PIPELINE_GUIDE.md](FEATURE_PIPELINE_GUIDE.md)
