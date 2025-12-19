# Sepsis Prediction Model - Inference

Quick reference for making predictions with the trained model.

## Setup

### 1. Train the Model

```bash
# Find best configuration
python generate_features.py

# Train and save model
python train_best_model.py --full
```

This creates:
- `data/models/best_model_config.json`
- `data/models/best_model.pkl`
- `data/models/feature_names.json`

## Making Predictions

### Command Line Interface

```bash
# Basic prediction on CSV file
python predict.py --input new_data.csv --output predictions.csv

# Include probability scores
python predict.py --input new_data.csv --output predictions.csv --probabilities

# Evaluate on test set
python predict.py --test

# Interactive mode
python predict.py --interactive
```

### Python API

```python
from feature_pipeline import ProductionPipeline
import pandas as pd

# Load model
pipeline = ProductionPipeline(
    config_path='data/models/best_model_config.json',
    model_path='data/models/best_model.pkl'
)

# Load data
df = pd.read_csv('new_patients.csv')

# Predict with probabilities
predictions = pipeline.predict(df, return_probabilities=True)

# Results have columns: visit_id, client, sepsis_probability
print(predictions)
```

## Input Data Format

Your CSV must have these columns:
- `visit_id` - Unique patient visit identifier
- `client` - Client identifier
- `observation_code` - Medical observation code
- `observation_value` - Numeric value

Example:
```csv
visit_id,client,observation_code,observation_value
V001,C1,HR,85
V001,C1,TEMP,98.6
V001,C1,BP_SYS,120
V002,C1,HR,110
V002,C1,TEMP,102.3
```

## Output Format

### With Probabilities (`--probabilities`)

```csv
visit_id,client,probability,prediction
V001,C1,0.1234,0
V002,C1,0.8567,1
```

### Without Probabilities

```csv
visit_id,client,prediction
V001,C1,0
V002,C1,1
```

- `prediction`: 0 = No sepsis, 1 = Sepsis
- `probability`: Risk score from 0 to 1

## Risk Interpretation

- **0.0 - 0.3**: Low risk
- **0.3 - 0.7**: Medium risk
- **0.7 - 1.0**: High risk

## Examples

### Batch Processing

```bash
# Process multiple files
for file in data/incoming/*.csv; do
    python predict.py \
        --input "$file" \
        --output "predictions/$(basename $file)" \
        --probabilities
done
```

### Filter High-Risk Cases

```python
from feature_pipeline import ProductionPipeline
import pandas as pd

pipeline = ProductionPipeline(
    config_path='data/models/best_model_config.json',
    model_path='data/models/best_model.pkl'
)

df = pd.read_csv('patients.csv')
predictions = pipeline.predict(df, return_probabilities=True)

# Filter high-risk (> 70% probability)
high_risk = predictions[predictions['sepsis_probability'] > 0.7]
print(f"High-risk cases: {len(high_risk)}")
print(high_risk[['visit_id', 'sepsis_probability']])
```

### REST API

```python
from flask import Flask, request, jsonify
from feature_pipeline import ProductionPipeline
import pandas as pd

app = Flask(__name__)
pipeline = ProductionPipeline(
    config_path='data/models/best_model_config.json',
    model_path='data/models/best_model.pkl'
)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    result = pipeline.predict(df, return_probabilities=True)
    return jsonify(result.to_dict('records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Test with:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "visit_id": ["V001", "V001", "V001"],
    "client": ["C1", "C1", "C1"],
    "observation_code": ["HR", "TEMP", "BP_SYS"],
    "observation_value": [105, 101.2, 150]
  }'
```

## Documentation

- [INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md) - Complete inference documentation
- [FEATURE_PIPELINE_GUIDE.md](docs/FEATURE_PIPELINE_GUIDE.md) - Pipeline details
- [QUICK_START.md](docs/QUICK_START.md) - Getting started guide

## Troubleshooting

**"Model not found"**
```bash
python train_best_model.py --full
```

**"Configuration not found"**
```bash
python generate_features.py
```

**Import errors**
```bash
pip install -r requirements.txt
```

**Low accuracy**
- Check if test data matches training data format
- Retrain model with more recent data
- Verify observation codes are consistent

---

For detailed usage examples and integration patterns, see [docs/INFERENCE_GUIDE.md](docs/INFERENCE_GUIDE.md)
