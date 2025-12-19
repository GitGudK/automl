# Quick Start Guide

## Installation & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
# Standard mode (10 iterations)
python generate_features.py

# Or run continuously for 30 minutes
python generate_features.py --duration 30
```

That's it! The app will automatically:
- Test 10 feature combinations (or run for specified duration)
- Train 80 models
- Find the best configuration
- Save results to `models/` directory

**Tip**: Use `--duration` to cycle through standard combinations repeatedly with different random interaction samples!

## What to Check

### During Execution
Watch for:
- Progress through iterations (1-10)
- Model scores being printed
- Best score updates

### After Completion
Check these files in `models/`:

1. **optimization_summary.txt** - Start here for quick overview
2. **best_model_config.json** - Best model details
3. **optimization_log.json** - Complete experiment data

## Expected Runtime
- Small dataset (<1000 visits): ~5 minutes
- Medium dataset (1000-5000 visits): ~10 minutes
- Large dataset (>5000 visits): ~15+ minutes

## Understanding Results

### Good Performance
- Combined Score: > 1.6
- ROC AUC: > 0.8
- Average Precision: > 0.7

### Excellent Performance
- Combined Score: > 1.8
- ROC AUC: > 0.9
- Average Precision: > 0.85

## Production Deployment

### 3. Save the best model
```bash
python train_best_model.py --full
```

### 4. Use for new patients
```python
from feature_pipeline import ProductionPipeline

pipeline = ProductionPipeline(
    config_path='models/best_model_config.json',
    model_path='models/best_model.pkl'
)

predictions = pipeline.predict(new_patient_data)
```

See [FEATURE_PIPELINE_GUIDE.md](FEATURE_PIPELINE_GUIDE.md) for complete production usage.

## Common Issues

**Import errors?**
```bash
pip install -r requirements.txt
```

**Data not found?**
Ensure `data/data_science_project_data.csv` exists

**Too slow?**
Edit [generate_features.py](generate_features.py:1) line 474 to reduce iterations:
```python
optimizer.run_optimization(n_iterations=5)  # default: 10
```

## More Information

- Full usage guide: [APP_USAGE.md](APP_USAGE.md)
- Complete overview: [APP_OVERVIEW.md](APP_OVERVIEW.md)
- Production guide: [FEATURE_PIPELINE_GUIDE.md](FEATURE_PIPELINE_GUIDE.md)
- Run tests: `python run_tests.py`
