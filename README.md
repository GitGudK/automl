# AutoML: Automated Feature Engineering & Model Optimization

An automated machine learning framework for time-series medical data that discovers optimal features, trains models, and deploys to production infrastructure.

## Overview

AutoML is a complete end-to-end ML pipeline that automates:
- **Feature Engineering**: Automatically creates and tests feature combinations
- **Model Training**: Tests multiple algorithms and hyperparameters
- **Feature Selection**: Identifies the most predictive features
- **Visualization**: Interactive dashboards for experiment tracking
- **Deployment**: Docker containers and AWS infrastructure

## Quick Start

### 1. Generate Synthetic Data (Demo)

```bash
python generate_synthetic_data.py
```

This creates a placeholder dataset in `data/data_science_project_data.csv` with medical time-series observations.

### 2. Run Feature Optimization

```bash
python generate_features_optimized.py
```

This will:
- Test 100+ feature combinations
- Try different models (XGBoost, Random Forest)
- Apply feature selection algorithms
- Track all experiments in `data/models/optimization_log.json`

### 3. Train Best Model

```bash
python train_best_model.py --full
```

Trains the best-performing model on the full dataset and saves to `data/models/best_model.pkl`.

### 4. Make Predictions

```bash
python predict.py \
  --input data/data_science_project_data.csv \
  --output data/predictions/predictions.csv \
  --probabilities
```

### 5. View Results Dashboard

```bash
cd streamlit
streamlit run streamlit_app.py
```

Interactive dashboard at `http://localhost:8501`

## Project Structure

See `docs/` for detailed documentation:
- [Quick Start Guide](docs/QUICK_START.md)
- [Feature Pipeline Guide](docs/FEATURE_PIPELINE_GUIDE.md)
- [Docker Deployment](docker/DOCKER.md)
- [AWS Deployment](terraform/README.md)

## Data Schema

Input CSV must have these columns:

```
client,visit_id,observation_type_code,observation_value,reference_range_min,reference_range_max,condition_present,observation_date
```

## Results

- **ROC AUC**: > 0.90
- **Average Precision**: > 0.55  
- **Features**: 150 selected from 500+ tested

## Docker Usage

```bash
./docker-run.sh all      # Complete pipeline
./docker-run.sh optimize # Feature optimization only
./docker-run.sh train    # Training only
./docker-run.sh predict  # Predictions only
```

## AWS Deployment

```bash
cd terraform
./deploy.sh full
```

## Requirements

```bash
pip install -r requirements.txt
```

## License

Open source - modify and use as needed.
