# AutoML Setup Guide

## Quick Setup

This repository is ready to use with synthetic placeholder data.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python generate_synthetic_data.py
```

This creates `data/data_science_project_data.csv` with ~25K observations across 500 visits.

### 3. Run Your First Experiment

```bash
# Quick test (5 iterations)
python generate_features_optimized.py --iterations 5

# Full optimization (100+ iterations, ~20 minutes)
python generate_features_optimized.py
```

### 4. Train Best Model

```bash
python train_best_model.py --full
```

### 5. View Results

```bash
cd streamlit
streamlit run streamlit_app.py
```

Open browser to `http://localhost:8501`

## Using Your Own Data

### Data Format

Replace `data/data_science_project_data.csv` with your data in this format:

```csv
entity,session_id,metric_type,metric_value,threshold_lower,threshold_upper,flag_positive,timestamp
entity_A,1001,heartRate,75,,,FALSE,2023-01-01T10:00:00Z
entity_A,1001,temperature,98.6,,,FALSE,2023-01-01T10:05:00Z
entity_A,1002,heartRate,120,,,TRUE,2023-01-02T14:30:00Z
```

**Required columns:**
- `entity`: Group/entity identifier (string)
- `session_id`: Unique session/visit identifier (int or string)
- `metric_type`: Type of measurement (string)
- `metric_value`: Measured value (float)
- `threshold_lower`: Normal range minimum (float, optional)
- `threshold_upper`: Normal range maximum (float, optional)
- `flag_positive`: Target variable (TRUE/FALSE or 1/0)
- `timestamp`: ISO timestamp (string)

### Customization

**Change feature types**: Edit `feature_pipeline.py`
**Change models**: Edit `generate_features_optimized.py`
**Change target**: Use your own binary `flag_positive` column

## Docker Deployment

```bash
cd docker
./docker-run.sh build
./docker-run.sh all
```

## AWS Deployment

```bash
cd terraform
./deploy.sh init
./deploy.sh plan
./deploy.sh apply
./deploy.sh build
```

## Next Steps

1. See [README.md](README.md) for overview
2. See [docs/QUICK_START.md](docs/QUICK_START.md) for detailed guide
3. See [docs/FEATURE_PIPELINE_GUIDE.md](docs/FEATURE_PIPELINE_GUIDE.md) for customization
4. See [docker/DOCKER.md](docker/DOCKER.md) for containerization
5. See [terraform/README.md](terraform/README.md) for cloud deployment
