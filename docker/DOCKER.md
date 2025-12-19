# Docker Deployment Guide

This project includes Dockerized microservices for the ML pipeline. Each service can run independently or as part of a complete pipeline.

## Architecture

The system consists of three microservices:

1. **Feature Optimization Service** (`Dockerfile.optimize`)
   - Runs automated feature engineering experiments
   - Finds optimal feature sets and hyperparameters
   - Outputs: `data/models/best_model_config.json`, `data/models/optimization_log.json`

2. **Model Training Service** (`Dockerfile.train`)
   - Trains the best model using optimized configuration
   - Outputs: `data/models/best_model.pkl`, `data/models/feature_names.json`

3. **Prediction Service** (`Dockerfile.predict`)
   - Loads trained model and makes predictions on new data
   - Outputs: `data/predictions/predictions.csv`

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available for Docker
- Input data file: `data/data_science_project_data.csv`

## Quick Start

### Run Complete Pipeline

Execute all three services in sequence:

```bash
docker-compose --profile pipeline up
```

This will:
1. Optimize features and hyperparameters
2. Train the best model
3. Generate predictions

### Run Individual Services

#### 1. Feature Optimization Only

```bash
docker-compose --profile optimize up
```

**Output:**
- `data/models/best_model_config.json` - Best configuration found
- `data/models/optimization_log.json` - Complete experiment history

#### 2. Model Training Only

Requires existing optimization results.

```bash
docker-compose --profile train up
```

**Output:**
- `data/models/best_model.pkl` - Trained XGBoost model
- `data/models/feature_names.json` - Feature information
- `data/models/production_model_info.txt` - Model metadata

#### 3. Prediction Only

Requires trained model.

```bash
docker-compose --profile predict up
```

**Output:**
- `data/predictions/predictions.csv` - Predictions with probabilities

## Building Images

### Build All Images

```bash
docker-compose build
```

### Build Individual Services

```bash
# Feature optimization
docker build -f Dockerfile.optimize -t automl-optimize .

# Model training
docker build -f Dockerfile.train -t automl-train .

# Prediction service
docker build -f Dockerfile.predict -t automl-predict .
```

## Running Services Directly with Docker

### Feature Optimization

```bash
docker run -v $(pwd)/data:/app/data automl-optimize
```

### Model Training

```bash
docker run -v $(pwd)/data:/app/data automl-train
```

### Prediction

```bash
docker run -v $(pwd)/data:/app/data \
  automl-predict \
  python predict.py \
    --input data/data_science_project_data.csv \
    --output data/predictions/predictions.csv \
    --probabilities
```

## Custom Configurations

### Override Default Commands

#### Run optimization with specific iterations

```bash
docker-compose run --rm optimize \
  python generate_features_optimized.py
```

#### Train without test split

```bash
docker-compose run --rm train \
  python train_best_model.py
```

#### Predict on custom data

```bash
docker-compose run --rm predict \
  python predict.py \
    --input data/my_custom_data.csv \
    --output data/predictions/my_predictions.csv \
    --probabilities
```

### Environment Variables

Set in `docker-compose.yml` or pass at runtime:

```bash
docker-compose run -e PYTHONUNBUFFERED=1 optimize
```

## Volume Mounts

All services mount `./data` to `/app/data` in the container. This ensures:

- Input data is accessible to containers
- Output files persist after container stops
- Models can be shared between services

**Directory structure:**
```
data/
├── data_science_project_data.csv    # Input data
├── models/                           # Model artifacts
│   ├── best_model.pkl
│   ├── best_model_config.json
│   ├── feature_names.json
│   └── optimization_log.json
└── predictions/                      # Prediction outputs
    └── predictions.csv
```

## Production Deployment

### Standalone Prediction API (Future)

The prediction service exposes port 8000 for potential API integration:

```yaml
predict:
  ports:
    - "8000:8000"
```

### Resource Limits

Add resource constraints for production:

```yaml
services:
  optimize:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

## Troubleshooting

### Common Issues

**1. "Configuration not found" error in training service**

Solution: Run optimization first:
```bash
docker-compose --profile optimize up
```

**2. "Model not found" error in prediction service**

Solution: Run training first:
```bash
docker-compose --profile train up
```

**3. Out of memory errors**

Solution: Increase Docker memory limit in Docker Desktop settings (recommend 8GB+)

**4. Permission errors on data directory**

Solution: Ensure data directory has correct permissions:
```bash
chmod -R 755 data/
```

### Viewing Logs

```bash
# View logs from all services
docker-compose logs

# Follow logs from specific service
docker-compose logs -f optimize

# View logs from stopped containers
docker-compose logs pipeline
```

### Cleanup

```bash
# Remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: ML Pipeline

on: [push]

jobs:
  optimize-and-train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build images
        run: docker-compose build

      - name: Run optimization
        run: docker-compose --profile optimize up

      - name: Train model
        run: docker-compose --profile train up

      - name: Archive model
        uses: actions/upload-artifact@v3
        with:
          name: trained-model
          path: data/models/
```

## Performance Notes

### Typical Runtimes

- **Feature Optimization**: 10-30 minutes (depends on iterations)
- **Model Training**: 1-3 minutes
- **Prediction**: <1 minute for 50k records

### Optimization Tips

1. **Use Docker BuildKit** for faster builds:
   ```bash
   DOCKER_BUILDKIT=1 docker-compose build
   ```

2. **Use volume caching** for pip packages:
   ```yaml
   volumes:
     - pip-cache:/root/.cache/pip
   ```

3. **Parallel builds**:
   ```bash
   docker-compose build --parallel
   ```

## Security Considerations

- Images use Python 3.11-slim (minimal attack surface)
- No root user execution (can be added if needed)
- Secrets should be passed via environment variables, not baked into images
- For production, use private registry and signed images

## Next Steps

- Add health check endpoints to services
- Implement RESTful API for prediction service
- Add model versioning and A/B testing support
- Integrate with Kubernetes for orchestration
- Add monitoring with Prometheus/Grafana

## Support

For issues or questions:
- Check logs: `docker-compose logs -f [service-name]`
- Verify data directory structure
- Ensure sufficient Docker resources
- Review error messages in service output
