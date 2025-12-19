#!/bin/bash
# AutoML Docker Helper Script
# Quick commands for running Docker services

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose not found. Please install Docker Compose."
    exit 1
fi

# Display help
show_help() {
    cat << EOF
AutoML ML Pipeline - Docker Helper

Usage: ./docker-run.sh [command]

Commands:
  all              Run complete pipeline (optimize -> train -> predict)
  optimize         Run feature optimization only
  train            Run model training only
  predict          Run prediction service only
  build            Build all Docker images
  clean            Remove all containers and images
  logs [service]   Show logs (optional: specify service name)
  help             Show this help message

Examples:
  ./docker-run.sh all         # Run full pipeline
  ./docker-run.sh optimize    # Just optimize features
  ./docker-run.sh logs train  # View training logs
EOF
}

# Main script logic
case "$1" in
    all|pipeline)
        print_header "Running Complete ML Pipeline"
        print_info "This will run: Optimization -> Training -> Prediction"
        docker-compose --profile pipeline up
        print_success "Pipeline completed!"
        print_info "Check data/predictions/predictions.csv for results"
        ;;

    optimize)
        print_header "Running Feature Optimization"
        docker-compose --profile optimize up
        print_success "Optimization completed!"
        print_info "Check data/models/best_model_config.json"
        ;;

    train)
        print_header "Running Model Training"
        docker-compose --profile train up
        print_success "Training completed!"
        print_info "Check data/models/best_model.pkl"
        ;;

    predict)
        print_header "Running Prediction Service"
        docker-compose --profile predict up
        print_success "Predictions completed!"
        print_info "Check data/predictions/predictions.csv"
        ;;

    build)
        print_header "Building Docker Images"
        docker-compose build --parallel
        print_success "All images built successfully!"
        ;;

    clean)
        print_header "Cleaning Up Docker Resources"
        docker-compose down --rmi all -v
        print_success "Cleanup completed!"
        ;;

    logs)
        if [ -n "$2" ]; then
            docker-compose logs -f "$2"
        else
            docker-compose logs
        fi
        ;;

    help|--help|-h|"")
        show_help
        ;;

    *)
        echo "Unknown command: $1"
        echo "Run './docker-run.sh help' for usage information"
        exit 1
        ;;
esac
