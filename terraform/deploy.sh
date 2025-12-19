#!/bin/bash
# Deployment script for AutoML ML Pipeline on AWS

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI not found. Please install it first."
        exit 1
    fi
    print_success "AWS CLI found"

    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        print_error "Terraform not found. Please install it first."
        exit 1
    fi
    print_success "Terraform found"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install it first."
        exit 1
    fi
    print_success "Docker found"

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured. Run 'aws configure' first."
        exit 1
    fi
    print_success "AWS credentials configured"
}

# Initialize Terraform
init_terraform() {
    print_header "Initializing Terraform"
    cd terraform
    terraform init
    print_success "Terraform initialized"
    cd ..
}

# Plan Terraform changes
plan_terraform() {
    print_header "Planning Terraform Changes"
    cd terraform
    terraform plan -out=tfplan
    print_success "Terraform plan created"
    cd ..
}

# Apply Terraform
apply_terraform() {
    print_header "Applying Terraform Configuration"
    cd terraform
    terraform apply tfplan
    print_success "Infrastructure deployed"
    cd ..
}

# Build and push Docker images
build_and_push() {
    print_header "Building and Pushing Docker Images"

    # Get ECR URLs from Terraform output
    cd terraform
    OPTIMIZE_REPO=$(terraform output -raw ecr_repository_optimize_url)
    TRAIN_REPO=$(terraform output -raw ecr_repository_train_url)
    PREDICT_REPO=$(terraform output -raw ecr_repository_predict_url)
    PIPELINE_REPO=$(terraform output -raw ecr_repository_pipeline_url)
    AWS_REGION=$(terraform output -raw aws_region 2>/dev/null || echo "us-east-1")
    cd ..

    # Login to ECR
    print_info "Logging in to ECR..."
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${OPTIMIZE_REPO%%/*}
    print_success "Logged in to ECR"

    # Build images
    print_info "Building Docker images..."
    docker build -f Dockerfile.optimize -t automl-optimize .
    docker build -f Dockerfile.train -t automl-train .
    docker build -f Dockerfile.predict -t automl-predict .
    docker build -f Dockerfile.pipeline -t automl-pipeline .
    print_success "Docker images built"

    # Tag images
    print_info "Tagging images..."
    docker tag automl-optimize:latest $OPTIMIZE_REPO:latest
    docker tag automl-train:latest $TRAIN_REPO:latest
    docker tag automl-predict:latest $PREDICT_REPO:latest
    docker tag automl-pipeline:latest $PIPELINE_REPO:latest
    print_success "Images tagged"

    # Push images
    print_info "Pushing images to ECR..."
    docker push $OPTIMIZE_REPO:latest
    docker push $TRAIN_REPO:latest
    docker push $PREDICT_REPO:latest
    docker push $PIPELINE_REPO:latest
    print_success "Images pushed to ECR"
}

# Upload data to S3
upload_data() {
    print_header "Uploading Data to S3"

    cd terraform
    S3_BUCKET=$(terraform output -raw s3_bucket_name)
    cd ..

    if [ -f "data/data_science_project_data.csv" ]; then
        print_info "Uploading training data to S3..."
        aws s3 cp data/data_science_project_data.csv s3://$S3_BUCKET/data/
        print_success "Training data uploaded"
    else
        print_info "No training data found, skipping upload"
    fi
}

# Run optimization task
run_optimize() {
    print_header "Running Optimization Task"

    cd terraform
    CLUSTER_NAME=$(terraform output -raw ecs_cluster_name)
    TASK_DEF=$(terraform output -raw ecs_task_definition_optimize_arn | cut -d'/' -f2)
    SUBNETS=$(terraform output -json private_subnet_ids | jq -r 'join(",")')
    SG=$(terraform output -raw ecs_tasks_security_group_id)
    cd ..

    print_info "Starting optimization task..."
    TASK_ARN=$(aws ecs run-task \
        --cluster $CLUSTER_NAME \
        --task-definition $TASK_DEF \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG],assignPublicIp=DISABLED}" \
        --query 'tasks[0].taskArn' \
        --output text)

    print_success "Optimization task started: $TASK_ARN"
    print_info "Monitor logs: aws logs tail /ecs/automl-optimize --follow"
}

# Show outputs
show_outputs() {
    print_header "Deployment Summary"
    cd terraform
    terraform output
    cd ..
}

# Main deployment flow
main() {
    case "$1" in
        init)
            check_prerequisites
            init_terraform
            ;;
        plan)
            check_prerequisites
            plan_terraform
            ;;
        apply)
            check_prerequisites
            apply_terraform
            ;;
        build)
            check_prerequisites
            build_and_push
            ;;
        upload)
            upload_data
            ;;
        optimize)
            run_optimize
            ;;
        full)
            print_header "Full Deployment"
            check_prerequisites
            init_terraform
            plan_terraform
            apply_terraform
            build_and_push
            upload_data
            show_outputs
            print_success "Deployment complete!"
            ;;
        destroy)
            print_header "Destroying Infrastructure"
            cd terraform
            terraform destroy
            cd ..
            print_success "Infrastructure destroyed"
            ;;
        outputs)
            show_outputs
            ;;
        *)
            echo "AutoML ML Pipeline - AWS Deployment Script"
            echo ""
            echo "Usage: ./deploy.sh [command]"
            echo ""
            echo "Commands:"
            echo "  init      Initialize Terraform"
            echo "  plan      Plan Terraform changes"
            echo "  apply     Apply Terraform configuration"
            echo "  build     Build and push Docker images"
            echo "  upload    Upload data to S3"
            echo "  optimize  Run optimization task"
            echo "  full      Run complete deployment (init + plan + apply + build + upload)"
            echo "  outputs   Show Terraform outputs"
            echo "  destroy   Destroy all infrastructure"
            echo ""
            echo "Examples:"
            echo "  ./deploy.sh full      # Complete deployment"
            echo "  ./deploy.sh build     # Just build and push images"
            echo "  ./deploy.sh optimize  # Run optimization task"
            ;;
    esac
}

main "$@"
