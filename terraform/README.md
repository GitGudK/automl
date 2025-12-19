# AutoML ML Pipeline - AWS Terraform Deployment

Infrastructure as Code for deploying the AutoML ML pipeline microservices to AWS using ECS Fargate.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         AWS Cloud                           │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                    VPC (10.0.0.0/16)                  │ │
│  │                                                       │ │
│  │  ┌─────────────┐         ┌─────────────┐            │ │
│  │  │   Public    │         │   Public    │            │ │
│  │  │  Subnet 1   │         │  Subnet 2   │            │ │
│  │  │             │         │             │            │ │
│  │  │  NAT GW 1   │         │  NAT GW 2   │            │ │
│  │  └─────────────┘         └─────────────┘            │ │
│  │         │                        │                   │ │
│  │  ┌─────────────┐         ┌─────────────┐            │ │
│  │  │   Private   │         │   Private   │            │ │
│  │  │  Subnet 1   │         │  Subnet 2   │            │ │
│  │  │             │         │             │            │ │
│  │  │ ECS Tasks   │         │ ECS Tasks   │            │ │
│  │  └─────────────┘         └─────────────┘            │ │
│  │                                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐        │
│  │    ECR     │   │    ECS     │   │     S3     │        │
│  │ Repositories│   │  Cluster   │   │   Bucket   │        │
│  └────────────┘   └────────────┘   └────────────┘        │
│                                                             │
│  ┌────────────┐   ┌────────────┐                          │
│  │ CloudWatch │   │    IAM     │                          │
│  │    Logs    │   │   Roles    │                          │
│  └────────────┘   └────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Resources Created

### Networking
- **VPC**: Isolated network with DNS support
- **Subnets**: 2 public + 2 private across availability zones
- **NAT Gateways**: For private subnet internet access
- **Internet Gateway**: For public subnet connectivity
- **Route Tables**: Public and private routing
- **VPC Endpoints**: S3 endpoint for cost optimization

### Container Infrastructure
- **ECR Repositories**: 4 repositories (optimize, train, predict, pipeline)
- **ECS Cluster**: Fargate cluster with Container Insights
- **Task Definitions**: 3 task definitions with configurable CPU/memory
- **Security Groups**: Network security for ECS tasks

### Storage
- **S3 Bucket**: Versioned, encrypted bucket for data and models
  - Lifecycle policies for cost optimization
  - Automatic archival of old predictions

### IAM
- **ECS Execution Role**: For pulling images and logging
- **ECS Task Role**: For S3 access and CloudWatch
- **Step Functions Role**: (Optional) For pipeline orchestration

### Monitoring
- **CloudWatch Log Groups**: Separate logs for each service
- **CloudWatch Alarms**: CPU and memory monitoring
- **CloudWatch Dashboard**: Centralized monitoring view

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
   ```bash
   aws configure
   ```
3. **Terraform** >= 1.0
   ```bash
   brew install terraform  # macOS
   # or download from https://terraform.io
   ```
4. **Docker** for building images
5. **jq** (optional, for deployment script)

## Quick Start

### Option 1: Automated Deployment

```bash
# Run complete deployment
cd terraform
./deploy.sh full
```

This will:
1. Initialize Terraform
2. Create AWS infrastructure
3. Build Docker images
4. Push images to ECR
5. Upload data to S3

### Option 2: Step-by-Step Deployment

```bash
cd terraform

# 1. Initialize Terraform
terraform init

# 2. Review planned changes
terraform plan

# 3. Apply infrastructure
terraform apply

# 4. Build and push Docker images
./deploy.sh build

# 5. Upload data to S3
./deploy.sh upload
```

## Configuration

### Variables

Customize deployment by creating `terraform.tfvars`:

```hcl
# terraform.tfvars
project_name = "automl"
environment  = "prod"
aws_region   = "us-west-2"

# VPC Configuration
vpc_cidr = "10.0.0.0/16"
az_count = 2

# ECS Task Resources
optimize_cpu    = 4096   # 4 vCPUs
optimize_memory = 8192   # 8 GB

train_cpu    = 2048   # 2 vCPUs
train_memory = 4096   # 4 GB

predict_cpu    = 1024   # 1 vCPU
predict_memory = 2048   # 2 GB

# Services
enable_prediction_service = true
prediction_service_count  = 2

# Logging
log_retention_days = 30
```

### Environment-Specific Deployments

```bash
# Development
terraform apply -var="environment=dev" -var="optimize_cpu=2048"

# Production
terraform apply -var="environment=prod" -var="optimize_cpu=4096"
```

## Running Tasks

### Using Deployment Script

```bash
# Run optimization task
./deploy.sh optimize

# View outputs
./deploy.sh outputs
```

### Using AWS CLI

After deployment, Terraform outputs provide quick-start commands:

```bash
# Get outputs
terraform output quick_start_commands

# Run optimization task
aws ecs run-task \
  --cluster automl-cluster \
  --task-definition automl-optimize \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=DISABLED}"

# Run training task
aws ecs run-task \
  --cluster automl-cluster \
  --task-definition automl-train \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=DISABLED}"
```

### Monitoring Task Execution

```bash
# List running tasks
aws ecs list-tasks --cluster automl-cluster

# Describe task
aws ecs describe-tasks --cluster automl-cluster --tasks <task-arn>

# View logs (real-time)
aws logs tail /ecs/automl-optimize --follow

# View logs (last 10 minutes)
aws logs tail /ecs/automl-train --since 10m
```

## Data Management

### Uploading Data to S3

```bash
# Get bucket name
BUCKET=$(terraform output -raw s3_bucket_name)

# Upload training data
aws s3 cp ../data/data_science_project_data.csv s3://$BUCKET/data/

# Upload entire data directory
aws s3 sync ../data/ s3://$BUCKET/data/ --exclude "*.pkl"
```

### Downloading Results

```bash
# Download model artifacts
aws s3 sync s3://$BUCKET/models/ ../data/models/

# Download predictions
aws s3 sync s3://$BUCKET/predictions/ ../data/predictions/
```

## Cost Optimization

### Estimated Monthly Costs

**Development Environment:**
- VPC + NAT Gateways: ~$65/month
- ECS Fargate (on-demand): $0.04048/vCPU-hour + $0.004445/GB-hour
- ECR Storage: ~$1/month (10 GB)
- S3 Storage: ~$2/month (100 GB)
- CloudWatch Logs: ~$5/month (10 GB)
- **Estimated Total**: ~$75-100/month (assuming 10 hours/month task runtime)

### Cost Reduction Strategies

1. **Use Fargate Spot** (60-70% cost savings):
   ```hcl
   capacity_provider_strategy {
     capacity_provider = "FARGATE_SPOT"
     weight           = 100
   }
   ```

2. **Stop NAT Gateways when not needed**:
   ```bash
   # Set az_count = 0 to disable NAT Gateways
   terraform apply -var="az_count=0"
   ```

3. **Use S3 Intelligent-Tiering**:
   - Automatically moves data to cheaper storage classes

4. **Reduce log retention**:
   ```hcl
   log_retention_days = 7  # Instead of 30
   ```

## Security

### Best Practices Implemented

✅ **Network Isolation**: Tasks run in private subnets
✅ **Encryption**: S3 server-side encryption enabled
✅ **IAM**: Least privilege roles
✅ **VPC Endpoints**: No internet for S3 access
✅ **Security Groups**: Restrictive ingress rules
✅ **Image Scanning**: ECR scans on push

### Additional Hardening

1. **Enable VPC Flow Logs**:
   ```hcl
   resource "aws_flow_log" "main" {
     vpc_id          = aws_vpc.main.id
     traffic_type    = "ALL"
     iam_role_arn    = aws_iam_role.flow_logs.arn
     log_destination = aws_cloudwatch_log_group.flow_logs.arn
   }
   ```

2. **Enable AWS Config**:
   - Monitor configuration compliance

3. **Use Secrets Manager** for sensitive data:
   ```hcl
   resource "aws_secretsmanager_secret" "api_key" {
     name = "automl/api-key"
   }
   ```

## Troubleshooting

### Common Issues

**1. Task fails to start**
```bash
# Check task stopped reason
aws ecs describe-tasks --cluster automl-cluster --tasks <task-arn>

# Common causes:
# - Image not found in ECR (push images first)
# - Insufficient resources (increase CPU/memory)
# - IAM permissions (check execution role)
```

**2. Cannot pull image from ECR**
```bash
# Verify image exists
aws ecr describe-images --repository-name automl-optimize

# Verify task execution role has ECR permissions
aws iam get-role-policy --role-name automl-ecs-execution-role --policy-name automl-ecs-execution-ecr-policy
```

**3. Task cannot access S3**
```bash
# Verify bucket exists
aws s3 ls | grep automl

# Verify task role has S3 permissions
aws iam get-role-policy --role-name automl-ecs-task-role --policy-name automl-ecs-task-s3-policy
```

**4. High NAT Gateway costs**
```bash
# Use VPC Flow Logs to identify traffic
# Consider VPC endpoints for AWS services
```

### Terraform Issues

**State Lock**
```bash
# If state is locked
terraform force-unlock <lock-id>
```

**Drift Detection**
```bash
# Check for manual changes
terraform plan -refresh-only
```

**Clean Start**
```bash
# Remove local state (careful!)
rm -rf .terraform terraform.tfstate*
terraform init
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2

      - name: Terraform Init
        run: |
          cd terraform
          terraform init

      - name: Terraform Apply
        run: |
          cd terraform
          terraform apply -auto-approve

      - name: Build and Push Images
        run: |
          cd terraform
          ./deploy.sh build
```

## Cleanup

### Destroy Infrastructure

```bash
# Using deployment script
./deploy.sh destroy

# Or manually
terraform destroy
```

**Important**: This will delete:
- All ECS tasks and services
- ECR repositories and images
- S3 bucket and all data
- CloudWatch logs
- VPC and networking resources

### Preserve Data

To keep S3 data before destroying:

```bash
# Backup data
aws s3 sync s3://$(terraform output -raw s3_bucket_name)/ ./backup/

# Destroy infrastructure
terraform destroy
```

## Advanced Topics

### Step Functions Orchestration

Enable Step Functions for automated pipeline:

```hcl
enable_stepfunctions = true
```

This creates a state machine that runs:
1. Optimization task
2. Training task (waits for optimization)
3. Prediction task (waits for training)

### Auto-Scaling

Add auto-scaling for prediction service:

```hcl
resource "aws_appautoscaling_target" "predict" {
  max_capacity       = 10
  min_capacity       = 1
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.predict.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}
```

### Blue-Green Deployments

Use CodeDeploy for zero-downtime updates:

```hcl
deployment_controller {
  type = "CODE_DEPLOY"
}
```

## Support

### Documentation
- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Main Docker Documentation](../DOCKER.md)

### Getting Help
1. Check CloudWatch logs for task errors
2. Review Terraform plan before applying
3. Verify AWS service quotas
4. Check IAM permissions

## File Structure

```
terraform/
├── main.tf           # Provider and terraform configuration
├── variables.tf      # Input variables
├── outputs.tf        # Output values
├── vpc.tf           # VPC and networking
├── ecs.tf           # ECS cluster and tasks
├── ecr.tf           # ECR repositories
├── s3.tf            # S3 buckets
├── iam.tf           # IAM roles and policies
├── cloudwatch.tf    # Logging and monitoring
├── deploy.sh        # Deployment automation script
└── README.md        # This file
```

## License

This infrastructure code is part of the AutoML ML Pipeline project.
