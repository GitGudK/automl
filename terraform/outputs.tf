# Terraform Outputs

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = aws_subnet.private[*].id
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = aws_subnet.public[*].id
}

# ECR Outputs
output "ecr_repository_optimize_url" {
  description = "URL of the optimization service ECR repository"
  value       = aws_ecr_repository.optimize.repository_url
}

output "ecr_repository_train_url" {
  description = "URL of the training service ECR repository"
  value       = aws_ecr_repository.train.repository_url
}

output "ecr_repository_predict_url" {
  description = "URL of the prediction service ECR repository"
  value       = aws_ecr_repository.predict.repository_url
}

output "ecr_repository_pipeline_url" {
  description = "URL of the pipeline ECR repository"
  value       = aws_ecr_repository.pipeline.repository_url
}

# ECS Outputs
output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.main.arn
}

output "ecs_task_definition_optimize_arn" {
  description = "ARN of the optimization task definition"
  value       = aws_ecs_task_definition.optimize.arn
}

output "ecs_task_definition_train_arn" {
  description = "ARN of the training task definition"
  value       = aws_ecs_task_definition.train.arn
}

output "ecs_task_definition_predict_arn" {
  description = "ARN of the prediction task definition"
  value       = aws_ecs_task_definition.predict.arn
}

# S3 Outputs
output "s3_bucket_name" {
  description = "Name of the S3 data bucket"
  value       = aws_s3_bucket.data.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 data bucket"
  value       = aws_s3_bucket.data.arn
}

# IAM Outputs
output "ecs_execution_role_arn" {
  description = "ARN of the ECS execution role"
  value       = aws_iam_role.ecs_execution.arn
}

output "ecs_task_role_arn" {
  description = "ARN of the ECS task role"
  value       = aws_iam_role.ecs_task.arn
}

# CloudWatch Outputs
output "cloudwatch_log_group_optimize" {
  description = "Name of the optimization service log group"
  value       = aws_cloudwatch_log_group.optimize.name
}

output "cloudwatch_log_group_train" {
  description = "Name of the training service log group"
  value       = aws_cloudwatch_log_group.train.name
}

output "cloudwatch_log_group_predict" {
  description = "Name of the prediction service log group"
  value       = aws_cloudwatch_log_group.predict.name
}

output "cloudwatch_dashboard_name" {
  description = "Name of the CloudWatch dashboard"
  value       = aws_cloudwatch_dashboard.main.dashboard_name
}

# Security Group Outputs
output "ecs_tasks_security_group_id" {
  description = "ID of the ECS tasks security group"
  value       = aws_security_group.ecs_tasks.id
}

# Quick Start Commands
output "docker_login_command" {
  description = "AWS CLI command to authenticate Docker with ECR"
  value       = "aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
}

output "quick_start_commands" {
  description = "Quick start commands for deploying services"
  value = {
    push_optimize = "docker tag automl-optimize:latest ${aws_ecr_repository.optimize.repository_url}:latest && docker push ${aws_ecr_repository.optimize.repository_url}:latest"
    push_train    = "docker tag automl-train:latest ${aws_ecr_repository.train.repository_url}:latest && docker push ${aws_ecr_repository.train.repository_url}:latest"
    push_predict  = "docker tag automl-predict:latest ${aws_ecr_repository.predict.repository_url}:latest && docker push ${aws_ecr_repository.predict.repository_url}:latest"
    run_optimize  = "aws ecs run-task --cluster ${aws_ecs_cluster.main.name} --task-definition ${aws_ecs_task_definition.optimize.family} --launch-type FARGATE --network-configuration 'awsvpcConfiguration={subnets=[${join(",", aws_subnet.private[*].id)}],securityGroups=[${aws_security_group.ecs_tasks.id}],assignPublicIp=DISABLED}'"
    run_train     = "aws ecs run-task --cluster ${aws_ecs_cluster.main.name} --task-definition ${aws_ecs_task_definition.train.family} --launch-type FARGATE --network-configuration 'awsvpcConfiguration={subnets=[${join(",", aws_subnet.private[*].id)}],securityGroups=[${aws_security_group.ecs_tasks.id}],assignPublicIp=DISABLED}'"
    run_predict   = "aws ecs run-task --cluster ${aws_ecs_cluster.main.name} --task-definition ${aws_ecs_task_definition.predict.family} --launch-type FARGATE --network-configuration 'awsvpcConfiguration={subnets=[${join(",", aws_subnet.private[*].id)}],securityGroups=[${aws_security_group.ecs_tasks.id}],assignPublicIp=DISABLED}'"
  }
}
