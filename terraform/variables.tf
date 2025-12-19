# Terraform Variables

# General Configuration
variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "automl"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "az_count" {
  description = "Number of availability zones to use"
  type        = number
  default     = 2
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access prediction service"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# ECS Task Configuration - Optimization Service
variable "optimize_cpu" {
  description = "CPU units for optimization task (1024 = 1 vCPU)"
  type        = number
  default     = 4096  # 4 vCPUs
}

variable "optimize_memory" {
  description = "Memory for optimization task in MB"
  type        = number
  default     = 8192  # 8 GB
}

# ECS Task Configuration - Training Service
variable "train_cpu" {
  description = "CPU units for training task (1024 = 1 vCPU)"
  type        = number
  default     = 2048  # 2 vCPUs
}

variable "train_memory" {
  description = "Memory for training task in MB"
  type        = number
  default     = 4096  # 4 GB
}

# ECS Task Configuration - Prediction Service
variable "predict_cpu" {
  description = "CPU units for prediction task (1024 = 1 vCPU)"
  type        = number
  default     = 1024  # 1 vCPU
}

variable "predict_memory" {
  description = "Memory for prediction task in MB"
  type        = number
  default     = 2048  # 2 GB
}

# ECS Service Configuration
variable "enable_prediction_service" {
  description = "Enable long-running prediction service"
  type        = bool
  default     = false
}

variable "prediction_service_count" {
  description = "Desired count for prediction service tasks"
  type        = number
  default     = 1
}

# CloudWatch Configuration
variable "log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 30
}

# Step Functions Configuration
variable "enable_stepfunctions" {
  description = "Enable Step Functions for pipeline orchestration"
  type        = bool
  default     = false
}

# Tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}
