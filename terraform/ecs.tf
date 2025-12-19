# ECS Cluster and Task Definitions

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "${var.project_name}-ecs-cluster"
  }
}

# Security Group for ECS Tasks
resource "aws_security_group" "ecs_tasks" {
  name        = "${var.project_name}-ecs-tasks-sg"
  description = "Security group for ECS tasks"
  vpc_id      = aws_vpc.main.id

  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  # Allow prediction service port (for future API)
  ingress {
    protocol    = "tcp"
    from_port   = 8000
    to_port     = 8000
    cidr_blocks = var.allowed_cidr_blocks
    description = "Allow inbound traffic on port 8000 for prediction API"
  }

  tags = {
    Name = "${var.project_name}-ecs-tasks-sg"
  }
}

# Task Definition: Feature Optimization
resource "aws_ecs_task_definition" "optimize" {
  family                   = "${var.project_name}-optimize"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.optimize_cpu
  memory                   = var.optimize_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "optimize"
    image = "${aws_ecr_repository.optimize.repository_url}:latest"

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.optimize.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }

    environment = [
      {
        name  = "PYTHONUNBUFFERED"
        value = "1"
      },
      {
        name  = "S3_BUCKET"
        value = aws_s3_bucket.data.id
      }
    ]

    mountPoints = []
    volumesFrom = []
  }])

  tags = {
    Name = "${var.project_name}-optimize-task"
  }
}

# Task Definition: Model Training
resource "aws_ecs_task_definition" "train" {
  family                   = "${var.project_name}-train"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.train_cpu
  memory                   = var.train_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "train"
    image = "${aws_ecr_repository.train.repository_url}:latest"

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.train.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }

    environment = [
      {
        name  = "PYTHONUNBUFFERED"
        value = "1"
      },
      {
        name  = "S3_BUCKET"
        value = aws_s3_bucket.data.id
      }
    ]

    mountPoints = []
    volumesFrom = []
  }])

  tags = {
    Name = "${var.project_name}-train-task"
  }
}

# Task Definition: Prediction Service
resource "aws_ecs_task_definition" "predict" {
  family                   = "${var.project_name}-predict"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.predict_cpu
  memory                   = var.predict_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "predict"
    image = "${aws_ecr_repository.predict.repository_url}:latest"

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.predict.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }

    environment = [
      {
        name  = "PYTHONUNBUFFERED"
        value = "1"
      },
      {
        name  = "S3_BUCKET"
        value = aws_s3_bucket.data.id
      }
    ]

    mountPoints = []
    volumesFrom = []
  }])

  tags = {
    Name = "${var.project_name}-predict-task"
  }
}

# ECS Service for Prediction (long-running)
resource "aws_ecs_service" "predict" {
  count           = var.enable_prediction_service ? 1 : 0
  name            = "${var.project_name}-predict-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.predict.arn
  desired_count   = var.prediction_service_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  tags = {
    Name = "${var.project_name}-predict-service"
  }
}
