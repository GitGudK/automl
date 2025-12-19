# CloudWatch Logging and Monitoring

# CloudWatch Log Group: Optimization Service
resource "aws_cloudwatch_log_group" "optimize" {
  name              = "/ecs/${var.project_name}-optimize"
  retention_in_days = var.log_retention_days

  tags = {
    Name    = "${var.project_name}-optimize-logs"
    Service = "Optimization"
  }
}

# CloudWatch Log Group: Training Service
resource "aws_cloudwatch_log_group" "train" {
  name              = "/ecs/${var.project_name}-train"
  retention_in_days = var.log_retention_days

  tags = {
    Name    = "${var.project_name}-train-logs"
    Service = "Training"
  }
}

# CloudWatch Log Group: Prediction Service
resource "aws_cloudwatch_log_group" "predict" {
  name              = "/ecs/${var.project_name}-predict"
  retention_in_days = var.log_retention_days

  tags = {
    Name    = "${var.project_name}-predict-logs"
    Service = "Prediction"
  }
}

# CloudWatch Metric Alarm: High CPU for Optimization
resource "aws_cloudwatch_metric_alarm" "optimize_cpu" {
  alarm_name          = "${var.project_name}-optimize-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "This metric monitors optimization service CPU utilization"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = "${var.project_name}-optimize"
  }

  tags = {
    Name = "${var.project_name}-optimize-cpu-alarm"
  }
}

# CloudWatch Metric Alarm: High Memory for Training
resource "aws_cloudwatch_metric_alarm" "train_memory" {
  alarm_name          = "${var.project_name}-train-high-memory"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "MemoryUtilization"
  namespace           = "AWS/ECS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "This metric monitors training service memory utilization"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = "${var.project_name}-train"
  }

  tags = {
    Name = "${var.project_name}-train-memory-alarm"
  }
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-ml-pipeline"

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/ECS", "CPUUtilization", { stat = "Average" }],
            [".", "MemoryUtilization", { stat = "Average" }]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "ECS Task Resource Utilization"
        }
      },
      {
        type = "log"
        properties = {
          query   = "SOURCE '/ecs/${var.project_name}-optimize' | fields @timestamp, @message | sort @timestamp desc | limit 20"
          region  = var.aws_region
          title   = "Recent Optimization Logs"
        }
      },
      {
        type = "log"
        properties = {
          query   = "SOURCE '/ecs/${var.project_name}-train' | fields @timestamp, @message | sort @timestamp desc | limit 20"
          region  = var.aws_region
          title   = "Recent Training Logs"
        }
      }
    ]
  })
}
