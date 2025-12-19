# S3 Bucket for Data Storage

# S3 Bucket for ML Data
resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-ml-data-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "${var.project_name}-data-bucket"
    Description = "Storage for ML pipeline data, models, and predictions"
  }
}

# S3 Bucket Versioning
resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id

  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 Bucket Public Access Block
resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Lifecycle Policy
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  # Move old predictions to Glacier after 90 days
  rule {
    id     = "archive-predictions"
    status = "Enabled"

    filter {
      prefix = "predictions/"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }

  # Delete optimization logs after 180 days
  rule {
    id     = "expire-logs"
    status = "Enabled"

    filter {
      prefix = "logs/"
    }

    expiration {
      days = 180
    }
  }
}

# S3 Bucket for Terraform State (optional, commented out)
# Uncomment if you want to use S3 backend for Terraform state
# resource "aws_s3_bucket" "terraform_state" {
#   bucket = "${var.project_name}-terraform-state-${data.aws_caller_identity.current.account_id}"
#
#   tags = {
#     Name        = "${var.project_name}-terraform-state"
#     Description = "Terraform state storage"
#   }
# }
#
# resource "aws_s3_bucket_versioning" "terraform_state" {
#   bucket = aws_s3_bucket.terraform_state.id
#
#   versioning_configuration {
#     status = "Enabled"
#   }
# }
#
# resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
#   bucket = aws_s3_bucket.terraform_state.id
#
#   rule {
#     apply_server_side_encryption_by_default {
#       sse_algorithm = "AES256"
#     }
#   }
# }
