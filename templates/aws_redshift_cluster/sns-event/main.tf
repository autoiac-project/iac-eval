resource "aws_redshift_cluster" "example" {
  cluster_identifier  = "tf-redshift-cluster"
  database_name       = "mydb"
  master_username     = "exampleuser"
  master_password     = "Mustbe8characters"
  node_type           = "ra3.xlplus"
  cluster_type        = "single-node"
  skip_final_snapshot = true
  cluster_parameter_group_name = aws_redshift_parameter_group.bar.id
}

resource "aws_redshift_parameter_group" "bar" {
  name   = "parameter-group-test-terraform"
  family = "redshift-1.0"

  parameter {
    name  = "require_ssl"
    value = "true"
  }

  parameter {
    name  = "query_group"
    value = "example"
  }

  parameter {
    name  = "enable_user_activity_logging"
    value = "true"
  }
}

resource "aws_sns_topic" "default" {
  name = "redshift-events"
}

resource "aws_redshift_event_subscription" "default" {
  name          = "redshift-event-sub"
  sns_topic_arn = aws_sns_topic.default.arn

  source_type = "cluster-parameter-group"
  source_ids  = [aws_redshift_parameter_group.bar.id]

  severity = "INFO"

  event_categories = [
    "configuration",
    "management",
    "monitoring",
    "security",
  ]

  tags = {
    Name = "default"
  }
}