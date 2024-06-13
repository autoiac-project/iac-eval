resource "aws_neptune_cluster" "default" {
  allow_major_version_upgrade = false
  apply_immediately           = true
  availability_zones = [ "us-east-1a" ]
  backup_retention_period = 2
  cluster_identifier                  = "neptune-cluster-demo"
  # cluster_identifier_prefix = "cluster-prefix" # CONFLICTS WITH `cluster_identifier`
  copy_tags_to_snapshot = false
  enable_cloudwatch_logs_exports = [ "audit", "slowquery" ]
  engine                              = "neptune"
  engine_version = "1.2.1.0"
  final_snapshot_identifier = "cluster-final-snapshot"
  # global_cluster_identifer = aws_neptune_global_cluster.<resource name>.id
  # iam_roles = [ aws_iam_role.<resource name>.arn ]
  iam_database_authentication_enabled = false
  # kms_key_arn = aws_kms_key.<resource name>.arn # REQUIRES `storage_encrypted = true`
  neptune_subnet_group_name = aws_neptune_subnet_group.example_subnet_group.id
  neptune_cluster_parameter_group_name = aws_neptune_cluster_parameter_group.example_cluster_parameter_group.id
  # neptune_instance_parameter_group_name = aws_neptune_instance_parameter_group.<resource name>.id
  preferred_backup_window             = "07:00-09:00"
  port = 8182
  # replication_source_identifier = aws_neptune_cluster.<resource name>.arn
  skip_final_snapshot                 = true
  # snapshot_identifier = aws_neptune_cluster_snapshot.<resource name>.arn
  storage_encrypted = false
  tags = {
    test-tag = "insert tag"
  }
  # vpc_security_group_ids = [ aws_security_group.<resource name>.id ]
  deletion_protection = false
  # serverless_v2_scaling_configuration { }
}

resource "aws_neptune_cluster_parameter_group" "example_cluster_parameter_group" {
  family      = "neptune1.2" # neptune1 is also valid but will not work with engines version 1.2.0.0 and higher
  name        = "example"
  # name_prefix = "example-prefix" # CONFLICTS WTIH `name`
  description = "terraform neptune cluster parameter group"
  parameter {
    name  = "neptune_enable_audit_log"
    value = 1
  }
  tags = {
    Name = "My neptune cluster parameter group"
  }
}

resource "aws_neptune_subnet_group" "example_subnet_group" {
  name       = "example"
  # name_prefix = "example-prefix" # CONFLICTS WTIH `name`
  subnet_ids = [aws_subnet.subnet1.id, aws_subnet.subnet2.id]
  description = "terraform neptune subnet group"
  tags = {
    Name = "My neptune subnet group"
  }
}

resource "aws_vpc" "example_vpc" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_subnet" "subnet1" {
  vpc_id     = aws_vpc.example_vpc.id
  cidr_block = "10.0.1.0/24"
}

resource "aws_subnet" "subnet2" {
  vpc_id     = aws_vpc.example_vpc.id
  cidr_block = "10.0.2.0/24"
}