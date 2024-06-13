resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  instance_tenancy = "default"
  enable_dns_support   = true
  enable_network_address_usage_metrics = false
  enable_dns_hostnames = true
  tags = {
    test-tag = "{insert tag}"
  }
}