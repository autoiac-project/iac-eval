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

resource "aws_subnet" "main" {
  vpc_id     = aws_vpc.main.id

  assign_ipv6_address_on_creation = false
  availability_zone = "us-east-1a"
  cidr_block = "10.0.1.0/24"
  # customer_owned_ipv4_pool = 
  enable_dns64 = false
  enable_lni_at_device_index = 1
  enable_resource_name_dns_aaaa_record_on_launch = false
  enable_resource_name_dns_a_record_on_launch = false
  # ipv6_cidr_block = 
  ipv6_native = false

  # outpost_arn = 




  tags = {
    Name = "Main"
  }
} 