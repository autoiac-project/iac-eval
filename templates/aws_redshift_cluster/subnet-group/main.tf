resource "aws_redshift_cluster" "example" {
  cluster_identifier = "tf-redshift-cluster"
  database_name      = "mydb"
  master_username    = "exampleuser"
  master_password    = "Mustbe8characters"
  node_type          = "ra3.xlplus"
  cluster_type       = "single-node"
  skip_final_snapshot = true
}

resource "aws_redshift_endpoint_access" "example" {
  endpoint_name      = "example"
  subnet_group_name  = aws_redshift_subnet_group.foobar.id
  cluster_identifier = aws_redshift_cluster.example.cluster_identifier
}

resource "aws_vpc" "foo" {
  cidr_block = "10.1.0.0/16"
}

resource "aws_subnet" "foo" {
  cidr_block        = "10.1.1.0/24"
  availability_zone = "us-east-1a"
  vpc_id            = aws_vpc.foo.id

  tags = {
    Name = "tf-dbsubnet-test-1"
  }
}

resource "aws_subnet" "bar" {
  cidr_block        = "10.1.2.0/24"
  availability_zone = "us-east-1b"
  vpc_id            = aws_vpc.foo.id

  tags = {
    Name = "tf-dbsubnet-test-2"
  }
}

resource "aws_redshift_subnet_group" "foobar" {
  name       = "foo"
  subnet_ids = [aws_subnet.foo.id, aws_subnet.bar.id]

  tags = {
    environment = "Production"
  }
}