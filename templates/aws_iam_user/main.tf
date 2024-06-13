resource "aws_iam_user" "example" {
  name = "example_user"
  path = "/system/"
  force_destroy = false

  tags = {
    tag-key = "tag-value"
  }
}