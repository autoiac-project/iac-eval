resource "aws_s3_bucket" "example" {
  bucket = "my-bucket-1234567890"
  force_destroy = false 
  object_lock_enabled = false

  tags = {
    test-tag = "insert tag"
  }
}
