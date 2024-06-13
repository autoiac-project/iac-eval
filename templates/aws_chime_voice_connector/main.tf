resource "aws_chime_voice_connector" "test" {
  name               = "connector-test-1"
  require_encryption = true
  aws_region         = "us-east-1"
  tags = {
    test-tag = "{insert tag}"
  }
}
