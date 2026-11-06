output "public_ip" {
  description = "Elastic IP of the app server"
  value       = aws_eip.app.public_ip
}

output "ssh_command" {
  description = "SSH into the instance"
  value       = "ssh -i infra/aws/terraform/${var.project_name}-key.pem ubuntu@${aws_eip.app.public_ip}"
}

output "app_url" {
  description = "HTTP URL (use HTTPS once domain_name + DNS are set)"
  value       = var.domain_name != "" ? "https://${var.domain_name}" : "http://${aws_eip.app.public_ip}"
}

output "uploads_bucket" {
  description = "S3 bucket for future upload storage"
  value       = aws_s3_bucket.uploads.bucket
}

output "instance_id" {
  value = aws_instance.app.id
}

output "key_name" {
  value = local.key_name
}
