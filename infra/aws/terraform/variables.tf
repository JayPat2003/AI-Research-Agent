variable "aws_region" {
  description = "AWS region for the GenAI stack"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name prefix for resources"
  type        = string
  default     = "ai-research-agent"
}

variable "instance_type" {
  description = "EC2 size. Use t3.micro for Free Tier portfolio demos (with aws-free compose). t3.large for full BGE stack."
  type        = string
  default     = "t3.micro"
}

variable "root_volume_gb" {
  description = "Root EBS size. Free Tier often covers 30GB for 12 months on eligible accounts."
  type        = number
  default     = 30
}

variable "enable_swap_gb" {
  description = "Create a swapfile on boot (needed for t3.micro; set 0 on large instances)."
  type        = number
  default     = 2
}

variable "ssh_cidr" {
  description = "CIDR allowed to SSH (your public IP /32 recommended)"
  type        = string
  default     = "0.0.0.0/0"
}

variable "app_cidr" {
  description = "CIDR allowed to hit HTTP/HTTPS"
  type        = string
  default     = "0.0.0.0/0"
}

variable "key_name" {
  description = "Existing EC2 key pair name for SSH. Leave empty to create one."
  type        = string
  default     = ""
}

variable "create_key_pair" {
  description = "Create a new key pair and write the private key locally"
  type        = bool
  default     = true
}

variable "domain_name" {
  description = "Optional public hostname for Caddy TLS (leave empty for HTTP via Elastic IP)"
  type        = string
  default     = ""
}
