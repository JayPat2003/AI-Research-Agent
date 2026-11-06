#!/usr/bin/env bash
# Create VPC + EC2 + Elastic IP + S3 via Terraform.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="$ROOT/infra/aws/terraform"

if ! command -v aws >/dev/null 2>&1; then
  echo "AWS CLI not found. Run: ./scripts/aws-bootstrap-tools.sh"
  exit 1
fi
if ! command -v terraform >/dev/null 2>&1; then
  echo "Terraform not found. Run: ./scripts/aws-bootstrap-tools.sh"
  exit 1
fi

if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "AWS credentials not configured. Run: aws configure"
  exit 1
fi

cd "$TF_DIR"

if [[ ! -f terraform.tfvars ]]; then
  cp terraform.tfvars.example terraform.tfvars
  echo "Created terraform.tfvars — edit region / ssh_cidr / instance_type if needed, then re-run."
  exit 0
fi

terraform init
terraform apply

echo
echo "Provision complete."
terraform output
echo
echo "Wait ~2 minutes for Docker install on the instance, then:"
echo "  cp .env.aws.example .env.aws   # if you have not already"
echo "  ./scripts/aws-deploy.sh"
