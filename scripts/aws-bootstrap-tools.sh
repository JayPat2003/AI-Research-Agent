#!/usr/bin/env bash
# Install AWS CLI v2 + Terraform on macOS (Apple Silicon or Intel).
set -euo pipefail

echo "==> Checking Homebrew"
if ! command -v brew >/dev/null 2>&1; then
  echo "Install Homebrew first: https://brew.sh"
  exit 1
fi

echo "==> Installing AWS CLI + Terraform"
brew install awscli terraform

echo "==> Versions"
aws --version
terraform version

echo
echo "Next: configure credentials"
echo "  aws configure"
echo "  # or: export AWS_PROFILE=your-profile"
echo
echo "Then:"
echo "  cp .env.aws.example .env.aws   # set GEMINI_API_KEY"
echo "  cp infra/aws/terraform/terraform.tfvars.example infra/aws/terraform/terraform.tfvars"
echo "  ./scripts/aws-provision.sh"
echo "  ./scripts/aws-deploy.sh"
