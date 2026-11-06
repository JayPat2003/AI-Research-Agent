#!/usr/bin/env bash
# Tear down all Terraform-managed AWS resources for this project.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="$ROOT/infra/aws/terraform"

cd "$TF_DIR"
terraform destroy
echo "Destroyed. S3 bucket contents are deleted with the bucket if empty; check AWS console if destroy fails on a non-empty bucket."
