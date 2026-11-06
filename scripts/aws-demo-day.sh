#!/usr/bin/env bash
# Provision a Free Tier EC2 host, deploy the stack, print the public URL.
# Tear down with ./scripts/aws-destroy.sh when finished.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== AWS Free Tier deploy ==="
echo "Creates real AWS resources. Run ./scripts/aws-destroy.sh when finished."
echo

if [[ ! -f .env.aws ]]; then
  cp .env.aws.example .env.aws
  echo "Created .env.aws — set GEMINI_API_KEY, then re-run."
  exit 1
fi

if [[ ! -f infra/aws/terraform/terraform.tfvars ]]; then
  cp infra/aws/terraform/terraform.tfvars.free.example infra/aws/terraform/terraform.tfvars
  echo "Wrote Free Tier terraform.tfvars"
fi

./scripts/aws-provision.sh
AWS_FREE=1 ./scripts/aws-deploy.sh

PUBLIC_IP="$(cd infra/aws/terraform && terraform output -raw public_ip)"

cat <<EOF

App URL:  http://${PUBLIC_IP}
Health:   curl http://${PUBLIC_IP}/health

Tear down when done:
  ./scripts/aws-destroy.sh
EOF
