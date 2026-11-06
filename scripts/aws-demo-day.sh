#!/usr/bin/env bash
# Portfolio workflow: provision Free Tier box → deploy → print checklist → YOU destroy after recording.
# Goal: prove AWS skills without leaving a $50–90/mo bill running.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== AWS demo day (Free Tier profile) ==="
echo "This will create real AWS resources. Destroy them when you finish filming."
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

============================================================
SHOWCASE CHECKLIST (record a 2–4 min Loom / screen capture)
============================================================
1. Browser:  http://${PUBLIC_IP}
2. Health:   curl http://${PUBLIC_IP}/health
3. Run one research question in the UI
4. Show this repo: Terraform + docker-compose.aws*.yml + deploy scripts
5. Show AWS console: EC2 instance, VPC, S3 uploads bucket, Elastic IP

THEN STOP THE BILL (do not skip):
  ./scripts/aws-destroy.sh

Typical cost if destroyed same day: well under \$1–3 on Free Tier-eligible accounts
(mostly nothing if still inside the 12-month Free Tier for EC2/EBS).
============================================================
EOF
