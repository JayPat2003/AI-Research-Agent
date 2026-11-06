#!/usr/bin/env bash
# Sync this repo to the EC2 host and bring the GenAI stack up with Docker Compose + Caddy.
# Default = free-tier profile (hash embeddings). Full BGE: AWS_FREE=0 ./scripts/aws-deploy.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="$ROOT/infra/aws/terraform"
APP_DIR="/opt/ai-research-agent"
REMOTE_USER="ubuntu"
AWS_FREE="${AWS_FREE:-1}"

COMPOSE_FILES=(-f docker-compose.yml -f docker-compose.aws.yml)
if [[ "$AWS_FREE" == "1" ]]; then
  COMPOSE_FILES+=(-f docker-compose.aws-free.yml)
fi
COMPOSE_ARGS="${COMPOSE_FILES[*]}"

if [[ ! -f "$ROOT/.env.aws" ]]; then
  echo "Missing .env.aws — copy from .env.aws.example and set GEMINI_API_KEY"
  exit 1
fi

if ! grep -qE '^GEMINI_API_KEY=.+' "$ROOT/.env.aws"; then
  echo "GEMINI_API_KEY is empty in .env.aws"
  exit 1
fi

cd "$TF_DIR"
PUBLIC_IP="$(terraform output -raw public_ip 2>/dev/null || true)"
KEY_NAME="$(terraform output -raw key_name 2>/dev/null || true)"
PROJECT="$(grep -E '^\s*project_name\s*=' terraform.tfvars 2>/dev/null | sed -E 's/.*=\s*"?([^"]+)"?.*/\1/' || echo ai-research-agent)"
KEY_FILE="$TF_DIR/${PROJECT}-key.pem"

if [[ -z "$PUBLIC_IP" ]]; then
  echo "No Terraform public_ip output. Run ./scripts/aws-provision.sh first."
  exit 1
fi

if [[ ! -f "$KEY_FILE" ]]; then
  # Allow override: AWS_SSH_KEY=/path/to/key.pem ./scripts/aws-deploy.sh
  KEY_FILE="${AWS_SSH_KEY:-}"
fi
if [[ -z "$KEY_FILE" || ! -f "$KEY_FILE" ]]; then
  echo "SSH private key not found. Expected $TF_DIR/${PROJECT}-key.pem or set AWS_SSH_KEY."
  exit 1
fi

SSH=(ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
SCP=(scp -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new)
RSYNC_SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=accept-new"

echo "==> Waiting for SSH on $PUBLIC_IP"
for _ in $(seq 1 30); do
  if "${SSH[@]}" "${REMOTE_USER}@${PUBLIC_IP}" "test -f ${APP_DIR}/.bootstrap-ready" 2>/dev/null; then
    break
  fi
  sleep 10
done

if ! "${SSH[@]}" "${REMOTE_USER}@${PUBLIC_IP}" "test -f ${APP_DIR}/.bootstrap-ready"; then
  echo "Instance is up but Docker bootstrap marker missing. Check: ${SSH[*]} ${REMOTE_USER}@${PUBLIC_IP}"
  exit 1
fi

echo "==> Syncing project (excluding local junk)"
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.env' \
  --exclude '.env.aws' \
  --exclude 'node_modules/' \
  --exclude 'frontend/.next/' \
  --exclude 'models/' \
  --exclude 'data/uploads/' \
  --exclude 'data/chroma/' \
  --exclude 'infra/aws/terraform/.terraform/' \
  --exclude 'infra/aws/terraform/*.tfstate*' \
  --exclude 'infra/aws/terraform/*.pem' \
  --exclude '__pycache__/' \
  --exclude '.DS_Store' \
  -e "$RSYNC_SSH" \
  "$ROOT/" "${REMOTE_USER}@${PUBLIC_IP}:${APP_DIR}/"

echo "==> Uploading secrets (.env.aws -> .env on host)"
"${SCP[@]}" "$ROOT/.env.aws" "${REMOTE_USER}@${PUBLIC_IP}:${APP_DIR}/.env"

# Point CORS at the public URL (same host the browser uses)
APP_URL="http://${PUBLIC_IP}"
"${SSH[@]}" "${REMOTE_USER}@${PUBLIC_IP}" "grep -q '^CORS_ORIGINS=' ${APP_DIR}/.env && sed -i 's|^CORS_ORIGINS=.*|CORS_ORIGINS=${APP_URL},http://localhost:3000|' ${APP_DIR}/.env || echo 'CORS_ORIGINS=${APP_URL}' >> ${APP_DIR}/.env"

if [[ "$AWS_FREE" == "1" ]]; then
  echo "==> Free-tier profile: hash embeddings (no BGE download). Expect a slow first boot on t3.micro."
  "${SSH[@]}" "${REMOTE_USER}@${PUBLIC_IP}" \
    "grep -q '^EMBEDDING_BACKEND=' ${APP_DIR}/.env && sed -i 's|^EMBEDDING_BACKEND=.*|EMBEDDING_BACKEND=hash|' ${APP_DIR}/.env || echo 'EMBEDDING_BACKEND=hash' >> ${APP_DIR}/.env"
else
  echo "==> Paid profile: BGE models will download on first start (needs t3.large+)."
fi

echo "==> Building and starting containers"
"${SSH[@]}" "${REMOTE_USER}@${PUBLIC_IP}" "cd ${APP_DIR} && docker compose ${COMPOSE_ARGS} up -d --build"

echo
if [[ "$AWS_FREE" == "1" ]]; then
  echo "Deployed. Profile: FREE (t3.micro + hash embeddings)"
else
  echo "Deployed. Profile: FULL (BGE)"
fi
echo "  UI / API (via Caddy): ${APP_URL}"
echo "  Health:               ${APP_URL}/health"
echo "  SSH:                  ssh -i ${KEY_FILE} ${REMOTE_USER}@${PUBLIC_IP}"
echo
echo "When the demo is done, stop billing:"
echo "  ./scripts/aws-destroy.sh"
echo
echo "Tail logs:"
echo "  ssh -i ${KEY_FILE} ${REMOTE_USER}@${PUBLIC_IP} 'cd ${APP_DIR} && docker compose ${COMPOSE_ARGS} logs -f api worker web caddy'"
