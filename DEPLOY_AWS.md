# Deploy AI Research Agent on AWS

## Free way to showcase AWS skills (recommended)

You do **not** need a $50–90/mo box left running. Interviewers care that you can **design, provision, deploy, and tear down** with IaC — not that a URL stays up forever.

### Strategy A — Demo day, then destroy (≈ $0–3)

Works on most **new AWS accounts** still inside the 12-month Free Tier (`t3.micro` + ~30GB EBS often covered).

```bash
./scripts/aws-bootstrap-tools.sh
aws configure

cp .env.aws.example .env.aws          # set GEMINI_API_KEY
./scripts/aws-demo-day.sh             # provision t3.micro + deploy free profile
# …record Loom: UI → /health → Terraform files → AWS console…
./scripts/aws-destroy.sh              # mandatory — stops almost all spend
```

Free profile uses:

| Piece | Choice | Why |
| --- | --- | --- |
| EC2 | `t3.micro` + 2GB swap | Free Tier eligible |
| Embeddings | `hash` (not BGE) | Fits in 1GB RAM; no model download |
| LLM | Gemini free API key | No Bedrock/GPU bill |
| Compose | `docker-compose.aws-free.yml` | Memory caps |

Quality is demo-grade (hash RAG). That is fine for proving **AWS deploy skill**.

### Strategy B — Zero ongoing cloud spend (still shows AWS)

Keep these in the repo / portfolio; no instance required:

1. **Terraform** under `infra/aws/terraform/` (VPC, EC2, EIP, SG, S3, IAM)
2. **GitHub Actions** [`.github/workflows/aws-iac.yml`](.github/workflows/aws-iac.yml) — `terraform fmt/validate` + compose config on every PR
3. Architecture diagram + Loom of a past demo-day deploy/destroy
4. README section: “How I deploy to AWS”

This alone is enough to claim AWS deployment experience honestly.

### Strategy C — Stop instead of destroy (small residual cost)

```bash
aws ec2 stop-instances --instance-ids "$(cd infra/aws/terraform && terraform output -raw instance_id)"
```

Stops compute. You still pay a little for EBS (~few USD/mo if outside Free Tier) and the Elastic IP if the instance is stopped (AWS charges unused EIPs). Prefer **destroy** when money is tight.

---

## What gets deployed (when you do run it)

```
Browser → Elastic IP :80 (Caddy)
            ├─ /query, /ingest, /health, … → FastAPI
            └─ /*                            → Next.js
         EC2 also runs: Postgres, Redis, worker
         S3 bucket created for future uploads
         Gemini cloud LLM (Ollama disabled on AWS)
```

| Profile | Command | Instance | Cost if left on |
| --- | --- | --- | --- |
| **Free (default)** | `./scripts/aws-demo-day.sh` or `AWS_FREE=1 ./scripts/aws-deploy.sh` | `t3.micro` | ~$0 on Free Tier; else ~$7–10/mo |
| Paid / full BGE | copy `terraform.tfvars.paid.example` + `AWS_FREE=0 ./scripts/aws-deploy.sh` | `t3.large` | ~$50–90/mo |

---

## Prerequisites

1. AWS account + IAM user that can manage EC2/VPC/IAM/S3/EIP  
2. [Gemini API key](https://aistudio.google.com/apikey)  
3. On your Mac:

```bash
./scripts/aws-bootstrap-tools.sh
aws configure   # Access Key, Secret, region (e.g. us-east-1)
```

Check Free Tier status in the AWS Billing console → **Free Tier**. If your account is older than 12 months, `t3.micro` is cheap (~$7–8/mo) but not free — still destroy after demos.

---

## Manual provision / deploy

```bash
cp .env.aws.example .env.aws
cp infra/aws/terraform/terraform.tfvars.free.example infra/aws/terraform/terraform.tfvars

./scripts/aws-provision.sh
AWS_FREE=1 ./scripts/aws-deploy.sh
./scripts/aws-destroy.sh    # when done
```

---

## HTTPS (optional)

1. DNS **A record** → Elastic IP  
2. `domain_name = "research.example.com"` in `terraform.tfvars`, re-apply  
3. Point Caddyfile at that hostname for Let’s Encrypt, redeploy  

---

## Phase 2 (later, when you have budget or an employer cloud)

ECS Fargate + ALB, RDS (pgvector), ElastiCache, SQS ingest, S3-backed uploads in app code, Cognito. Not required to showcase fundamentals.

---

## Security notes

- Prefer `ssh_cidr = "YOUR.IP/32"`  
- Never commit `.env.aws`, `*.pem`, or `*.tfstate`  
- Always run `./scripts/aws-destroy.sh` after a portfolio recording session  
