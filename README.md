# AI Research & Decision Intelligence Platform

Ask a complex research question. Multi-agent pipeline searches a **private knowledge base** and **public literature**, optionally queries structured trial data, cross-checks evidence, and returns a **cited decision briefing**.

Agents are domain-agnostic. This repo ships a **public healthcare demo corpus** (GLP-1 vs SGLT2 cardiometabolic outcomes) so you can run without proprietary data. Swap `data/seed/` for another field.

Analyst tool only — **not medical advice**.

---

## Architecture

```
User → Next.js → FastAPI
              → Router (Gemini 3.5 Flash-Lite)
              → Retrieval Agent  (hybrid RAG: pgvector + Postgres FTS + RRF + BGE rerank)
              → Research Agent   (Google Scholar, arXiv, Semantic Scholar)
              → Data Agent       (read-only Text-to-SQL on demo clinical_trials)
              → Report Generator (Gemini 3.5 Flash)
              → Critic Agent     (pass / retrieve-again, max 2 retries)
              → Cited report + conversation memory
```

- **LangChain / LangGraph** — agents, tools, structured output  
- **LlamaIndex** — document load / clean / chunk  
- **LLMs** — Gemini (primary) + optional local Ollama fallback  
- **Embeddings** — local BGE (`fastembed`); `hash` mode for tests / Free Tier AWS  
- **No AWS Bedrock** — AWS is used for hosting (EC2/VPC/S3), not for foundation models  

---

## Repository layout

```
AI-Research-Agent/
├── backend/                 # FastAPI + LangGraph agents + RAG + worker
│   ├── app/
│   │   ├── agents/          # Graph, prompts, schemas, invoke
│   │   ├── api/             # HTTP routes + request schemas
│   │   ├── db/              # SQLAlchemy models, session, SQL seed
│   │   ├── eval/            # Retrieval / generation evaluation
│   │   ├── ingestion/       # LlamaIndex loaders + Redis job queue
│   │   ├── llm/             # Gemini / Ollama clients + embeddings
│   │   ├── memory/          # Redis short-term + prefs helpers
│   │   ├── pipeline/        # Legacy Chroma dense-RAG path (used by cli.py)
│   │   ├── retrieval/       # pgvector + FTS + RRF + rerank
│   │   └── tools/           # Web literature + guarded Text-to-SQL
│   ├── tests/
│   └── Dockerfile
├── frontend/                # Next.js 15 research workspace UI
├── data/
│   ├── seed/                # Demo markdown corpus (auto-ingested)
│   ├── eval/                # Eval questions
│   └── uploads/             # User uploads (gitignored)
├── infra/aws/
│   ├── terraform/           # VPC, EC2, EIP, SG, S3, IAM
│   └── caddy/               # Reverse proxy config
├── scripts/                 # aws-bootstrap / provision / deploy / destroy / demo-day
├── .github/workflows/       # Terraform + compose CI (no cloud spend)
├── docker-compose.yml       # Local / base stack
├── docker-compose.aws.yml   # AWS overlay (Caddy, Gemini-only)
├── docker-compose.aws-free.yml  # Free Tier overlay (hash + mem caps)
├── cli.py                   # Optional Chroma CLI pipeline
├── .env.example             # Local secrets template → copy to .env
└── .env.aws.example         # AWS secrets template → copy to .env.aws
```

| Path | Keep? | Notes |
| --- | --- | --- |
| `docker-compose.yml` | Yes | Local default |
| `docker-compose.aws.yml` | Yes | EC2 + Caddy |
| `docker-compose.aws-free.yml` | Yes | Low-cost `t3.micro` deploy (hash embeddings) |
| `cli.py` + `backend/app/pipeline/` | Yes (optional) | Legacy Chroma path; main product is LangGraph + pgvector API |
| `.env` / `.env.aws` | Never commit | Gitignored |

---

## Requirements

- Linux (x86_64) or macOS (Intel / Apple Silicon)  
- Python 3.12 via Conda ([`environment.yml`](environment.yml)) or the API container  
- Docker + Docker Compose  
- Node.js 22 (local frontend only)  
- Free [Gemini API key](https://aistudio.google.com/apikey)  

On Apple Silicon use a native `arm64` Conda (e.g. Miniforge). `uname -m` should report `arm64`.

---

## Quick start (local)

```bash
git clone https://github.com/JayPat2003/AI-Research-Agent.git
cd AI-Research-Agent
conda env create -f environment.yml
conda activate ai-research-agent
cp .env.example .env          # set GEMINI_API_KEY
docker compose up --build
```

- UI: [http://localhost:3000](http://localhost:3000)  
- API health: [http://localhost:8000/health](http://localhost:8000/health)  

Example question:

> Compare GLP-1 receptor agonists and SGLT2 inhibitors for adults with type 2 diabetes and established cardiovascular disease. What do the major outcomes trials show, which populations were studied, and which class is more suitable as the first add-on in a research briefing for a clinical-operations team?

Follow up with “What about kidney outcomes?” — the router uses conversation summary + recent turns.

If `pip` still hits `~/.local` after `conda activate`: `export PYTHONNOUSERSITE=1`.

### Hybrid: API on Conda, Postgres + Redis in Docker

```bash
docker compose up postgres redis
conda activate ai-research-agent
cd backend && uvicorn app.main:app --reload --port 8000
# other terminals: python -m app.worker · cd frontend && npm install && npm run dev
```

BGE weights download once into `./models`.

---

## Env files

| File | Purpose |
| --- | --- |
| `.env.example` → `.env` | **Local** development |
| `.env.aws.example` → `.env.aws` | **AWS** deploy only (`aws-deploy.sh` / `aws-demo-day.sh`) |

Do not mix them. Local uses `localhost`; AWS uses Docker service hostnames.

---

## Model stack

| Role | Default | Notes |
| --- | --- | --- |
| Research, critic, report | `gemini-3.5-flash` | Long context, citations |
| Router, SQL, summary | `gemini-3.5-flash-lite` | Saves free-tier quota |
| Ollama research fallback | `qwen2.5:7b` | On Gemini 429 / quota |
| Ollama fast fallback | `llama3.2:latest` | Router / SQL / summary |
| Embeddings | `BAAI/bge-small-en-v1.5` | Local; no embedding API cost |
| Rerank | `BAAI/bge-reranker-base` | MiniLM fallback if needed |

```bash
ollama pull qwen2.5:7b
ollama pull llama3.2:latest
```

Fully offline: `LLM_PROVIDER=ollama`. Embeddings stay local BGE (or `hash`).

---

## Deploy on AWS

Default path targets AWS Free Tier (`t3.micro`, hash embeddings, tear down when finished):

```bash
./scripts/aws-bootstrap-tools.sh
aws configure
cp .env.aws.example .env.aws    # set GEMINI_API_KEY
./scripts/aws-demo-day.sh
./scripts/aws-destroy.sh       # tear down when done
```

On the EC2 host the stack is:

```text
docker-compose.yml + docker-compose.aws.yml + docker-compose.aws-free.yml
```

For full BGE quality, use `terraform.tfvars.paid.example` (`t3.large`) and `AWS_FREE=0 ./scripts/aws-deploy.sh`.

CI: [`.github/workflows/aws-iac.yml`](.github/workflows/aws-iac.yml) validates Terraform and compose config.

---

## Evaluation & tests

```bash
cd backend
python -m app.eval.run --variant naive
python -m app.eval.run --variant hybrid
python -m app.eval.run --variant hybrid_rerank
python -m app.eval.run --variant hybrid_rerank --generation   # extra Gemini calls

EMBEDDING_BACKEND=hash python -m pytest -q
```

Metrics: Recall@5, Precision@5, MRR, nDCG@5 (+ optional faithfulness / citation checks).

---

## Ingestion

`POST /ingest/file` · `/ingest/url` · `/ingest/arxiv` · `/ingest/text`

PDF, DOCX, Markdown, HTML, URL, and arXiv ids → LlamaIndex → chunk (~512 / 64) → local embed → Postgres (`vector` + `tsvector`). Redis worker runs ingest off the request thread.

---

## Demo corpus

Type 2 diabetes treatment classes, SGLT2 / GLP-1 outcomes trials, comparison framing, evidence grading, and a `clinical_trials` SQL catalog of well-known CVOTs.

Never commit API keys or proprietary docs. Uploads go in `data/uploads/` (gitignored).
