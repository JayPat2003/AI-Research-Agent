# AI Research & Decision Intelligence Platform

An agentic, citation-grounded research intelligence platform that answers complex analytical questions by combining **hybrid private knowledge-base retrieval**, **live academic literature search**, **guarded Text-to-SQL**, and a **self-correcting critic loop**.

The architecture is **domain-agnostic**. The repository ships with a public cardiometabolic clinical-trials demo corpus (GLP-1 receptor agonists vs. SGLT2 inhibitors) and a seeded `clinical_trials` SQL catalog so the system can be run and evaluated out of the box. Swap `data/seed/` to target another domain.

> **Disclaimer:** Analyst decision-support tool only — not medical, legal, or financial advice.

---

## Latest Approach & Architectural Evolution

The project has been consolidated onto a **single unified production architecture** powered by **FastAPI + LangGraph + PostgreSQL (`pgvector` + FTS) + Redis**:

- **Removed legacy pipeline:** The earlier standalone ChromaDB CLI pipeline (`cli.py`, `backend/app/pipeline/`) has been removed to eliminate duplicate retrieval, memory, and catalog paths.
- **Unified Hybrid RAG:** Replaced dense-only Chroma search with PostgreSQL `pgvector` (HNSW cosine similarity) + PostgreSQL full-text search (`TSVECTOR` + GIN) combined via **Reciprocal Rank Fusion (RRF)** and **local cross-encoder reranking** (`BAAI/bge-reranker-base`).
- **Off-Topic Guard & Live Literature Escalation:** Retrieved private chunks pass through a rerank score + topical-overlap filter (`_topical_overlap`). When the private KB lacks on-topic evidence, the graph clears irrelevant hits and automatically escalates to live academic search (Google Scholar, arXiv, Semantic Scholar).
- **Critic-Driven Self-Correction:** A dedicated Critic agent verifies claim support, citation validity, internal consistency, and question coverage before finalizing—triggering targeted re-retrieval (`missing_queries`) up to `critic_max_retries = 2`.
- **Two-Tier Conversation Memory:** Short-term turns are cached in Redis (`short_term_turns = 8`), while long-term conversation summaries and user preferences are persisted in PostgreSQL and refreshed asynchronously after each briefing.
- **Zero-Cost Embedding & Resilient LLM Stack:** Embeddings and reranking run locally via `fastembed` (`BAAI/bge-small-en-v1.5` / `bge-reranker-base`), while LLM calls use Gemini (`gemini-3.5-flash` / `gemini-3.5-flash-lite`) with automatic retry/backoff and optional fallback to local Ollama (`qwen2.5:7b` / `llama3.2:latest`).

---

## System Architecture

```text
                              ┌────────────────────────────────────────┐
                              │         Client / Next.js UI (:3000)    │
                              │   SSE Streaming · KB Docs · Uploads    │
                              └───────────────────┬────────────────────┘
                                                  │ HTTP / SSE
                              ┌───────────────────▼────────────────────┐
                              │          FastAPI Backend (:8000)       │
                              │  /query (SSE) · /ingest/* · /documents │
                              └──┬──────────────────────────────────┬──┘
                                 │                                  │
           ┌─────────────────────▼──────────────────┐   ┌───────────▼────────────┐
           │     LangGraph Agent Orchestrator       │   │   Async Ingest Queue   │
           │                                        │   │     (Redis RPUSH)      │
           │  1. Memory   (Redis + Postgres)        │   └───────────┬────────────┘
           │  2. Router   (Intent + Tool Flags)     │               │ BLPOP
           │  3. Retrieve (pgvector + FTS + RRF     │   ┌───────────▼────────────┐
           │               + BGE Rerank + Filter)   │   │   Ingestion Worker     │
           │  4. Research (GScholar, arXiv, S2)     │   │  LlamaIndex Loaders    │
           │  5. Data     (sqlglot-guarded SELECT)  │   │  Chunk (512 / 64)      │
           │  6. Draft    (CitedReport JSON + MD)   │   │  Local BGE Embedder    │
           │  7. Critic   (Verify → Retry or Pass)  │   └───────────┬────────────┘
           │  8. Finalize (Persist + Summarize)     │               │
           └─────────┬─────────────────┬────────────┘               │
                     │                 │                            │
      ┌──────────────▼───┐   ┌─────────▼──────────┐                 │
      │   LLM Layer      │   │  Postgres +        │◄────────────────┘
      │ • Gemini Flash   │   │  pgvector (pg16)   │
      │ • Flash-Lite     │   │ • documents        │
      │ • Ollama Fallback│   │ • chunks (HNSW+GIN)│
      └──────────────────┘   │ • clinical_trials  │
                             │ • conversations    │
                             │ • ingest_jobs      │
                             └────────────────────┘
```

---

## Approach Flow

### 1. Query & Decision-Briefing Flow (`POST /query` via SSE)

1. **Memory Load (`memory` node):** Ensures a `Conversation` and `UserPreference` record exist in PostgreSQL, loads the rolling conversation summary + user preferences from Postgres and recent turns from Redis, and formats them into prompt context.
2. **Intent Routing & Query Rewriting (`router` node):** `gemini-3.5-flash-lite` classifies the query (`knowledge | research | comparison | data | calculation | web`), resolves follow-up references (e.g., *"What about kidney outcomes?"*) into a self-contained `rewritten_query`, and outputs tool flags (`needs_retrieval`, `needs_web`, `needs_sql`).
3. **Hybrid Private KB Retrieval (`retrieve` node):**
   - **Dense search:** Query embedded with `BAAI/bge-small-en-v1.5` (with BGE retrieval prefix) against `pgvector` HNSW cosine index (`dense_k = 20`).
   - **Sparse search:** PostgreSQL `plainto_tsquery` against the `TSVECTOR` GIN index (`sparse_k = 20`).
   - **Fusion & Rerank:** Combined via Reciprocal Rank Fusion (`k = 60`) and reranked with `BAAI/bge-reranker-base` (`rerank_k = 8`).
   - **Topical Filter:** Chunks failing score/overlap thresholds are discarded; if no on-topic private chunks remain (`weak_kb = True`), live literature search is forced (`needs_web = True`).
4. **Live Academic Research (`research` node):** When `needs_web` is active, queries **Google Scholar** (HTML parser + `scholarly` fallback), **arXiv**, and **Semantic Scholar** and normalizes titles, authors, publication dates, snippets, and URLs.
5. **Guarded Text-to-SQL (`data` node):** When `needs_sql` is active, generates a read-only PostgreSQL `SELECT` against `clinical_trials`. `sqlglot` AST validation (`backend/app/tools/sql_guard.py`) blocks mutations, multi-statements, and non-allowlisted tables, and enforces a `LIMIT 100`.
6. **Report Synthesis (`draft` node):** `gemini-3.5-flash` synthesizes a structured `CitedReport` (`title`, `executive_summary`, `body_markdown`, `recommendation`, `limitations`, `citations`) and renders a citation-linked Markdown briefing.
7. **Verification & Self-Correction (`critic` node):** Checks that claims are grounded, citations exist in retrieved/web sources, there are no contradictions/hallucinations, and the user's question is answered.
   - **Pass** → proceeds to `finalize`.
   - **Fail** (up to `critic_max_retries = 2`) → emits targeted `missing_queries` and routes back to `retrieve` + `research` to gather missing evidence before re-drafting.
8. **Finalize (`finalize` node):** Appends user/assistant turns to Redis, persists messages and citations to Postgres, compresses/refreshes `Conversation.summary`, and streams the `final` payload over SSE.

### 2. Asynchronous Ingestion Flow (`POST /ingest/*` → `worker`)

1. API creates an `IngestJob` (`status = queued`) in Postgres and pushes a job payload onto the Redis queue (`ingest_jobs`).
2. `backend/app/worker.py` pops jobs via `BLPOP` and dispatches by `source_type`:
   - `file`: PDF (`PDFReader`), DOCX (`DocxReader`), Markdown / TXT / HTML
   - `url`: Web extraction via `trafilatura`
   - `arxiv`: Paper metadata & abstract via `arxiv`
   - `text`: Direct Markdown/text note
3. LlamaIndex `SentenceSplitter` chunks content (`chunk_size = 512`, `chunk_overlap = 64`).
4. Local `BAAI/bge-small-en-v1.5` computes 384-dim embeddings; Postgres stores `Document` + `Chunk` rows along with `TSVECTOR` triggers/indexes (`HNSW` + `GIN`), and marks the `IngestJob` `done`.
5. On startup, `data/seed/` markdown files and the `clinical_trials` catalog are automatically seeded if missing.

---

## Repository Layout

```text
AI-Research-Agent/
├── backend/
│   ├── app/
│   │   ├── agents/          # LangGraph state machine, prompts, Pydantic schemas, LLM invoke helper
│   │   ├── api/             # FastAPI routes (SSE /query, /ingest/*, /documents, /conversations, /preferences)
│   │   ├── db/              # SQLAlchemy async/sync sessions, ORM models, clinical_trials SQL seeder
│   │   ├── eval/            # Retrieval (Recall@5, Precision@5, MRR, nDCG@5) & generation eval runner
│   │   ├── ingestion/       # LlamaIndex loaders, chunker, and Redis-backed ingestion pipeline
│   │   ├── llm/             # Gemini + Ollama retry/fallback factory & local BGE/hash embedder/reranker
│   │   ├── memory/          # Redis short-term turns + Postgres conversation summary & preferences
│   │   ├── retrieval/       # pgvector dense + FTS sparse + RRF fusion + BGE reranker + topical filter
│   │   ├── tools/           # Google Scholar, arXiv, Semantic Scholar + sqlglot-guarded Text-to-SQL
│   │   ├── config.py        # Pydantic Settings configuration
│   │   ├── main.py          # FastAPI app entrypoint & startup seed lifespan
│   │   └── worker.py        # Redis queue consumer for async document ingestion
│   ├── tests/               # Unit tests (RRF, metrics, SQL guard, chunking, topical overlap, Scholar parser)
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/                # Next.js 15 / React 19 UI (SSE step trace, critic badge, citations, upload)
├── data/
│   ├── seed/                # Default cardiometabolic markdown corpus (auto-ingested)
│   ├── eval/                # Evaluation dataset (questions.jsonl)
│   └── uploads/             # Uploaded files at runtime (gitignored)
├── infra/aws/
│   ├── terraform/           # VPC, subnet, IGW, SG, EC2 + swap user_data, EIP, S3 uploads bucket, IAM
│   └── caddy/               # Reverse proxy routing (/query, /ingest/*, /health -> API; /* -> Next.js)
├── scripts/                 # AWS helper scripts (bootstrap tools, provision, deploy, demo-day, destroy)
├── .github/workflows/       # CI checks (terraform fmt/validate + docker compose config validation)
├── docker-compose.yml       # Local stack (postgres+pgvector, redis, api, worker, web)
├── docker-compose.aws.yml   # AWS overlay (Caddy reverse proxy, Gemini-only, same-origin frontend)
├── docker-compose.aws-free.yml  # Free Tier t3.micro overlay (hash embeddings + container memory limits)
├── environment.yml          # Conda Python 3.12 environment
├── .env.example             # Local environment template -> copy to .env
└── .env.aws.example         # AWS environment template -> copy to .env.aws
```

---

## Model & Embedding Stack

| Component | Default | Fallback / Mode | Notes |
| --- | --- | --- | --- |
| **Research / Draft / Critic LLM** | `gemini-3.5-flash` | `qwen2.5:7b` (Ollama) | Long context, structured citations |
| **Fast / Router / SQL / Summary LLM** | `gemini-3.5-flash-lite` | `llama3.2:latest` (Ollama) | Preserves free-tier RPM quota |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (`fastembed`) | `EMBEDDING_BACKEND=hash` | 384-dim local inference; zero API cost |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-base` | `Xenova/ms-marco-MiniLM-L-6-v2` | Token-overlap scoring in `hash` mode |

- On `429` / rate-limit / transient errors, `FallbackChat` automatically retries with exponential backoff and falls back to local Ollama when `OLLAMA_FALLBACK=true`.
- Set `LLM_PROVIDER=ollama` to run fully offline (pull models first: `ollama pull qwen2.5:7b && ollama pull llama3.2:latest`).
- **No AWS Bedrock:** AWS is used purely for infrastructure hosting (`EC2`, `VPC`, `EIP`, `S3`), not for model inference.

---

## API Endpoints

| Method & Path | Description |
| --- | --- |
| `GET /health` | Liveness check (`{"status": "ok"}`) |
| `POST /query` | Streams LangGraph node events (`memory`, `route`, `retrieve`, `research`, `data`, `draft`, `critic`, `final`) over Server-Sent Events (`text/event-stream`) |
| `POST /ingest/file` | Uploads PDF/DOCX/MD/TXT/HTML and enqueues an ingestion job |
| `POST /ingest/url` | Enqueues web page extraction & indexing |
| `POST /ingest/arxiv` | Enqueues arXiv abstract ingestion by `arxiv_id` |
| `POST /ingest/text` | Enqueues raw Markdown/text ingestion |
| `GET /ingest/{job_id}` | Polls `IngestJob` status (`queued`, `processing`, `done`, `failed`) |
| `GET /documents` | Lists indexed knowledge-base documents |
| `GET /conversations/{conversation_id}` | Returns conversation summary and persisted messages + citations |
| `PUT /preferences/{user_id}` | Updates user domain and citation style preferences |

---

## Quick Start (Local)

### Prerequisites

- Linux (`x86_64`) or macOS (Intel / Apple Silicon — use native `arm64` Conda such as Miniforge)
- Docker & Docker Compose
- Python 3.12 (via [`environment.yml`](environment.yml) or Docker)
- Node.js 22 (only if running the frontend outside Docker)
- Free [Gemini API key](https://aistudio.google.com/apikey)

### Option 1: Full Docker Compose Stack

```bash
git clone https://github.com/JayPat2003/AI-Research-Agent.git
cd AI-Research-Agent
cp .env.example .env          # set GEMINI_API_KEY
docker compose up --build
```

- **Frontend UI:** [http://localhost:3000](http://localhost:3000)
- **FastAPI Docs / Health:** [http://localhost:8000/docs](http://localhost:8000/docs) · [http://localhost:8000/health](http://localhost:8000/health)

> First startup downloads `BAAI/bge-small-en-v1.5` and first retrieval downloads `BAAI/bge-reranker-base` into the `./models` / `hf_models` volume (cached thereafter).

### Option 2: Hybrid Local Dev (Conda API + Worker, Docker Postgres + Redis)

```bash
docker compose up -d postgres redis
conda env create -f environment.yml
conda activate ai-research-agent
cp .env.example .env          # set GEMINI_API_KEY

# Terminal 1: API
cd backend && uvicorn app.main:app --reload --port 8000

# Terminal 2: Async ingestion worker
cd backend && python -m app.worker

# Terminal 3: Frontend (optional)
cd frontend && npm install && npm run dev
```

If `pip` still resolves to `~/.local` inside Conda, run `export PYTHONNOUSERSITE=1`.

### Example Queries

- **Comparison + KB:** *"Compare GLP-1 receptor agonists and SGLT2 inhibitors for adults with type 2 diabetes and established cardiovascular disease. What do the major outcomes trials show, which populations were studied, and which class is more suitable as the first add-on in a research briefing for a clinical-operations team?"*
- **Follow-up (Memory):** *"What about kidney outcomes?"*
- **Catalog SQL:** *"How many completed SGLT2 trials are in the catalog and what were their primary endpoints?"*

---

## Environment Configuration

| Template | Target File | Used For |
| --- | --- | --- |
| `.env.example` | `.env` | Local development (`localhost` Postgres/Redis, `OLLAMA_FALLBACK=true`) |
| `.env.aws.example` | `.env.aws` | AWS EC2 deployment (`postgres`/`redis` Docker service hostnames, `OLLAMA_FALLBACK=false`) |

Never commit `.env`, `.env.aws`, `*.pem`, or `*.tfstate` (all gitignored).

---

## Evaluation & Testing

Run retrieval ablation benchmarks (after seeding the database):

```bash
cd backend
python -m app.eval.run --variant naive
python -m app.eval.run --variant hybrid
python -m app.eval.run --variant hybrid_rerank
python -m app.eval.run --variant hybrid_rerank --generation   # includes LLM faithfulness & citation checks
```

Tracked metrics (`backend/app/eval/metrics.py`): `Recall@5`, `Precision@5`, `MRR`, `nDCG@5`, and `citation_correctness`.

Run core unit tests offline with deterministic hash embeddings:

```bash
cd backend
EMBEDDING_BACKEND=hash python -m pytest tests/test_core.py -q
```

---

## AWS Deployment (Terraform + Docker Compose)

The repository includes an Infrastructure-as-Code path under `infra/aws/` and `scripts/` targeting an **AWS Free Tier** single-EC2 deployment (`t3.micro` + 2 GB swap behind Caddy):

```text
Browser → Elastic IP :80 (Caddy)
            ├─ /health, /query, /documents, /ingest/*, /conversations/*, /preferences/* → FastAPI (:8000)
            └─ /* → Next.js (:3000)
```

```bash
./scripts/aws-bootstrap-tools.sh   # installs AWS CLI + Terraform (macOS)
aws configure
cp .env.aws.example .env.aws       # set GEMINI_API_KEY
./scripts/aws-demo-day.sh          # provisions VPC/EC2/EIP/S3 + deploys compose free profile
./scripts/aws-destroy.sh           # tears down all Terraform resources to stop billing
```

- Compose stack on EC2: `docker-compose.yml` + `docker-compose.aws.yml` + `docker-compose.aws-free.yml` (`EMBEDDING_BACKEND=hash` + memory caps for 1 GB RAM).
- For full BGE embeddings/reranking on EC2, set `instance_type = "t3.large"` and `enable_swap_gb = 0` in `infra/aws/terraform/terraform.tfvars` and deploy with `AWS_FREE=0 ./scripts/aws-deploy.sh`.
- CI ([`.github/workflows/aws-iac.yml`](.github/workflows/aws-iac.yml)) runs `terraform fmt -check`, `terraform validate`, and `docker compose config` validation without provisioning cloud resources.
