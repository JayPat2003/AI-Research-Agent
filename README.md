# AI Research & Decision Intelligence Platform

Give the system a complex research question. Specialized agents search a private knowledge base and public sources, cross-check evidence, and return a **cited decision briefing**.

The agents are domain-agnostic. This repository includes a **public healthcare demo corpus** (cardiometabolic outcomes trials: GLP-1 vs SGLT2) so the stack can be run without proprietary data. Replace `data/seed/` to point the same pipeline at another field.

This is an analyst tool, not medical advice.

## Architecture

```
User → Next.js → FastAPI
              → Router (Gemini 2.5 Flash-Lite)
              → Retrieval Agent  (hybrid RAG: pgvector + Postgres FTS + RRF + BGE rerank)
              → Research Agent   (DuckDuckGo, arXiv, Semantic Scholar)
              → Data Agent       (read-only Text-to-SQL on demo `clinical_trials`)
              → Report Generator (Gemini 2.5 Flash)
              → Critic Agent     (pass / retrieve-again, max 2 retries)
              → Cited report + conversation memory
```

LangChain / LangGraph own agents, tools, and structured output. LlamaIndex owns document loading, cleaning, and chunking.

## Requirements

- **Linux** (x86_64) or **macOS** (Intel or Apple Silicon)
- Python 3.12 (Conda via [`environment.yml`](environment.yml), or the API container)
- Docker and Docker Compose
- Node.js 22 for a local frontend dev server
- A free [Gemini API key](https://aistudio.google.com/apikey)

On **Apple Silicon**, use a native `arm64` Conda distribution (for example Miniforge). `uname -m` should report `arm64`.

Git ignores machine-local artifacts: `.env`, `models/`, `node_modules/`, `data/uploads/`, and Docker volumes. After a fresh clone, recreate the Conda env, copy `.env.example` → `.env`, and let seed ingestion rebuild the demo index.

## Quick start

```bash
git clone https://github.com/JayPat2003/AI-Research-Agent.git
cd AI-Research-Agent
conda env create -f environment.yml
conda activate ai-research-agent
cp .env.example .env   # set GEMINI_API_KEY
docker compose up --build
```

UI: [http://localhost:3000](http://localhost:3000) · API health: [http://localhost:8000/health](http://localhost:8000/health)

Example question:

> Compare GLP-1 receptor agonists and SGLT2 inhibitors for adults with type 2 diabetes and established cardiovascular disease. What do the major outcomes trials show, which populations were studied, and which class is more suitable as the first add-on in a research briefing for a clinical-operations team?

Follow up with “What about kidney outcomes?” The router resolves that against conversation summary and recent turns, not a raw transcript dump.

If `pip` still resolves to `~/.local` after `conda activate`, run `export PYTHONNOUSERSITE=1`.

### API on Conda, Postgres and Redis in Docker

```bash
docker compose up postgres redis
conda activate ai-research-agent
cd backend
uvicorn app.main:app --reload --port 8000
# second terminal
python -m app.worker
# third terminal
cd frontend && npm install && npm run dev
```

Embedding and reranker weights download once per machine into `./models`.

## Model stack (free APIs)

| Role | Default | Notes |
| --- | --- | --- |
| Research, critic, report | `gemini-2.5-flash` | Long context, structured citations |
| Router, SQL, conversation summary | `gemini-2.5-flash-lite` | Conserves free-tier quota |
| Embeddings | local `BAAI/bge-small-en-v1.5` (fastembed) | No embedding API cost |
| Rerank | local `BAAI/bge-reranker-base` | Falls back to MiniLM if needed |

Gemini free-tier RPM is the main constraint. Agent LLM calls are sequential. On `429` responses, set `LLM_PROVIDER=groq` and `GROQ_API_KEY`. Groq has no embeddings; local BGE still handles retrieval.

## Evaluation

After the knowledge base is seeded:

```bash
cd backend
python -m app.eval.run --variant naive
python -m app.eval.run --variant hybrid
python -m app.eval.run --variant hybrid_rerank
python -m app.eval.run --variant hybrid_rerank --generation   # extra Gemini calls
```

Metrics: Recall@5, Precision@5, MRR, nDCG@5, plus optional faithfulness and citation checks.

```bash
cd backend
EMBEDDING_BACKEND=hash python -m pytest -q
```

## Demo corpus

Included: type 2 diabetes treatment classes, SGLT2 and GLP-1 outcomes trials, comparison framing, evidence grading, and a `clinical_trials` table of well-known CVOTs.

Do not commit API keys or proprietary documents. Uploads belong in `data/uploads/` (gitignored).

Deferred: Qdrant as a second vector store, AWS (ECS, RDS, S3, SQS), production GitHub Actions deploy.

## Ingestion

`POST /ingest/file` · `POST /ingest/url` · `POST /ingest/arxiv` · `POST /ingest/text`

PDF, DOCX, Markdown, HTML, URL, and arXiv ids are parsed with LlamaIndex, chunked (~512 / 64 overlap), embedded locally, and indexed in Postgres (`vector` + `tsvector`). A Redis worker runs ingestion off the request thread.
