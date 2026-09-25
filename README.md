# Kubernetes Docs RAG Assistant

[GitHub Repo](https://github.com/arthuersundar/rag-platform)

A Retrieval-Augmented Generation (RAG) system that answers Kubernetes questions
using the official Kubernetes documentation — deployed on Kubernetes, with
automated evals for retrieval accuracy and answer grounding.

**Status:** core pipeline, Kubernetes deployment, and evals complete. CI,
hybrid search/reranking, and the LLM gateway are in progress (see Roadmap).

## What it does

Ask a question like *"how do I set a memory limit on a container?"* and get an
answer grounded in the official docs, with source links — instead of an LLM
guessing from memory. The system is tested to:
- **Answer correctly** when the docs contain the answer, citing sources.
- **Refuse honestly** ("I don't know") when a question falls outside the
  loaded documentation, rather than falling back on the model's own
  unverifiable training knowledge.

## Architecture

```
Question → Embed (nomic-embed-text) → pgvector similarity search (top 5 chunks)
        → Context + Question → Llama 3.2 3B → Answer + Sources
```

Running on Kubernetes:
- **Postgres + pgvector** — StatefulSet, stores chunks and embeddings, HNSW index for fast similarity search.
- **RAG API** — FastAPI service (Deployment + Service, deployed via Helm), exposes `POST /query`.
- **Ollama** — serves `llama3.2:3b` (generation) and `nomic-embed-text` (embeddings). Runs on the host during local development; see Roadmap for GPU-backed serving.

### Components

- **Ingestion** (`app/ingest.py`): clones the Kubernetes docs, splits them into
  ~300-word chunks by markdown heading, embeds each chunk, and stores it in
  Postgres with a content hash — so re-running the script only embeds new or
  changed content, and skips everything else.
- **Retrieval** (`app/search.py`): embeds a question and finds the 5 nearest
  chunks by cosine similarity.
- **API** (`app/main.py`): retrieves context, prompts Llama to answer *only*
  from that context, and returns the answer with clickable source links.
- **Evals** (`evals/`): automated scripts that measure retrieval accuracy and
  answer grounding — see below.

## Stack

Kubernetes (kind/Docker Desktop) · Helm · Docker · Ollama (llama3.2:3b, nomic-embed-text) ·
PostgreSQL + pgvector · FastAPI · Python

## Dataset

~[X] files / ~[Y] chunks from the official [Kubernetes documentation](https://github.com/kubernetes/website)
(`content/en/docs`), licensed under CC BY 4.0. Not vendored — `app/ingest.py`
fetches and loads it.

## Evals

Two automated scripts, run against the real, deployed API — not a mocked copy:

| Script | Measures | Result (initial 9-case set) |
|---|---|---|
| `evals/eval_retrieval.py` | Does the top-5 retrieval include the expected source page? | 6/6 = 100% |
| `evals/eval_answers.py` | Refusal accuracy (correctly says "I don't know" when out of scope) | 3/3 = 100% |
| `evals/eval_answers.py` | Grounded-answer rate (an LLM-judge call checks the answer is supported by the retrieved sources, not fabricated) | 6/6 = 100% |

**Note on sample size:** these results are from an initial 9-question set and
are a proof of concept, not a full validation — the test set is being expanded
to 20-30 questions across more topics before drawing firmer conclusions.
One early "failure" was traced back to an overly strict test assertion, not a
real retrieval miss — a useful reminder to interrogate eval failures rather
than take them at face value.

Run them yourself:
```bash
python evals/eval_retrieval.py
python evals/eval_answers.py
```

## Running it

```bash
# 1. Cluster + models
kind create cluster --name rag
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. Deploy Postgres/pgvector
kubectl apply -f deploy/postgres.yaml

# 3. Load the docs
python app/ingest.py

# 4. Build and deploy the API
docker build -t rag-api:latest .
kind load docker-image rag-api:latest --name rag   # skip if using Docker Desktop Kubernetes
helm install rag charts/rag-api

# 5. Query it
kubectl port-forward svc/rag-rag-api 8000:8000
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "how do I set a memory limit on a container?"}'
```

## Roadmap

**In progress**
- [ ] Expand eval set to 20-30 questions across more topics
- [ ] Wire evals into GitHub Actions CI
- [ ] Hybrid search (vector + Postgres full-text) and reranking

**Planned**
- [ ] LLM gateway: routing, per-team token budgets, rate limiting, caching
- [ ] Access-controlled RAG: document-level permissions, PII masking
- [ ] Incremental ingestion via Kubernetes CronJob
- [ ] Simple web UI
- [ ] Response streaming

**Stretch**
- [ ] vLLM GPU serving with cost-per-million-tokens comparison vs. a hosted API
- [ ] LoRA fine-tune (citation formatting, refusal consistency) with before/after eval comparison

## Notable design decisions

- **Content-hash based ingestion** makes re-running ingestion idempotent and
  cheap: unchanged chunks are skipped, so a daily refresh only costs the time
  to embed what actually changed.
- **The system prompt restricts generation to retrieved context only**,
  verified by testing that the assistant refuses out-of-scope questions
  (e.g. Docker Compose, version-specific details not in the loaded docs)
  instead of answering from the model's general training knowledge.
- **Config via env vars and a Kubernetes Secret** (not hardcoded), so the same
  image runs unchanged across environments.

## Author

Built by Sundar Djeabalane — [LinkedIn](https://www.linkedin.com/in/sundar-djeabalane/) — [GitHub](https://github.com/arthuersundar/)
