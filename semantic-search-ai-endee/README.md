# AI Resources Search — Semantic Search + RAG with Endee

A small semantic search and RAG demo over AI/ML learning docs. Uses [Endee](https://github.com/endee-io/endee) as the vector database and sentence-transformers for embeddings (runs locally, no API keys required for search).

> **Evaluation note**: This project was built for the Endee internship evaluation. Per the requirements: the official [Endee repo](https://github.com/endee-io/endee) should be starred and forked first. This app is self-contained and uses Endee via Docker + Python SDK — you can run it as-is or drop it into your forked Endee repo (e.g. under `examples/`).

---

## What’s going on here

**Problem**: Keyword search is limited. You want to find docs by meaning, not just exact words. “How does retrieval-augmented generation work?” should surface content about RAG even if the exact phrase isn’t there.

**Approach**: Embed docs into vectors, store them in Endee, and at query time embed the question and run a similarity search. Optionally, feed the top results to an LLM for a synthesized answer (RAG).

---

## Project layout

```
.
├── data/           # Markdown docs (indexed by ingest)
├── static/         # Simple HTML/JS UI
├── api.py          # FastAPI: /search, /ask
├── ingest.py       # Loads data/, embeds, upserts to Endee
├── embedding.py    # sentence-transformers wrapper
├── config.py       # Config from env
├── requirements.txt
└── docker-compose.yml   # Endee server
```

---

## How Endee fits in

Endee is the vector store. It’s used for:

1. **Index**: One dense index, `ai_resources`, 384‑dimensional vectors, cosine similarity.
2. **Upsert**: Ingest script embeds each doc chunk and calls `index.upsert(...)`.
3. **Query**: API embeds the user query and calls `index.query(vector=..., top_k=5)`.

Endee runs as a separate process (Docker). The Python app talks to it over HTTP via the `endee` Python SDK. No schema migrations — we create the index once and upsert/query.

---

## Setup

### 1. Prerequisites

- Python 3.10+
- Docker (for Endee)

### 2. Start Endee

```bash
docker compose up -d
```

Endee runs at `http://localhost:8080`. You can open that URL to see its dashboard.

### 3. Python env

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 4. Ingest data

```bash
python ingest.py          # Upsert (stable IDs, no duplicates on re-run)
python ingest.py --clear  # Delete index, recreate, then upsert (fresh start)
```

This reads all `.md` files in `data/`, chunks them by paragraph, embeds with `all-MiniLM-L6-v2`, and upserts to Endee. First run will download the model (~80MB). Re-running without `--clear` updates existing chunks by stable ID.

### 5. Run the API

```bash
uvicorn api:app --reload --port 8000
```

Open **http://localhost:8000** — you’ll see the search UI.

---

## Usage

- **Search**: Type a question, click Search. You get the top‑k most similar chunks with similarity scores.
- **Ask (RAG)**: Same flow, but if `OPENAI_API_KEY` is set, the top chunks are passed to GPT‑4o‑mini for a generated answer. Without the key, you only get the context.

---

## API

- `GET /` — search UI
- `GET /health` — health check
- `POST /search` — `{"query": "...", "top_k": 5}` → list of matching chunks
- `POST /ask` — `{"query": "...", "top_k": 3}` → RAG answer + context (if API key set)

---

## Configuration

Copy `.env.example` to `.env` and adjust if needed:

- `ENDEE_URL` — default `http://localhost:8080`
- `ENDEE_AUTH_TOKEN` — only if Endee is configured with auth
- `OPENAI_API_KEY` — optional, for RAG answers
- `MIN_SIMILARITY` — optional, filter results below this score (0–1); default 0

---

## Internals

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims). Runs locally.
- **Chunking**: Split by paragraph (`\n\n`). No overlap — simple and fine for small docs.
- **RAG**: Uses OpenAI `gpt-4o-mini` when the key is set. Could be swapped for another provider.

---

## Adding your own docs

Drop more `.md` files into `data/` and run `python ingest.py` again. Chunks use stable IDs (hash of source + text), so re-running updates rather than duplicates. Use `python ingest.py --clear` for a fresh index.

---

## License

MIT.
