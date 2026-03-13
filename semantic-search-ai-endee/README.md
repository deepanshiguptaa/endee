# AI assets seek — Semantic search + RAG with Endee

A small semantic search and RAG demo over AI/ML studying medical doctors. makes use of [Endee](https://github.com/endee-io/endee) because the vector database and sentence-transformers for embeddings (runs regionally, no API keys required for search).

> **assessment observe**: This undertaking became constructed for the Endee internship assessment. according to the requirements: the respectable [Endee repo](https://github.com/endee-io/endee) should be starred and forked first. This app is self-contained and makes use of Endee thru Docker + Python SDK — you can run it as-is or drop it into your forked Endee repo (e.g. beneath `examples/`).

---

## What’s going on here

**hassle**: key-word search is confined. You need to discover docs through meaning, now not just specific phrases. “How does retrieval-augmented technology work?” must floor content approximately RAG even supposing the exact word isn’t there.

**approach**: Embed medical doctors into vectors, shop them in Endee, and at question time embed the question and run a similarity search. Optionally, feed the top outcomes to an LLM for a synthesized solution (RAG).

---

## mission format

```
.
├── facts/          # Markdown medical doctors (listed through ingest)
├── static/         # simple HTML/JS UI
├── api.py          # FastAPI: /seek, /ask
├── ingest.py       # hundreds facts/, embeds, upserts to Endee
├── embedding.py    # sentence-transformers wrapper
├── config.py       # Config from env
├── requirements.txt
└── docker-compose.yml   # Endee server
```

---

## How Endee suits in

Endee is the vector store. It’s used for:

1. **Index**: One dense index, `ai_resources`, 384‑dimensional vectors, cosine similarity.
2. **Upsert**: Ingest script embeds every doc bite and calls `index.upsert(...)`.
3. **query**: API embeds the user query and calls `index.query(vector=..., top_k=five)`.

Endee runs as a separate system (Docker). The Python app talks to it over HTTP through the `endee` Python SDK. No schema migrations — we create the index as soon as and upsert/query.

---

## Setup

### 1. conditions

- Python three.10+
- Docker (for Endee)

### 2. begin Endee

```bash
docker compose up -d
```

Endee runs at `http://localhost:8080`. you may open that URL to peer its dashboard.

### 3. Python env

```bash
python -m venv .venv
.venvScriptsactivate   # home windows
# source .venv/bin/set off   # Mac/Linux

pip install -r necessities.txt
```

### 4. Ingest facts

```bash
python ingest.py          # Upsert (stable IDs, no duplicates on re-run)
python ingest.py --clean  # Delete index, recreate, then upsert (fresh start)
```

This reads all `.md` files in `statistics/`, chunks them by using paragraph, embeds with `all-MiniLM-L6-v2`, and upserts to Endee. First run will download the model (~80MB). Re-going for walks with out `--clear` updates current chunks through solid identification.

### 5. Run the API

```bash
uvicorn api:app --reload --port 8000
```

Open **http://localhost:8000** — you’ll see the search UI.

---

## utilization

- **search**: kind a query, click seek. You get the pinnacle‑ok most similar chunks with similarity ratings.
- **Ask (RAG)**: equal drift, however if `OPENAI_API_KEY` is about, the pinnacle chunks are handed to GPT‑4o‑mini for a generated solution. with out the key, you only get the context.

---

## API

- `GET /` — search UI
- `GET /fitness` — health take a look at
- `post /seek` — `{"question": "...", "top_k": 5}` → list of matching chunks
- `post /ask` — `{"question": "...", "top_k": three}` → RAG solution + context (if API key set)

---

## Configuration

replica `.env.example` to `.env` and regulate if wished:

- `ENDEE_URL` — default `http://localhost:8080`
- `ENDEE_AUTH_TOKEN` — only if Endee is configured with auth
- `OPENAI_API_KEY` — optional, for RAG solutions
- `MIN_SIMILARITY` — optional, clear out outcomes under this score (0–1); default zero

---

## Internals

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims). Runs domestically.
- **Chunking**: cut up through paragraph (`nn`). No overlap — easy and best for small medical doctors.
- **RAG**: uses OpenAI `gpt-4o-mini` while the important thing is ready. may be swapped for some other provider.

---

## including your own doctors

Drop extra `.md` documents into `statistics/` and run `python ingest.py` once more. Chunks use stable IDs (hash of supply + textual content), so re-running updates instead of duplicates. Use `python ingest.py --clean` for a sparkling index.

---

## License

MIT.