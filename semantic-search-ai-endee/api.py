"""
FastAPI server for semantic search and RAG.

Endpoints:
  GET  /health  — simple health check
  POST /search  — semantic search (returns top-k chunks)
  POST /ask     — RAG: search + optional LLM answer (needs OPENAI_API_KEY)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from config import ENDEE_AUTH, ENDEE_URL, INDEX_NAME, MIN_SIMILARITY, OPENAI_API_KEY
from embedding import embed_one
from endee import Endee
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="AI Resources Search",
    description="Semantic search + RAG over AI/ML learning docs, powered by Endee",
)   

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy client init (single process, good enough for demo).
_client: Endee | None = None


def _base_url() -> str:
    return f"{ENDEE_URL.rstrip('/')}/api/v1"


def get_client() -> Endee:
    global _client
    if _client is None:
        _client = Endee(ENDEE_AUTH) if ENDEE_AUTH else Endee()
        _client.set_base_url(_base_url())
    return _client


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class AskRequest(BaseModel):
    query: str
    top_k: int = 3


def _get_index():
    try:
        client = get_client()
        return client.get_index(name=INDEX_NAME)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not reach Endee: {e}")


def _extract_text(r: dict[str, Any]) -> str:
    meta = r.get("meta") or {}
    return meta.get("text", "") or ""


def _passes_similarity(r: dict[str, Any]) -> bool:
    if MIN_SIMILARITY <= 0:
        return True
    sim = r.get("similarity")
    return sim is not None and sim >= MIN_SIMILARITY


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the search UI."""
    path = Path(__file__).parent / "static" / "index.html"
    if path.exists():
        return FileResponse(path)
    return HTMLResponse("<p>Add static/index.html for the UI</p>")


@app.post("/search")
def search(req: SearchRequest):
    """Semantic search — returns top-k matching chunks."""
    index = _get_index()

    vec = embed_one(req.query)
    results = index.query(vector=vec, top_k=req.top_k)

    out = []
    for r in results:
        if not _passes_similarity(r):
            continue
        meta = r.get("meta") or {}
        out.append(
            {
                "id": r.get("id"),
                "text": meta.get("text", ""),
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "similarity": r.get("similarity"),
            }
        )
    return {"results": out}


@app.post("/ask")
def ask(req: AskRequest):
    """
    RAG: search for relevant chunks, then optionally generate an answer.
    If OPENAI_API_KEY is set, uses GPT to synthesize. Otherwise returns context only.
    """
    index = _get_index()

    vec = embed_one(req.query)
    try:
        results = index.query(vector=vec, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    if not results:
        return {"answer": None, "context": "", "sources": []}

    context_parts = [_extract_text(r) for r in results if _extract_text(r)]

    context = "\n\n".join(context_parts)

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client_oa = OpenAI(api_key=OPENAI_API_KEY)
            resp = client_oa.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Answer the user's question using only the provided context. If the context doesn't contain enough info, say so.",
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n\n{context}\n\nQuestion: {req.query}",
                    },
                ],
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            return {
                "answer": None,
                "context": context,
                "error": str(e),
            }
    else:
        answer = None

    return {
        "answer": answer,
        "context": context,
        "sources": [
            {"text": (r.get("meta") or {}).get("text", "")[:200] + "..."}
            for r in results
        ],
    }
