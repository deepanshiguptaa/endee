#!/usr/bin/env python3
"""
Ingest markdown files from data/ into Endee.

Run this after starting Endee (e.g. docker run ... endeeio/endee-server:latest).
Chunks documents by paragraph, embeds each chunk, and upserts to the index.

Usage:
  python ingest.py           # Upsert chunks (adds/updates by stable ID)
  python ingest.py --clear   # Delete index, recreate, then upsert (fresh start)
"""

import argparse
import hashlib
from pathlib import Path

from endee import Endee, Precision

from config import ENDEE_AUTH, ENDEE_URL, EMBEDDING_DIM, INDEX_NAME
from embedding import embed


DATA_DIR = Path(__file__).parent / "data"


def chunk_id(source: str, chunk_index: int, text: str) -> str:
    """Stable ID so re-ingesting doesn't create duplicates."""
    payload = f"{source}:{chunk_index}:{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def load_docs():
    """Load all .md files and split into chunks (paragraphs)."""
    chunks = []
    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = path.stem.replace("-", " ").title()
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, para in enumerate(paras):
            chunks.append({
                "text": para,
                "source": path.name,
                "title": title,
                "chunk_idx": i,
            })
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest markdown docs into Endee")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete index and recreate before upserting",
    )
    args = parser.parse_args()

    chunks = load_docs()
    if not chunks:
        print("No .md files found in data/")
        return

    print(f"Loaded {len(chunks)} chunks from {DATA_DIR}")

    client = Endee(ENDEE_AUTH) if ENDEE_AUTH else Endee()
    client.set_base_url(f"{ENDEE_URL.rstrip('/')}/api/v1")

    if args.clear:
        try:
            client.delete_index(name=INDEX_NAME)
            print(f"Deleted index '{INDEX_NAME}'")
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                print(f"Index '{INDEX_NAME}' did not exist, skipping delete")
            else:
                raise

    try:
        client.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            space_type="cosine",
            precision=Precision.INT8,
        )
        print(f"Created index '{INDEX_NAME}'")
    except Exception as e:
        if "already exists" in str(e).lower() or "exist" in str(e).lower():
            print(f"Index '{INDEX_NAME}' already exists, will upsert into it")
        else:
            raise

    index = client.get_index(name=INDEX_NAME)

    texts = [c["text"] for c in chunks]
    vectors = embed(texts)

    records = []
    for chunk, vec in zip(chunks, vectors):
        records.append(
            {
                "id": chunk_id(chunk["source"], chunk["chunk_idx"], chunk["text"]),
                "vector": vec,
                "meta": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "title": chunk["title"],
                },
            }
        )

    batch_size = 50
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        index.upsert(batch)
        print(f"Upserted {min(i + batch_size, len(records))}/{len(records)}")

    print("Done. Try searching via the API or UI.")


if __name__ == "__main__":
    main()
