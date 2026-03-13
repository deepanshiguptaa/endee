"""App configuration (environment variables + sensible defaults)."""

import os

ENDEE_URL: str = os.getenv("ENDEE_URL", "http://localhost:8080")
ENDEE_AUTH_TOKEN: str = os.getenv("ENDEE_AUTH_TOKEN", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Backwards-compatible alias (used by older code in this repo)
ENDEE_AUTH: str = ENDEE_AUTH_TOKEN

INDEX_NAME: str = os.getenv("INDEX_NAME", "ai_resources")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))  # all-MiniLM-L6-v2

# Minimum similarity (0–1) to include in results; set to 0 to disable
MIN_SIMILARITY: float = float(os.getenv("MIN_SIMILARITY", "0"))
