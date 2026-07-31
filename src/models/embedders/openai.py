"""OpenAI embedding backend.

Reads config from environment variables (via .env):
  OPENAI_API_KEY
  OPENAI_EMBEDDING_MODEL      (default: text-embedding-3-large)
  OPENAI_EMBEDDING_DIMENSIONS (default: 3072)
  OPENAI_EMBEDDING_BATCH_SIZE (default: 256)
"""

import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
DIMENSIONS = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", 3072))
BATCH_SIZE = int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE", 256))


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=MODEL,
        input=texts,
        dimensions=DIMENSIONS,
    )
    # API returns items sorted by index, but sort defensively
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def embed(texts: list[str]) -> tuple[np.ndarray, str]:
    """Embed texts via OpenAI API. Returns (embeddings float32, model_name)."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        all_embeddings.extend(_embed_batch(client, batch))
        print(f"  Embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")

    return np.array(all_embeddings, dtype=np.float32), MODEL
