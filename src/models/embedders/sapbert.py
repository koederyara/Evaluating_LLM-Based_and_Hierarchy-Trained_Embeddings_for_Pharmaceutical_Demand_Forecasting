"""SapBERT embedding backend.

SapBERT (Self-Alignment Pretraining for BERT) is trained for biomedical entity
linking. Unlike general BERT models, it uses the [CLS] token as the entity
representation — not mean pooling.

Reads config from environment variables (via .env):
  SAPBERT_MODEL      (default: cambridgeltl/SapBERT-from-PubMedBERT-fulltext)
  SAPBERT_BATCH_SIZE (default: 64)
"""

import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = os.getenv(
    "SAPBERT_MODEL",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
)
BATCH_SIZE = int(os.getenv("SAPBERT_BATCH_SIZE", 64))


def _embed_batch(
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    texts: list[str],
    device: torch.device,
) -> np.ndarray:
    texts = [str(t) if t is not None else "" for t in texts]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)
    # [CLS] token is the entity representation SapBERT was trained with
    return output.last_hidden_state[:, 0, :].cpu().numpy()


def embed(texts: list[str]) -> tuple[np.ndarray, str]:
    """Embed texts via SapBERT. Returns (embeddings float32, model_name)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model {MODEL_NAME} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    all_embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        all_embeddings.append(_embed_batch(tokenizer, model, batch, device))
        print(f"  Embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")

    return np.vstack(all_embeddings).astype(np.float32), MODEL_NAME
