"""
Embeddings utility module.

Provides deterministic fixed-length embeddings (128 or 512 dims) suitable
for both development/demo and production use.

Priority order:
  1. Ollama (via REST API) — real VLM embeddings when server is available
  2. Sentence-transformers — semantic embeddings when installed
  3. Deterministic MD5-seeded random — always works, reproducible

Usage:
    from src.common.embeddings import get_embedding, get_text_embedding

    vec = get_text_embedding("a photo of the Golden Gate Bridge at sunset")
    # Returns list[float] of length DIM
"""

import hashlib
import os
import warnings
import numpy as np

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "ollama")
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", "11434"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "128"))


# --------------------------------------------------------------------------- #
# Deterministic fallback                                                        #
# --------------------------------------------------------------------------- #

def deterministic_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    Produce a reproducible unit-norm vector from a text string.

    Uses MD5 to seed a numpy RNG — same input always yields the same vector.
    Suitable for testing, CI, and environments without a GPU.
    """
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype("float32")
    norm = np.linalg.norm(vec) + 1e-9
    return (vec / norm).tolist()


# --------------------------------------------------------------------------- #
# Ollama embedding                                                              #
# --------------------------------------------------------------------------- #

def _ollama_embedding(text: str, model: str = "nomic-embed-text") -> list[float] | None:
    """Try Ollama's /api/embeddings endpoint. Returns None on any failure."""
    try:
        import requests
        r = requests.post(
            f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=10,
        )
        if r.status_code == 200:
            emb = r.json().get("embedding", [])
            if emb:
                arr = np.array(emb, dtype="float32")
                norm = np.linalg.norm(arr) + 1e-9
                return (arr / norm).tolist()
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Sentence-transformer embedding                                                #
# --------------------------------------------------------------------------- #

_st_model = None  # lazy-loaded

def _sentence_transformer_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float] | None:
    """Try sentence-transformers. Returns None if not installed."""
    global _st_model
    try:
        from sentence_transformers import SentenceTransformer
        if _st_model is None:
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = _st_model.encode(text, normalize_embeddings=True)
        # Truncate or pad to desired dim
        arr = np.array(emb, dtype="float32")
        if len(arr) >= dim:
            arr = arr[:dim]
        else:
            arr = np.pad(arr, (0, dim - len(arr)))
        norm = np.linalg.norm(arr) + 1e-9
        return (arr / norm).tolist()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #

def get_text_embedding(text: str, dim: int = EMBEDDING_DIM, prefer: str = "auto") -> list[float]:
    """
    Get a text embedding vector.

    Parameters
    ----------
    text   : str  — input text
    dim    : int  — desired embedding dimension (default: EMBEDDING_DIM env var)
    prefer : str  — "ollama" | "sentence_transformer" | "deterministic" | "auto"

    Returns
    -------
    list[float] of length `dim`, unit-norm.
    """
    if prefer == "ollama":
        emb = _ollama_embedding(text)
        if emb is None:
            warnings.warn("Ollama unavailable — falling back to deterministic embedding.")
            return deterministic_embedding(text, dim)
        return emb

    if prefer == "sentence_transformer":
        emb = _sentence_transformer_embedding(text, dim)
        if emb is None:
            warnings.warn("sentence-transformers unavailable — falling back to deterministic embedding.")
            return deterministic_embedding(text, dim)
        return emb

    if prefer == "deterministic":
        return deterministic_embedding(text, dim)

    # "auto" — try in order: ollama → sentence_transformer → deterministic
    emb = _ollama_embedding(text)
    if emb:
        return emb

    emb = _sentence_transformer_embedding(text, dim)
    if emb:
        return emb

    return deterministic_embedding(text, dim)


def get_embedding(row: dict, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    Build an embedding from a metadata row dict.

    Uses caption + vlm_labels if present, otherwise falls back to
    tags + image_id. Always returns a fixed-length vector.
    """
    parts = []
    if row.get("caption"):
        parts.append(str(row["caption"]))
    if row.get("vlm_labels"):
        labels = row["vlm_labels"]
        if isinstance(labels, list):
            parts.extend(labels)
        else:
            parts.append(str(labels))
    if row.get("tags"):
        tags = row["tags"]
        if isinstance(tags, list):
            parts.extend(tags)
        else:
            parts.append(str(tags))
    if not parts:
        parts.append(str(row.get("image_id", "unknown")))

    text = " ".join(parts)
    return get_text_embedding(text, dim)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two unit-norm vectors."""
    va = np.array(a, dtype="float32")
    vb = np.array(b, dtype="float32")
    return float(np.dot(va, vb))
