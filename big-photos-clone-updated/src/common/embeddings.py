from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Sequence

import numpy as np

DEFAULT_DIM = 128
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def _signed_index(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
    idx = int.from_bytes(digest[:8], "little") % dim
    sign = 1.0 if (digest[8] % 2 == 0) else -1.0
    mag = 0.5 + (digest[9] / 255.0)
    return idx, sign * mag


def normalize(vec: Sequence[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr.tolist()
    return (arr / norm).astype(np.float32).tolist()


def text_embedding(text: str, dim: int = DEFAULT_DIM) -> List[float]:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _tokens(text)
    if not tokens:
        tokens = ["empty"]
    for tok in tokens:
        idx, val = _signed_index(tok, dim)
        vec[idx] += val
    return normalize(vec)


def image_id_embedding(image_id: str, dim: int = DEFAULT_DIM) -> List[float]:
    return text_embedding(f"image {image_id}", dim=dim)


def record_embedding(
    caption: str = "",
    labels: Iterable[str] | None = None,
    category: str = "",
    tags: Iterable[str] | None = None,
    dim: int = DEFAULT_DIM,
) -> List[float]:
    labels = [str(x) for x in (labels or [])]
    tags = [str(x) for x in (tags or [])]
    text = " ".join([caption or "", category or "", " ".join(labels), " ".join(tags)])
    return text_embedding(text, dim=dim)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def embedding_to_list(value, dim: int = DEFAULT_DIM) -> List[float]:
    if value is None:
        return [0.0] * dim
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).tolist()
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, tuple):
        return [float(x) for x in value]
    if isinstance(value, str):
        stripped = value.strip().strip("[]")
        if not stripped:
            return [0.0] * dim
        return [float(x) for x in stripped.replace(",", " ").split()]
    try:
        return [float(x) for x in list(value)]
    except Exception:
        return [0.0] * dim
