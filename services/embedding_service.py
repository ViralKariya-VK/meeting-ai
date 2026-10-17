"""Sentence embeddings. fastembed (ONNX, no torch) with bge-small-en-v1.5; a hashed TF-IDF-style fallback
keeps the app and tests working offline."""
from __future__ import annotations

import numpy as np

from . import config

_model = None
_backend_used = None
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


def _hash_embed(texts: list[str], dim: int = 512) -> np.ndarray:
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(n_features=dim, ngram_range=(1, 2), alternate_sign=False, norm=None,
                           stop_words="english")
    return _normalize(hv.transform(texts).toarray().astype(np.float32))


def _load_fastembed():
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=config.EMBED_MODEL)
    return _model


def backend() -> str:
    """Name of the backend that will be (or was) used, stored with the index to detect mismatches."""
    global _backend_used
    if _backend_used:
        return _backend_used
    if config.EMBED_BACKEND() == "hash":
        _backend_used = "hash"
    else:
        try:
            _load_fastembed()
            _backend_used = "fastembed:" + config.EMBED_MODEL
        except Exception:
            _backend_used = "hash"
    return _backend_used


def embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    if backend() == "hash":
        return _hash_embed(texts)
    model = _load_fastembed()
    inputs = [BGE_QUERY_PREFIX + t for t in texts] if is_query else texts
    return _normalize(np.array(list(model.embed(inputs)), dtype=np.float32))
