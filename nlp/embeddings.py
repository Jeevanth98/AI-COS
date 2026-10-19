"""
AI-COS — Embedding Generator and Persistent Store
Uses sentence-transformers/all-MiniLM-L6-v2 to produce 384-dim embeddings.
Vectors are stored as NumPy .npy files; metadata lives in SQLite.

Design decisions:
- One .npy file per document version (prevents unbounded single-file growth).
- Each row in the .npy matrix maps 1-to-1 to an EmbeddingMetadata row via vector_index.
- Model is cached as a module-level singleton to avoid reload costs.
- Before reusing stored embeddings, model name and dimension are validated.
"""

from __future__ import annotations
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from config.settings import EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDINGS_DIR, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)

_model = None  # module-level singleton


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            _model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded (dim=%d).", EMBEDDING_DIM)
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise RuntimeError(
                f"Cannot load embedding model '{EMBEDDING_MODEL}'. "
                "Ensure sentence-transformers is installed and the model can be downloaded."
            ) from exc
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of strings to a float32 NumPy array of shape (n, EMBEDDING_DIM).
    Raises RuntimeError if the model cannot be loaded.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit-normalise for fast cosine via dot product
    )
    return vectors.astype(np.float32)


# ── Persistence ────────────────────────────────────────────────────────────────

def _version_npy_path(version_id: str) -> Path:
    return EMBEDDINGS_DIR / f"{version_id}.npy"


def save_embeddings(version_id: str, vectors: np.ndarray) -> Path:
    """Persist vectors for a version. Overwrites if already present."""
    path = _version_npy_path(version_id)
    np.save(str(path), vectors)
    logger.debug("Saved %d embeddings → %s", vectors.shape[0], path.name)
    return path


def load_embeddings(version_id: str) -> Optional[np.ndarray]:
    """Load persisted embeddings for a version. Returns None if not found."""
    path = _version_npy_path(version_id)
    if not path.exists():
        return None
    arr = np.load(str(path))
    logger.debug("Loaded %d embeddings ← %s", arr.shape[0], path.name)
    return arr


def delete_embeddings(version_id: str) -> bool:
    """Remove the .npy file for a version. Returns True if deleted."""
    path = _version_npy_path(version_id)
    if path.exists():
        path.unlink()
        return True
    return False


save_version_embeddings = save_embeddings
load_version_embeddings = load_embeddings


# ── Generation pipeline ────────────────────────────────────────────────────────

def generate_and_save_embeddings(
    version_id: str,
    items: List[Dict[str, Any]],
    source_type: str = "chunk",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Generate embeddings for a list of chunk or statement dicts and save them.

    Parameters
    ----------
    version_id  : DocumentVersion.version_id
    items       : list of dicts, each must have 'text' and 'id' keys
                  ('id' = chunk_id or statement_id)
    source_type : "chunk" or "statement"

    Returns
    -------
    (vectors, metadata_list)
    metadata_list: list of dicts ready to create EmbeddingMetadata rows
    """
    texts = [item["text"] for item in items]
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32), []

    vectors = embed_texts(texts)
    save_embeddings(version_id, vectors)

    metas: List[Dict[str, Any]] = []
    for idx, (item, _vec) in enumerate(zip(items, vectors)):
        meta: Dict[str, Any] = {
            "version_id": version_id,
            "source_type": source_type,
            "source_text": item["text"][:2000],  # cap for DB storage
            "model_name": EMBEDDING_MODEL,
            "model_version": "v1",
            "vector_file": str(_version_npy_path(version_id)),
            "vector_index": idx,
            "embedding_dim": int(vectors.shape[1]),
        }
        if source_type == "chunk":
            meta["chunk_id"] = item.get("id")
            meta["statement_id"] = None
        else:
            meta["statement_id"] = item.get("id")
            meta["chunk_id"] = None
        metas.append(meta)

    return vectors, metas


# ── Semantic search ────────────────────────────────────────────────────────────

def cosine_similarity_batch(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between a single query vector and a corpus matrix.
    Assumes both are L2-normalised (dot product == cosine similarity).
    """
    if corpus_vecs.ndim == 1:
        corpus_vecs = corpus_vecs.reshape(1, -1)
    return corpus_vecs @ query_vec  # shape (n,)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix between two sets of L2-normalised vectors."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return a @ b.T


def semantic_search(
    query: str,
    all_metas: List[Dict[str, Any]],
    top_k: int = 10,
    threshold: float = 0.35,
    filter_version_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search for the most semantically similar passages to *query*.

    Parameters
    ----------
    query              : natural-language query string
    all_metas          : list of EmbeddingMetadata-like dicts
                         (must include version_id, vector_file, vector_index, source_text)
    top_k              : maximum results
    threshold          : minimum cosine similarity score
    filter_version_ids : if provided, restrict search to these versions

    Returns
    -------
    list of result dicts sorted by similarity descending:
        {similarity, source_text, version_id, chunk_id, statement_id, source_type}
    """
    if not all_metas or not query.strip():
        return []

    query_vec = embed_texts([query])[0]

    # Group metas by their .npy file to minimise disk reads
    file_groups: Dict[str, List[Dict]] = {}
    for meta in all_metas:
        if filter_version_ids and meta["version_id"] not in filter_version_ids:
            continue
        vf = meta["vector_file"]
        file_groups.setdefault(vf, []).append(meta)

    results: List[Dict[str, Any]] = []
    for vector_file, metas_in_file in file_groups.items():
        try:
            corpus = np.load(vector_file)
        except Exception as exc:
            logger.warning("Cannot load %s: %s", vector_file, exc)
            continue

        indices = [m["vector_index"] for m in metas_in_file]
        # Guard against out-of-bounds
        valid = [(m, i) for m, i in zip(metas_in_file, indices) if i < len(corpus)]
        if not valid:
            continue

        valid_metas, valid_indices = zip(*valid)
        subset = corpus[list(valid_indices)]
        sims = cosine_similarity_batch(query_vec, subset)

        for meta, sim in zip(valid_metas, sims):
            if float(sim) >= threshold:
                results.append({
                    "similarity": float(sim),
                    "source_text": meta["source_text"],
                    "version_id": meta["version_id"],
                    "chunk_id": meta.get("chunk_id"),
                    "statement_id": meta.get("statement_id"),
                    "source_type": meta["source_type"],
                })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]
