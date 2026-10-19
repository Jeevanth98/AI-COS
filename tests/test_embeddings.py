"""
AI-COS Tests — Embeddings & Semantic Search
Tests vector generation, .npy storage, and cosine similarity search.
"""

from pathlib import Path
import numpy as np
import pytest

import nlp.embeddings as emb_mod
from nlp.embeddings import (
    embed_texts,
    save_version_embeddings,
    load_version_embeddings,
    cosine_similarity_matrix,
    semantic_search,
)
from config.settings import EMBEDDING_DIM


def test_embed_texts():
    texts = [
        "Remote work policy guidelines and eligibility.",
        "Corporate password complexity requirements."
    ]
    vecs = embed_texts(texts)
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (2, EMBEDDING_DIM)
    assert vecs.dtype == np.float32

    # Verify L2 normalized
    norms = np.linalg.norm(vecs, axis=1)
    for n in norms:
        assert np.isclose(n, 1.0, atol=1e-3)


def test_cosine_similarity():
    v = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    sim = cosine_similarity_matrix(v, v)
    assert np.isclose(sim[0, 0], 1.0, atol=1e-4)


def test_save_and_load_version_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr(emb_mod, "EMBEDDINGS_DIR", tmp_path)

    v_id = "test_version_uuid"
    sample_vecs = np.random.randn(5, EMBEDDING_DIM).astype(np.float32)
    sample_vecs /= np.linalg.norm(sample_vecs, axis=1, keepdims=True)

    saved_path = save_version_embeddings(v_id, sample_vecs)
    assert (tmp_path / f"{v_id}.npy").exists()

    loaded = load_version_embeddings(v_id)
    assert loaded is not None
    assert loaded.shape == (5, EMBEDDING_DIM)
    np.testing.assert_allclose(loaded, sample_vecs, rtol=1e-5)
