"""
AI-COS Tests — Restart & Persistence Resilience
Simulates restarting the application to ensure SQLite database,
vector embeddings on disk, and NetworkX knowledge graphs retain complete fidelity.
"""

from pathlib import Path
import numpy as np
import pytest

from database.db import get_session, init_db
from database import repositories as repo
from graph.graph_builder import save_graph, load_graph, add_document_node, add_edge
import graph.graph_builder as gb
import nlp.embeddings as emb_mod
from nlp.embeddings import save_version_embeddings, load_version_embeddings
from services.document_service import list_all_documents, get_document_detail


@pytest.fixture(autouse=True)
def setup_database():
    init_db()


def test_cross_session_database_persistence():
    test_title = "Cross-Session Persistence Test Document"
    doc_id = None
    with get_session() as session:
        doc = repo.create_document(session, title=test_title, department="Engineering")
        doc_id = doc.document_id

    # Re-query in a completely new session (simulating restart)
    with get_session() as session:
        retrieved = repo.get_document(session, doc_id)
        assert retrieved is not None
        assert retrieved.title == test_title
        assert retrieved.department == "Engineering"

    # Clean up test doc
    with get_session() as session:
        repo.delete_document(session, doc_id)


def test_cross_session_graph_persistence(tmp_path, monkeypatch):
    test_graph_file = tmp_path / "restart_test_graph.json"
    monkeypatch.setattr(gb, "_graph_path", lambda: test_graph_file)

    # 1. Build and save graph
    g1 = gb.nx.DiGraph()
    add_document_node(g1, "node-101", title="Safety Standard", code="SAFE-01")
    add_document_node(g1, "node-102", title="Emergency Manual", code="EMERG-01")
    add_edge(g1, "node-101", "node-102", "INFLUENCES")
    save_graph(g1)

    # 2. Clear global in-memory state (simulating complete app shutdown)
    gb._GRAPH = None

    # 3. Reload graph from file
    g2 = load_graph()
    assert g2.number_of_nodes() == 2
    assert g2.number_of_edges() == 1
    assert g2.has_edge("node-101", "node-102")
    assert g2["node-101"]["node-102"]["relationship"] == "INFLUENCES"


def test_cross_session_embeddings_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(emb_mod, "EMBEDDINGS_DIR", tmp_path)

    v_id = "v_restart_test"
    vecs = np.ones((3, 384), dtype=np.float32)

    saved_path = save_version_embeddings(v_id, vecs)
    assert (tmp_path / f"{v_id}.npy").exists()

    # Load afresh
    loaded = load_version_embeddings(v_id)
    assert loaded is not None
    assert loaded.shape == (3, 384)
    np.testing.assert_array_equal(loaded, vecs)
