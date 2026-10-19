"""
AI-COS Tests — Knowledge Graph
Tests NetworkX DiGraph creation, edge typing, JSON persistence, and reload.
"""

from pathlib import Path
import networkx as nx
import pytest

from graph.graph_builder import (
    add_document_node,
    add_version_node,
    add_policy_statement_node,
    add_edge,
    save_graph,
    load_graph,
    get_neighbors,
    get_graph_summary,
    add_dataset_relationships,
)


def test_graph_node_and_edge_creation():
    g = nx.DiGraph()

    add_document_node(g, "doc-1", title="Remote Work Policy", code="HR-001", dept="HR")
    add_document_node(g, "doc-2", title="Attendance Policy", code="HR-002", dept="HR")
    add_edge(g, "doc-1", "doc-2", "AFFECTS", inference_method="user_supplied")

    assert g.has_node("doc-1")
    assert g.has_node("doc-2")
    assert g.has_edge("doc-1", "doc-2")
    assert g["doc-1"]["doc-2"]["relationship"] == "AFFECTS"
    assert g["doc-1"]["doc-2"]["inference_method"] == "user_supplied"


def test_graph_persistence_and_reload(tmp_path, monkeypatch):
    test_graph_file = tmp_path / "test_knowledge_graph.json"
    import graph.graph_builder as gb
    monkeypatch.setattr(gb, "_graph_path", lambda: test_graph_file)

    g = nx.DiGraph()
    add_document_node(g, "doc-fin", title="Travel Policy", code="FIN-005")
    add_document_node(g, "doc-hr", title="Expense Policy", code="FIN-001")
    add_edge(g, "doc-fin", "doc-hr", "REFERENCES")

    save_graph(g)
    assert test_graph_file.exists()

    # Clear memory and reload
    gb._GRAPH = None
    loaded_g = load_graph()
    assert loaded_g.number_of_nodes() == 2
    assert loaded_g.number_of_edges() == 1
    assert loaded_g.has_edge("doc-fin", "doc-hr")
    assert loaded_g["doc-fin"]["doc-hr"]["relationship"] == "REFERENCES"


def test_graph_summary_and_neighbors():
    g = nx.DiGraph()
    add_document_node(g, "node-a", title="A", code="A-01")
    add_document_node(g, "node-b", title="B", code="B-01")
    add_document_node(g, "node-c", title="C", code="C-01")
    add_edge(g, "node-a", "node-b", "AFFECTS")
    add_edge(g, "node-c", "node-a", "CONTROLS")

    summary = get_graph_summary(g)
    assert summary["total_nodes"] == 3
    assert summary["total_edges"] == 2

    neighbors = get_neighbors(g, "node-a")
    assert len(neighbors["nodes"]) == 2
    assert len(neighbors["edges"]) == 2
