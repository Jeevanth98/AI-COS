"""
AI-COS — Knowledge Graph Builder and Persistence
Uses NetworkX; persists to JSON (node-link format) for reliable cross-session reload.
Distinguishes user_supplied / extracted / semantic_inferred relationship types.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import networkx as nx

from config.settings import GRAPH_FILE, GRAPH_SEMANTIC_EDGE_THRESHOLD

logger = logging.getLogger(__name__)

_GRAPH: Optional[nx.DiGraph] = None   # module-level singleton


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _graph_path() -> Path:
    return Path(GRAPH_FILE)


def save_graph(g: nx.DiGraph) -> None:
    """Persist the graph to JSON (node-link format)."""
    data = nx.node_link_data(g, edges="links")
    _graph_path().write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Graph saved: %d nodes, %d edges → %s",
                g.number_of_nodes(), g.number_of_edges(), _graph_path().name)


def load_graph() -> nx.DiGraph:
    """Load the persisted graph from disk. Creates an empty graph if not found."""
    global _GRAPH
    path = _graph_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            g = nx.node_link_graph(data, directed=True, multigraph=False, edges="links")
            logger.info("Graph loaded: %d nodes, %d edges ← %s",
                        g.number_of_nodes(), g.number_of_edges(), path.name)
            _GRAPH = g
            return g
        except Exception as exc:
            logger.warning("Could not load graph from %s: %s. Starting fresh.", path, exc)

    _GRAPH = nx.DiGraph()
    return _GRAPH


def get_graph() -> nx.DiGraph:
    """Return the in-memory graph, loading from disk if necessary."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = load_graph()
    return _GRAPH


# ── Node helpers ───────────────────────────────────────────────────────────────

_NODE_TYPES = {
    "document", "version", "policy_statement",
    "department", "concept", "requirement",
}


def add_document_node(g: nx.DiGraph, document_id: str, **attrs) -> None:
    g.add_node(document_id, node_type="document", **attrs)


def add_version_node(g: nx.DiGraph, version_id: str, document_id: str, **attrs) -> None:
    g.add_node(version_id, node_type="version", document_id=document_id, **attrs)
    # Version → Document edge
    if not g.has_edge(version_id, document_id):
        g.add_edge(
            version_id, document_id,
            relationship="VERSION_OF",
            inference_method="structural",
            evidence="System: version belongs to document.",
            verification_status="verified",
        )


def add_department_node(g: nx.DiGraph, department: str) -> None:
    dept_id = f"DEPT:{department}"
    if not g.has_node(dept_id):
        g.add_node(dept_id, node_type="department", label=department)


def add_policy_statement_node(g: nx.DiGraph, statement_id: str,
                               version_id: str, modality: str,
                               text_snippet: str) -> None:
    g.add_node(
        statement_id,
        node_type="policy_statement",
        version_id=version_id,
        modality=modality,
        text_snippet=text_snippet[:200],
    )
    if not g.has_edge(statement_id, version_id):
        g.add_edge(
            statement_id, version_id,
            relationship="BELONGS_TO",
            inference_method="structural",
            evidence="System: statement extracted from version.",
            verification_status="verified",
        )


# ── Edge helpers ───────────────────────────────────────────────────────────────

def add_edge(
    g: nx.DiGraph,
    source: str,
    target: str,
    relationship: str,
    evidence: str = "",
    inference_method: str = "user_supplied",
    confidence: Optional[float] = None,
    verification_status: str = "unverified",
) -> None:
    """Add or update a directed edge. Does NOT add duplicate edges."""
    if g.has_edge(source, target):
        # Update attributes if edge already exists
        existing = g.edges[source, target]
        if existing.get("relationship") == relationship:
            return
    g.add_edge(
        source, target,
        relationship=relationship,
        evidence=evidence,
        inference_method=inference_method,
        confidence=confidence,
        verification_status=verification_status,
    )


# ── Graph construction from DB data ───────────────────────────────────────────

def build_document_subgraph(
    g: nx.DiGraph,
    document_id: str,
    document_attrs: Dict[str, Any],
    version_id: str,
    version_attrs: Dict[str, Any],
) -> Tuple[int, int]:
    """
    Add/update document and version nodes for a newly ingested version.
    Returns (nodes_added, edges_added).
    """
    nodes_before = g.number_of_nodes()
    edges_before = g.number_of_edges()

    add_document_node(g, document_id, **document_attrs)
    add_version_node(g, version_id, document_id, **version_attrs)

    # Department node + edge
    dept = document_attrs.get("department")
    if dept:
        dept_id = f"DEPT:{dept}"
        add_department_node(g, dept)
        add_edge(
            g, document_id, dept_id,
            relationship="BELONGS_TO_DEPARTMENT",
            inference_method="structural",
            evidence=f"Document registered in department: {dept}",
            verification_status="verified",
        )

    return (g.number_of_nodes() - nodes_before,
            g.number_of_edges() - edges_before)


def add_dataset_relationships(g: nx.DiGraph, relationships: List[Dict[str, Any]]) -> int:
    """
    Bulk-load relationships from the dataset JSON (relationships.json).
    Maps source/target by doc_series_code.
    Returns number of edges added.
    """
    added = 0
    for rel in relationships:
        src = rel.get("source")
        tgt = rel.get("target")
        rtype = rel.get("relationship", "RELATED_TO")
        if not src or not tgt:
            continue
        # Node IDs in graph use document_id; dataset uses series codes like "HR-001"
        # We'll use the series codes as node identifiers if real doc nodes don't exist yet
        src_id = _find_node_by_series(g, src) or src
        tgt_id = _find_node_by_series(g, tgt) or tgt
        add_edge(
            g, src_id, tgt_id,
            relationship=rtype,
            inference_method="user_supplied",
            evidence=f"Dataset relationships.json: {src} → {tgt}",
            verification_status="verified",
        )
        added += 1
    return added


def _find_node_by_series(g: nx.DiGraph, series_code: str) -> Optional[str]:
    """Find a graph node whose 'doc_series_code' attribute matches."""
    for node_id, attrs in g.nodes(data=True):
        if attrs.get("doc_series_code") == series_code:
            return node_id
    return None


# ── Graph query helpers ────────────────────────────────────────────────────────

def get_neighbors(g: nx.DiGraph, node_id: str, depth: int = 1) -> Dict[str, Any]:
    """Return neighbors within *depth* hops."""
    if node_id not in g:
        return {"nodes": [], "edges": []}
    nodes = set()
    edges = []
    frontier = {node_id}
    for _ in range(depth):
        next_frontier = set()
        for n in frontier:
            for succ in g.successors(n):
                if succ not in nodes:
                    next_frontier.add(succ)
                    edge_data = g.edges[n, succ]
                    edges.append({"source": n, "target": succ, **edge_data})
            for pred in g.predecessors(n):
                if pred not in nodes:
                    next_frontier.add(pred)
                    edge_data = g.edges[pred, n]
                    edges.append({"source": pred, "target": n, **edge_data})
        nodes |= frontier
        frontier = next_frontier - nodes
    nodes |= frontier
    node_list = [{"id": n, **g.nodes[n]} for n in nodes if n != node_id]
    return {"nodes": node_list, "edges": edges}


def get_graph_summary(g: nx.DiGraph) -> Dict[str, Any]:
    node_types: Dict[str, int] = {}
    for _, attrs in g.nodes(data=True):
        nt = attrs.get("node_type", "unknown")
        node_types[nt] = node_types.get(nt, 0) + 1

    rel_types: Dict[str, int] = {}
    for _, _, attrs in g.edges(data=True):
        rt = attrs.get("relationship", "unknown")
        rel_types[rt] = rel_types.get(rt, 0) + 1

    return {
        "total_nodes": g.number_of_nodes(),
        "total_edges": g.number_of_edges(),
        "node_types": node_types,
        "relationship_types": rel_types,
    }
