"""
AI-COS — Document Service
Query-focused service layer used by the dashboard pages.
Keeps SQL and graph logic out of the UI code.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from database.db import get_session
from database import repositories as repo
from database.models import Document, DocumentVersion, Chunk, PolicyStatement
from graph.graph_builder import get_graph, get_graph_summary, get_neighbors, save_graph
from nlp.embeddings import semantic_search
from config.settings import SEMANTIC_TOP_K, SEMANTIC_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


def list_all_documents() -> List[Dict[str, Any]]:
    """Return all documents with their version counts."""
    with get_session() as session:
        docs = repo.list_documents(session)
        result = []
        for doc in docs:
            versions = repo.list_versions(session, doc.document_id)
            result.append({
                "document_id": doc.document_id,
                "title": doc.title,
                "department": doc.department,
                "document_type": doc.document_type,
                "issuing_authority": doc.issuing_authority,
                "doc_series_code": doc.doc_series_code,
                "version_count": len(versions),
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "latest_status": versions[0].processing_status if versions else "none",
            })
        return result


def get_document_detail(document_id: str) -> Optional[Dict[str, Any]]:
    with get_session() as session:
        doc = repo.get_document(session, document_id)
        if not doc:
            return None
        versions = repo.list_versions(session, document_id)
        return {
            "document_id": doc.document_id,
            "title": doc.title,
            "department": doc.department,
            "document_type": doc.document_type,
            "issuing_authority": doc.issuing_authority,
            "doc_series_code": doc.doc_series_code,
            "source": doc.source,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "versions": [_version_summary(v) for v in versions],
        }


def _version_summary(ver: DocumentVersion) -> Dict[str, Any]:
    return {
        "version_id": ver.version_id,
        "version_label": ver.version_label,
        "original_filename": ver.original_filename,
        "file_size_bytes": ver.file_size_bytes,
        "upload_timestamp": ver.upload_timestamp.isoformat() if ver.upload_timestamp else None,
        "processing_status": ver.processing_status,
        "extracted_text_length": ver.extracted_text_length,
        "chunk_count": ver.chunk_count,
        "policy_statement_count": ver.policy_statement_count,
        "embedding_count": ver.embedding_count,
        "graph_nodes_added": ver.graph_nodes_added,
        "graph_edges_added": ver.graph_edges_added,
        "warnings": ver.warnings,
        "effective_date": ver.effective_date,
        "publication_date": ver.publication_date,
    }


def get_version_detail(version_id: str) -> Optional[Dict[str, Any]]:
    with get_session() as session:
        ver = repo.get_version(session, version_id)
        if not ver:
            return None
        detail = _version_summary(ver)
        detail["processing_log"] = json.loads(ver.processing_log or "[]")
        detail["extracted_text_path"] = ver.extracted_text_path
        return detail


def get_extracted_text(version_id: str) -> str:
    """Read the cleaned text file for a version. Returns empty string on error."""
    with get_session() as session:
        ver = repo.get_version(session, version_id)
        if not ver or not ver.extracted_text_path:
            return ""
    path = Path(ver.extracted_text_path)
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def get_chunks_for_version(version_id: str) -> List[Dict[str, Any]]:
    with get_session() as session:
        chunks = repo.get_chunks(session, version_id)
        return [
            {
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "page_number": c.page_number,
                "section_heading": c.section_heading,
            }
            for c in chunks
        ]


def get_policy_statements_for_version(version_id: str) -> List[Dict[str, Any]]:
    with get_session() as session:
        stmts = repo.get_policy_statements(session, version_id)
        return [
            {
                "statement_id": s.statement_id,
                "original_text": s.original_text,
                "modality": s.modality,
                "actor": s.actor,
                "conditions": s.conditions or [],
                "exceptions": s.exceptions or [],
                "temporal_constraints": s.temporal_constraints or {},
                "numerical_threshold": s.numerical_threshold,
                "source_page": s.source_page,
                "source_section": s.source_section,
                "extraction_method": s.extraction_method,
                "verification_status": s.verification_status,
            }
            for s in stmts
        ]


def update_policy_statement(statement_id: str, updates: Dict[str, Any]) -> bool:
    """Allow dashboard users to correct extracted policy statements."""
    with get_session() as session:
        stmt = repo.update_policy_statement(session, statement_id, **updates)
        return stmt is not None


def update_document_metadata(document_id: str, updates: Dict[str, Any]) -> bool:
    with get_session() as session:
        doc = repo.update_document(session, document_id, **updates)
        return doc is not None


def get_system_stats() -> Dict[str, Any]:
    with get_session() as session:
        stats = repo.get_system_stats(session)
    g = get_graph()
    graph_summary = get_graph_summary(g)
    stats["graph_nodes"] = graph_summary["total_nodes"]
    stats["graph_edges"] = graph_summary["total_edges"]
    stats["graph_node_types"] = graph_summary["node_types"]
    stats["graph_rel_types"] = graph_summary["relationship_types"]
    return stats


def run_semantic_search(
    query: str,
    top_k: int = SEMANTIC_TOP_K,
    threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    filter_version_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search all stored embeddings for passages semantically related to *query*.
    Enriches results with document title and version label.
    """
    with get_session() as session:
        all_metas_orm = repo.get_all_embedding_metas(session)
        all_metas = [
            {
                "version_id": em.version_id,
                "chunk_id": em.chunk_id,
                "statement_id": em.statement_id,
                "source_type": em.source_type,
                "source_text": em.source_text,
                "model_name": em.model_name,
                "vector_file": em.vector_file,
                "vector_index": em.vector_index,
            }
            for em in all_metas_orm
        ]
        # Build version → doc title map
        version_map: Dict[str, Dict[str, str]] = {}
        all_versions = repo.list_all_versions(session)
        for ver in all_versions:
            doc = repo.get_document(session, ver.document_id)
            version_map[ver.version_id] = {
                "title": doc.title if doc else "Unknown",
                "version_label": ver.version_label or "—",
                "doc_series_code": (doc.doc_series_code if doc else "") or "",
            }

    results = semantic_search(
        query=query,
        all_metas=all_metas,
        top_k=top_k,
        threshold=threshold,
        filter_version_ids=filter_version_ids,
    )

    # Enrich with document info
    for r in results:
        vm = version_map.get(r["version_id"], {})
        r["document_title"] = vm.get("title", "Unknown")
        r["version_label"] = vm.get("version_label", "—")
        r["doc_series_code"] = vm.get("doc_series_code", "")

    return results


def delete_document_cascade(document_id: str) -> Dict[str, Any]:
    """
    Delete a document and all derived artifacts (versions, chunks, embeddings).
    Removes associated .npy files and graph nodes.
    Returns a report of what was deleted.
    """
    from nlp.embeddings import delete_embeddings
    from graph.graph_builder import get_graph, save_graph

    report: Dict[str, Any] = {"deleted": [], "warnings": []}

    with get_session() as session:
        doc = repo.get_document(session, document_id)
        if not doc:
            return {"error": f"Document {document_id} not found."}

        versions = repo.list_versions(session, document_id)
        for ver in versions:
            # Delete embedding files
            if delete_embeddings(ver.version_id):
                report["deleted"].append(f"Embeddings for version {ver.version_id}")

        if repo.delete_document(session, document_id):
            report["deleted"].append(f"Document {document_id} and {len(versions)} versions")

    # Remove from graph
    try:
        g = get_graph()
        if document_id in g:
            g.remove_node(document_id)
            save_graph(g)
            report["deleted"].append(f"Graph node {document_id}")
    except Exception as exc:
        report["warnings"].append(f"Graph cleanup warning: {exc}")

    return report
