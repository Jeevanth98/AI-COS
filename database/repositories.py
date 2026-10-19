"""
AI-COS Database Repositories
Query helpers for each major entity. Each function takes an explicit Session
so callers control transaction boundaries via get_session().
"""

from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session

from database.models import (
    Document, DocumentVersion, Chunk, PolicyStatement,
    EmbeddingMetadata, GraphRelationship, ProcessingError, AnalysisResult, Finding,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Document CRUD
# ─────────────────────────────────────────────────────────────────────────────

def create_document(session: Session, **kwargs) -> Document:
    doc = Document(**kwargs)
    session.add(doc)
    session.flush()
    return doc


def get_document(session: Session, document_id: str) -> Optional[Document]:
    return session.get(Document, document_id)


def get_document_by_series_code(session: Session, code: str) -> Optional[Document]:
    return session.query(Document).filter_by(doc_series_code=code).first()


def list_documents(session: Session) -> List[Document]:
    return session.query(Document).order_by(Document.created_at.desc()).all()


def update_document(session: Session, document_id: str, **kwargs) -> Optional[Document]:
    doc = session.get(Document, document_id)
    if doc:
        for k, v in kwargs.items():
            setattr(doc, k, v)
        doc.updated_at = datetime.utcnow()
        session.flush()
    return doc


def delete_document(session: Session, document_id: str) -> bool:
    doc = session.get(Document, document_id)
    if doc:
        session.delete(doc)
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# DocumentVersion CRUD
# ─────────────────────────────────────────────────────────────────────────────

def create_version(session: Session, **kwargs) -> DocumentVersion:
    ver = DocumentVersion(**kwargs)
    session.add(ver)
    session.flush()
    return ver


def get_version(session: Session, version_id: str) -> Optional[DocumentVersion]:
    return session.get(DocumentVersion, version_id)


def get_version_by_hash(session: Session, file_hash: str) -> Optional[DocumentVersion]:
    return session.query(DocumentVersion).filter_by(file_hash=file_hash).first()


def list_versions(session: Session, document_id: str) -> List[DocumentVersion]:
    return (
        session.query(DocumentVersion)
        .filter_by(document_id=document_id)
        .order_by(DocumentVersion.upload_timestamp.desc())
        .all()
    )


def list_all_versions(session: Session) -> List[DocumentVersion]:
    return session.query(DocumentVersion).order_by(DocumentVersion.upload_timestamp.desc()).all()


def update_version_status(session: Session, version_id: str, status: str, log_entry: str = "") -> None:
    ver = session.get(DocumentVersion, version_id)
    if ver:
        ver.processing_status = status
        # Append to JSON log
        try:
            log = json.loads(ver.processing_log or "[]")
        except (json.JSONDecodeError, TypeError):
            log = []
        if log_entry:
            log.append({"ts": datetime.utcnow().isoformat(), "msg": log_entry})
        ver.processing_log = json.dumps(log)
        session.flush()


def update_version_stats(session: Session, version_id: str, **kwargs) -> None:
    ver = session.get(DocumentVersion, version_id)
    if ver:
        for k, v in kwargs.items():
            setattr(ver, k, v)
        session.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Chunk CRUD
# ─────────────────────────────────────────────────────────────────────────────

def create_chunk(session: Session, **kwargs) -> Chunk:
    c = Chunk(**kwargs)
    session.add(c)
    session.flush()
    return c


def get_chunks(session: Session, version_id: str) -> List[Chunk]:
    return (
        session.query(Chunk)
        .filter_by(version_id=version_id)
        .order_by(Chunk.chunk_index)
        .all()
    )


def get_chunk(session: Session, chunk_id: str) -> Optional[Chunk]:
    return session.get(Chunk, chunk_id)


# ─────────────────────────────────────────────────────────────────────────────
# Policy Statement CRUD
# ─────────────────────────────────────────────────────────────────────────────

def create_policy_statement(session: Session, **kwargs) -> PolicyStatement:
    ps = PolicyStatement(**kwargs)
    session.add(ps)
    session.flush()
    return ps


def get_policy_statements(session: Session, version_id: str) -> List[PolicyStatement]:
    return (
        session.query(PolicyStatement)
        .filter_by(version_id=version_id)
        .all()
    )


def get_all_policy_statements(session: Session) -> List[PolicyStatement]:
    return session.query(PolicyStatement).all()


def update_policy_statement(session: Session, statement_id: str, **kwargs) -> Optional[PolicyStatement]:
    ps = session.get(PolicyStatement, statement_id)
    if ps:
        for k, v in kwargs.items():
            setattr(ps, k, v)
        session.flush()
    return ps


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Metadata CRUD
# ─────────────────────────────────────────────────────────────────────────────

def create_embedding_meta(session: Session, **kwargs) -> EmbeddingMetadata:
    em = EmbeddingMetadata(**kwargs)
    session.add(em)
    session.flush()
    return em


def get_embedding_metas(session: Session, version_id: str) -> List[EmbeddingMetadata]:
    return session.query(EmbeddingMetadata).filter_by(version_id=version_id).all()


def get_all_embedding_metas(session: Session) -> List[EmbeddingMetadata]:
    return session.query(EmbeddingMetadata).all()


def delete_embeddings_for_version(session: Session, version_id: str) -> int:
    deleted = (
        session.query(EmbeddingMetadata)
        .filter_by(version_id=version_id)
        .delete()
    )
    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Graph Relationship CRUD
# ─────────────────────────────────────────────────────────────────────────────

def create_relationship(session: Session, **kwargs) -> GraphRelationship:
    rel = GraphRelationship(**kwargs)
    session.add(rel)
    session.flush()
    return rel


def list_relationships(session: Session) -> List[GraphRelationship]:
    return session.query(GraphRelationship).all()


def get_relationships_for_node(session: Session, node_id: str) -> List[GraphRelationship]:
    return (
        session.query(GraphRelationship)
        .filter(
            (GraphRelationship.source_node == node_id) |
            (GraphRelationship.target_node == node_id)
        )
        .all()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Processing Errors
# ─────────────────────────────────────────────────────────────────────────────

def log_processing_error(session: Session, version_id: Optional[str],
                          stage: str, message: str,
                          severity: str = "error", detail: str = "") -> ProcessingError:
    err = ProcessingError(
        version_id=version_id,
        stage=stage,
        severity=severity,
        message=message,
        detail=detail,
    )
    session.add(err)
    session.flush()
    return err


def get_errors_for_version(session: Session, version_id: str) -> List[ProcessingError]:
    return session.query(ProcessingError).filter_by(version_id=version_id).all()


record_error = log_processing_error


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_system_stats(session: Session) -> Dict[str, Any]:
    """Return overview counts used on the dashboard Overview page."""
    return {
        "total_documents": session.query(Document).count(),
        "total_versions": session.query(DocumentVersion).count(),
        "total_chunks": session.query(Chunk).count(),
        "total_policy_statements": session.query(PolicyStatement).count(),
        "total_embeddings": session.query(EmbeddingMetadata).count(),
        "total_graph_relationships": session.query(GraphRelationship).count(),
        "pending_versions": session.query(DocumentVersion).filter_by(processing_status="pending").count(),
        "failed_versions": session.query(DocumentVersion).filter_by(processing_status="failed").count(),
        "done_versions": session.query(DocumentVersion).filter_by(processing_status="done").count(),
    }
