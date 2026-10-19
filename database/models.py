"""
AI-COS Database Models (SQLAlchemy ORM)
Full schema for Phase 1 persistent storage.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Text, Boolean,
    DateTime, ForeignKey, Index, JSON
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────────────────────────────────────

class Document(Base):
    """Represents a unique logical document (not a version)."""
    __tablename__ = "documents"

    document_id     = Column(String, primary_key=True, default=_uuid)
    title           = Column(String, nullable=False)
    source          = Column(String, nullable=True)          # URL / filename origin
    issuing_authority = Column(String, nullable=True)
    department      = Column(String, nullable=True)
    document_type   = Column(String, nullable=True)          # Policy / Regulation / Manual
    doc_series_code = Column(String, nullable=True, index=True)  # e.g. "HR-001"
    created_at      = Column(DateTime, default=_now)
    updated_at      = Column(DateTime, default=_now, onupdate=_now)

    versions = relationship("DocumentVersion", back_populates="document",
                            cascade="all, delete-orphan")


class DocumentVersion(Base):
    """One uploaded file = one version of a document."""
    __tablename__ = "document_versions"

    version_id       = Column(String, primary_key=True, default=_uuid)
    document_id      = Column(String, ForeignKey("documents.document_id"), nullable=False)
    version_label    = Column(String, nullable=True)          # "1.0", "2.0", "Draft-A"
    publication_date = Column(String, nullable=True)          # ISO date string, may be unknown
    effective_date   = Column(String, nullable=True)
    expiry_date      = Column(String, nullable=True)
    file_hash        = Column(String, nullable=True, index=True)
    file_path        = Column(String, nullable=False)         # path under data/uploads/
    original_filename = Column(String, nullable=False)
    file_size_bytes  = Column(Integer, nullable=True)
    mime_type        = Column(String, nullable=True)
    source_url       = Column(String, nullable=True)
    upload_timestamp = Column(DateTime, default=_now)
    processing_status = Column(String, default="pending")    # pending / processing / done / failed
    processing_log   = Column(Text, nullable=True)           # JSON-encoded list of log entries
    extracted_text_path = Column(String, nullable=True)      # path to .txt file under data/processed/
    extracted_text_length = Column(Integer, nullable=True)
    chunk_count      = Column(Integer, nullable=True)
    policy_statement_count = Column(Integer, nullable=True)
    embedding_count  = Column(Integer, nullable=True)
    graph_nodes_added = Column(Integer, nullable=True)
    graph_edges_added = Column(Integer, nullable=True)
    warnings         = Column(Integer, default=0)

    document  = relationship("Document", back_populates="versions")
    chunks    = relationship("Chunk", back_populates="version", cascade="all, delete-orphan")
    policy_statements = relationship("PolicyStatement", back_populates="version",
                                     cascade="all, delete-orphan")
    embeddings = relationship("EmbeddingMetadata", back_populates="version",
                              cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_docver_document_id", "document_id"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chunks
# ─────────────────────────────────────────────────────────────────────────────

class Chunk(Base):
    """A text chunk extracted from a document version."""
    __tablename__ = "chunks"

    chunk_id        = Column(String, primary_key=True, default=_uuid)
    version_id      = Column(String, ForeignKey("document_versions.version_id"), nullable=False)
    chunk_index     = Column(Integer, nullable=False)
    text            = Column(Text, nullable=False)
    page_number     = Column(Integer, nullable=True)
    section_heading = Column(String, nullable=True)
    char_start      = Column(Integer, nullable=True)
    char_end        = Column(Integer, nullable=True)
    chunk_type      = Column(String, default="paragraph")    # paragraph / table / heading

    version    = relationship("DocumentVersion", back_populates="chunks")
    embeddings = relationship("EmbeddingMetadata", back_populates="chunk",
                              cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_chunk_version_id", "version_id"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Policy Statements
# ─────────────────────────────────────────────────────────────────────────────

class PolicyStatement(Base):
    """A structured policy statement extracted from a chunk."""
    __tablename__ = "policy_statements"

    statement_id       = Column(String, primary_key=True, default=_uuid)
    version_id         = Column(String, ForeignKey("document_versions.version_id"), nullable=False)
    chunk_id           = Column(String, ForeignKey("chunks.chunk_id"), nullable=True)
    source_page        = Column(Integer, nullable=True)
    source_section     = Column(String, nullable=True)
    original_text      = Column(Text, nullable=False)
    actor              = Column(String, nullable=True)
    action             = Column(String, nullable=True)
    object_            = Column("object", String, nullable=True)
    modality           = Column(String, nullable=True)       # permission / obligation / prohibition / recommendation / factual
    conditions         = Column(JSON, nullable=True)         # list of strings
    exceptions         = Column(JSON, nullable=True)         # list of strings
    scope              = Column(String, nullable=True)
    temporal_constraints = Column(JSON, nullable=True)
    numerical_threshold = Column(String, nullable=True)
    authority          = Column(String, nullable=True)
    effective_date     = Column(String, nullable=True)
    extraction_method  = Column(String, default="rule_based")
    verification_status = Column(String, default="unverified")  # unverified / verified / rejected
    created_at         = Column(DateTime, default=_now)

    version = relationship("DocumentVersion", back_populates="policy_statements")
    chunk   = relationship("Chunk")

    __table_args__ = (
        Index("ix_polstmt_version_id", "version_id"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingMetadata(Base):
    """Metadata linking a stored numpy vector to its source."""
    __tablename__ = "embedding_metadata"

    embedding_id    = Column(String, primary_key=True, default=_uuid)
    version_id      = Column(String, ForeignKey("document_versions.version_id"), nullable=False)
    chunk_id        = Column(String, ForeignKey("chunks.chunk_id"), nullable=True)
    statement_id    = Column(String, ForeignKey("policy_statements.statement_id"), nullable=True)
    source_type     = Column(String, nullable=False)         # "chunk" or "statement"
    source_text     = Column(Text, nullable=False)           # the exact text that was embedded
    model_name      = Column(String, nullable=False)
    model_version   = Column(String, nullable=True)
    vector_file     = Column(String, nullable=False)         # .npy file path
    vector_index    = Column(Integer, nullable=False)        # row index within the .npy file
    embedding_dim   = Column(Integer, nullable=False)
    created_at      = Column(DateTime, default=_now)

    version = relationship("DocumentVersion", back_populates="embeddings")
    chunk   = relationship("Chunk")

    __table_args__ = (
        Index("ix_emb_version_id", "version_id"),
        Index("ix_emb_chunk_id", "chunk_id"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Graph Relationships (relational mirror of NetworkX edges)
# ─────────────────────────────────────────────────────────────────────────────

class GraphRelationship(Base):
    """Persistent record of every knowledge-graph edge."""
    __tablename__ = "graph_relationships"

    rel_id             = Column(String, primary_key=True, default=_uuid)
    source_node        = Column(String, nullable=False, index=True)
    target_node        = Column(String, nullable=False, index=True)
    relationship_type  = Column(String, nullable=False)
    evidence           = Column(Text, nullable=True)
    inference_method   = Column(String, default="user_supplied")
    # user_supplied / extracted / semantic_inferred / rule_inferred
    verification_status = Column(String, default="unverified")
    confidence         = Column(Float, nullable=True)
    created_at         = Column(DateTime, default=_now)


# ─────────────────────────────────────────────────────────────────────────────
# Processing Errors
# ─────────────────────────────────────────────────────────────────────────────

class ProcessingError(Base):
    """Any error or warning raised during document processing."""
    __tablename__ = "processing_errors"

    error_id    = Column(String, primary_key=True, default=_uuid)
    version_id  = Column(String, ForeignKey("document_versions.version_id"), nullable=True)
    stage       = Column(String, nullable=False)   # extraction / chunking / embedding / graph
    severity    = Column(String, default="error")  # error / warning / info
    message     = Column(Text, nullable=False)
    detail      = Column(Text, nullable=True)
    created_at  = Column(DateTime, default=_now)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Results (placeholder — fully used in Phase 2+)
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisResult(Base):
    """Records a completed analysis run (SIPA / ASCD / comparison)."""
    __tablename__ = "analysis_results"

    analysis_id      = Column(String, primary_key=True, default=_uuid)
    analysis_type    = Column(String, nullable=False)   # sipa / ascd / comparison
    input_versions   = Column(JSON, nullable=True)      # list of version_ids
    configuration    = Column(JSON, nullable=True)
    status           = Column(String, default="pending")
    result_summary   = Column(JSON, nullable=True)
    created_at       = Column(DateTime, default=_now)
    completed_at     = Column(DateTime, nullable=True)


class Finding(Base):
    """A single finding within an analysis (impact / conflict / recommendation)."""
    __tablename__ = "findings"

    finding_id      = Column(String, primary_key=True, default=_uuid)
    analysis_id     = Column(String, ForeignKey("analysis_results.analysis_id"), nullable=False)
    finding_type    = Column(String, nullable=False)
    evidence        = Column(JSON, nullable=True)
    explanation     = Column(Text, nullable=True)
    priority        = Column(String, nullable=True)
    verification_status = Column(String, default="unverified")
    reviewer_decision   = Column(String, nullable=True)
    reviewer_note       = Column(Text, nullable=True)
    reviewed_at         = Column(DateTime, nullable=True)
    created_at          = Column(DateTime, default=_now)
