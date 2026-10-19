"""
AI-COS Tests — Database Layer
Tests schema initialization, CRUD operations, relationships, and cascades.
"""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import (
    Base, Document, DocumentVersion, Chunk, PolicyStatement,
    EmbeddingMetadata, GraphRelationship, ProcessingError
)
from database import repositories as repo


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_create_and_get_document(db_session):
    doc = repo.create_document(
        db_session,
        title="Remote Work Policy",
        department="Human Resources",
        document_type="Policy",
        doc_series_code="HR-001"
    )
    db_session.commit()

    retrieved = repo.get_document(db_session, doc.document_id)
    assert retrieved is not None
    assert retrieved.title == "Remote Work Policy"
    assert retrieved.department == "Human Resources"
    assert retrieved.doc_series_code == "HR-001"


def test_document_versioning_and_cascade(db_session):
    doc = repo.create_document(db_session, title="Security Policy", doc_series_code="IT-001")
    db_session.commit()

    # Add Version 1.0
    v1 = repo.create_version(
        db_session,
        document_id=doc.document_id,
        version_label="1.0",
        original_filename="IT-001_v1.docx",
        file_path="data/uploads/IT-001_v1.docx",
        file_hash="hash123",
        file_size_bytes=1024,
        processing_status="done"
    )
    
    # Add Chunk
    c1 = repo.create_chunk(
        db_session,
        version_id=v1.version_id,
        chunk_index=0,
        text="All employees must use strong passwords.",
        char_start=0,
        char_end=41
    )

    # Add Policy Statement
    s1 = repo.create_policy_statement(
        db_session,
        version_id=v1.version_id,
        chunk_id=c1.chunk_id,
        original_text="All employees must use strong passwords.",
        modality="obligation",
        actor="employees",
        action="use strong passwords"
    )
    db_session.commit()

    versions = repo.list_versions(db_session, doc.document_id)
    assert len(versions) == 1
    assert versions[0].version_label == "1.0"

    chunks = repo.get_chunks(db_session, v1.version_id)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0

    stmts = repo.get_policy_statements(db_session, v1.version_id)
    assert len(stmts) == 1
    assert stmts[0].modality == "obligation"

    # Test Cascade Delete
    deleted = repo.delete_document(db_session, doc.document_id)
    assert deleted is True
    db_session.commit()

    assert repo.get_document(db_session, doc.document_id) is None
    assert repo.get_version(db_session, v1.version_id) is None
    assert len(repo.get_chunks(db_session, v1.version_id)) == 0
    assert len(repo.get_policy_statements(db_session, v1.version_id)) == 0


def test_duplicate_file_hash_detection(db_session):
    doc = repo.create_document(db_session, title="Expense Policy")
    db_session.commit()

    repo.create_version(
        db_session,
        document_id=doc.document_id,
        version_label="1.0",
        original_filename="FIN-001.docx",
        file_path="data/uploads/FIN-001.docx",
        file_hash="sha256_identical_hash",
        file_size_bytes=2048
    )
    db_session.commit()

    existing = repo.get_version_by_hash(db_session, "sha256_identical_hash")
    assert existing is not None
    assert existing.original_filename == "FIN-001.docx"

    non_existing = repo.get_version_by_hash(db_session, "non_existent_hash")
    assert non_existing is None


def test_processing_error_recording(db_session):
    doc = repo.create_document(db_session, title="Corrupted Doc")
    db_session.commit()
    v = repo.create_version(
        db_session,
        document_id=doc.document_id,
        version_label="1.0",
        original_filename="corrupt.pdf",
        file_path="data/uploads/corrupt.pdf",
        file_hash="hash_err",
        file_size_bytes=500
    )
    db_session.commit()

    err = repo.log_processing_error(
        db_session,
        version_id=v.version_id,
        stage="pdf_extraction",
        message="Document is password protected",
    )
    db_session.commit()

    errors = repo.get_errors_for_version(db_session, v.version_id)
    assert len(errors) == 1
    assert errors[0].stage == "pdf_extraction"
    assert "password protected" in errors[0].message
