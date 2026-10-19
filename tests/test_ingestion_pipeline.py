"""
AI-COS Tests — End-to-End Ingestion Pipeline
Tests ingestion from file upload bytes through to SQLite, embeddings, and graph.
"""

import uuid
from pathlib import Path
import pytest

from database.db import init_db
from services.ingestion_service import ingest_document
from services.document_service import (
    list_all_documents,
    get_document_detail,
    get_chunks_for_version,
    get_policy_statements_for_version,
    delete_document_cascade,
)


@pytest.fixture(autouse=True)
def setup_database():
    init_db()


def test_ingest_txt_document():
    unique_run_id = str(uuid.uuid4())
    raw_text = (
        f"1. Policy Scope ({unique_run_id})\n"
        "This policy applies to all full-time contractors and employees.\n\n"
        "2. Mandatory Attendance\n"
        "Employees must log working hours before 10:00 AM each business day.\n\n"
        "3. Working from Home\n"
        "Staff may request up to two remote days per week subject to supervisor approval."
    )
    file_bytes = raw_text.encode("utf-8")
    metadata = {
        "title": f"Working Hours Policy {unique_run_id[:8]}",
        "department": "Human Resources",
        "document_type": "Policy",
        "doc_series_code": f"HR-TEST-{unique_run_id[:6]}",
        "version_label": "1.0",
        "issuing_authority": "HR Director"
    }

    report = ingest_document(file_bytes, f"HR_Test_{unique_run_id[:6]}.txt", metadata)
    assert report["success"] is True
    assert report["document_id"] is not None
    assert report["version_id"] is not None
    assert report["stats"]["chunk_count"] >= 1
    assert report["stats"]["policy_statement_count"] >= 1

    # Verify via Document Service
    doc = get_document_detail(report["document_id"])
    assert doc is not None
    assert doc["title"] == metadata["title"]
    assert len(doc["versions"]) == 1

    chunks = get_chunks_for_version(report["version_id"])
    assert len(chunks) == report["stats"]["chunk_count"]

    statements = get_policy_statements_for_version(report["version_id"])
    assert len(statements) == report["stats"]["policy_statement_count"]
    modalities = {s["modality"] for s in statements}
    assert "obligation" in modalities or "permission" in modalities

    # Cleanup
    delete_document_cascade(report["document_id"])


def test_duplicate_file_rejection():
    unique_run_id = str(uuid.uuid4())
    file_bytes = f"Sample text for duplicate test checking ({unique_run_id}). Exactly the same bytes.".encode("utf-8")
    metadata = {"title": f"Duplicate Test Doc {unique_run_id[:6]}", "version_label": "1.0"}

    # First upload
    rep1 = ingest_document(file_bytes, f"first_{unique_run_id[:6]}.txt", metadata)
    assert rep1["success"] is True

    # Second upload with same content
    rep2 = ingest_document(file_bytes, f"second_{unique_run_id[:6]}.txt", metadata)
    assert rep2["success"] is False
    assert any("already uploaded" in e.lower() or "exact file" in e.lower() for e in rep2["errors"])

    # Cleanup
    if rep1.get("document_id"):
        delete_document_cascade(rep1["document_id"])
