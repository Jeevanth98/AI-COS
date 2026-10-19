"""
AI-COS Tests — Document Ingestion and Extraction
Tests text cleaning, document loaders, chunking, and file validation.
"""

from pathlib import Path
import pytest

from ingestion.text_cleaner import clean_text, normalize_whitespace
from ingestion.document_loader import validate_file, compute_file_hash, load_document
from ingestion.chunker import chunk_text
from ingestion.docx_extractor import extract_docx
from config.settings import DATASET_DIR


def test_clean_text_normalisation():
    raw = "Policy   Document\r\n\r\n\r\n\r\nSection 1:\tGeneral\u00a0Rules\n\fPage 1"
    cleaned = clean_text(raw)
    assert "\u00a0" not in cleaned
    assert "\f" not in cleaned
    assert "Section 1: General Rules" in cleaned


def test_normalize_whitespace():
    text = "   This   is   a   \n\n\t test.   "
    assert normalize_whitespace(text) == "This is a test."


def test_chunk_text():
    sample_text = (
        "1.0 Purpose and Scope\n"
        "This policy governs the access control procedures for all organizational systems.\n\n"
        "2.0 Password Requirements\n"
        "All employees must use passwords with at least 12 characters, including numbers and symbols. "
        "Passwords must be updated every 90 days without exception.\n\n"
        "3.0 Multi-Factor Authentication\n"
        "MFA is mandatory for all remote access connections to corporate infrastructure."
    )
    chunks = chunk_text(sample_text, chunk_size=150, chunk_overlap=30)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert "chunk_index" in chunk
        assert "text" in chunk
        assert "char_start" in chunk
        assert "char_end" in chunk
        assert chunk["char_end"] > chunk["char_start"]


def test_docx_extractor_with_real_dataset():
    doc_path = DATASET_DIR / "documents" / "HR_001_Remote_Work_Policy.docx"
    if not doc_path.exists():
        pytest.skip(f"Dataset file not found at {doc_path}")

    result = extract_docx(doc_path)
    assert result["error"] is None
    assert len(result["full_text"]) > 100
    assert len(result["paragraphs"]) > 0


def test_validate_file_checks(tmp_path):
    # Non-existent file
    res = validate_file(tmp_path / "does_not_exist.docx")
    assert not res["ok"]
    assert "not exist" in res["error"].lower()

    # Empty file
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    res = validate_file(empty_file)
    assert not res["ok"]
    assert "empty" in res["error"].lower()

    # Disallowed extension
    bad_ext = tmp_path / "script.exe"
    bad_ext.write_bytes(b"content")
    res = validate_file(bad_ext)
    assert not res["ok"]
    assert "unsupported" in res["error"].lower()
