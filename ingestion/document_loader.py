"""
AI-COS — Document Loader
Top-level entry point for loading and extracting any supported file type.
Routes to the appropriate extractor and performs quality checks.
"""

from __future__ import annotations
import hashlib
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any

from config.settings import (
    UPLOADS_DIR, PROCESSED_DIR, MAX_UPLOAD_SIZE_BYTES, ALLOWED_EXTENSIONS
)
from ingestion.pdf_extractor import extract_pdf
from ingestion.docx_extractor import extract_docx
from ingestion.text_cleaner import clean_text, remove_headers_footers_heuristic

logger = logging.getLogger(__name__)

_MIN_CHARS_FOR_USEFUL_CONTENT = 50


def compute_file_hash(file_path: Path) -> str:
    """SHA-256 hash of file bytes."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def validate_file(file_path: Path) -> Dict[str, Any]:
    """
    Validate a file before ingestion.

    Returns {'ok': bool, 'error': str | None, 'warnings': list[str]}
    """
    result = {"ok": True, "error": None, "warnings": []}

    if not file_path.exists():
        result["ok"] = False
        result["error"] = "File does not exist."
        return result

    if not file_path.is_file():
        result["ok"] = False
        result["error"] = "Path is not a file."
        return result

    suffix = file_path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        result["ok"] = False
        result["error"] = (
            f"Unsupported file type '{suffix}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
        return result

    size = file_path.stat().st_size
    if size == 0:
        result["ok"] = False
        result["error"] = "File is empty (0 bytes)."
        return result

    if size > MAX_UPLOAD_SIZE_BYTES:
        result["ok"] = False
        result["error"] = (
            f"File size {size / 1024 / 1024:.1f} MB exceeds the "
            f"configured limit of {MAX_UPLOAD_SIZE_BYTES // 1024 // 1024} MB."
        )
        return result

    return result


def save_uploaded_file(source_bytes: bytes, original_filename: str) -> Path:
    """
    Save an uploaded file to the uploads directory with a stable, safe name.
    Returns the saved path.
    """
    suffix = Path(original_filename).suffix.lower()
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    dest = UPLOADS_DIR / safe_name
    dest.write_bytes(source_bytes)
    return dest


def load_document(file_path: Path) -> Dict[str, Any]:
    """
    Load and extract text from a file.

    Returns a structured result suitable for the ingestion service:
    {
        full_text     : str,
        cleaned_text  : str,
        pages         : list (PDF only),
        paragraphs    : list (DOCX only),
        tables        : list (DOCX only),
        total_chars   : int,
        file_hash     : str,
        file_size     : int,
        mime_type     : str,
        is_likely_scanned: bool,
        warnings      : list[str],
        error         : str | None,
    }
    """
    suffix = file_path.suffix.lower()
    result: Dict[str, Any] = {
        "full_text": "",
        "cleaned_text": "",
        "pages": [],
        "paragraphs": [],
        "tables": [],
        "total_chars": 0,
        "file_hash": "",
        "file_size": 0,
        "mime_type": "",
        "is_likely_scanned": False,
        "warnings": [],
        "error": None,
    }

    # ── File hash ──────────────────────────────────────────────────────────
    try:
        result["file_hash"] = compute_file_hash(file_path)
        result["file_size"] = file_path.stat().st_size
    except Exception as exc:
        result["error"] = f"Cannot read file: {exc}"
        return result

    # ── Extraction ─────────────────────────────────────────────────────────
    if suffix == ".pdf":
        result["mime_type"] = "application/pdf"
        extracted = extract_pdf(file_path)
        result.update({
            "full_text": extracted["full_text"],
            "pages": extracted["pages"],
            "is_likely_scanned": extracted["is_likely_scanned"],
            "warnings": extracted["warnings"],
            "error": extracted["error"],
        })

    elif suffix == ".docx":
        result["mime_type"] = (
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        )
        extracted = extract_docx(file_path)
        result.update({
            "full_text": extracted["full_text"],
            "paragraphs": extracted["paragraphs"],
            "tables": extracted["tables"],
            "warnings": extracted["warnings"],
            "error": extracted["error"],
        })

    elif suffix == ".txt":
        result["mime_type"] = "text/plain"
        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
            result["full_text"] = raw
        except Exception as exc:
            result["error"] = f"Cannot read TXT file: {exc}"
            return result

    else:
        result["error"] = f"Unsupported extension: {suffix}"
        return result

    if result["error"]:
        return result

    # ── Quality checks ─────────────────────────────────────────────────────
    cleaned = clean_text(result["full_text"])
    cleaned = remove_headers_footers_heuristic(cleaned)
    result["cleaned_text"] = cleaned
    result["total_chars"] = len(cleaned.strip())

    if result["total_chars"] < _MIN_CHARS_FOR_USEFUL_CONTENT:
        result["warnings"].append(
            f"Extracted text is very short ({result['total_chars']} chars). "
            "The document may be empty, image-only, or extraction may have failed."
        )

    # ── Save cleaned text ──────────────────────────────────────────────────
    stem = file_path.stem
    txt_path = PROCESSED_DIR / f"{stem}_cleaned.txt"
    try:
        txt_path.write_text(cleaned, encoding="utf-8")
        result["extracted_text_path"] = str(txt_path)
    except Exception as exc:
        result["warnings"].append(f"Could not save cleaned text: {exc}")
        result["extracted_text_path"] = None

    logger.info("Loaded %s: %d chars  warnings=%d  error=%s",
                file_path.name, result["total_chars"],
                len(result["warnings"]), result["error"])
    return result
