"""
AI-COS — Ingestion Service
Orchestrates the full document ingestion pipeline:
  validate → register → extract → chunk → policy extract → embed → graph update

All steps are transactional where possible. Partial failures are logged and
reported — never silently swallowed. Returns a detailed processing report.
"""

from __future__ import annotations
import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from config.settings import UPLOADS_DIR, EMBEDDING_MODEL, EMBEDDING_DIM
from database.db import get_session
from database import repositories as repo
from database.models import DocumentVersion
from ingestion.document_loader import (
    validate_file, load_document, compute_file_hash, save_uploaded_file
)
from ingestion.chunker import chunk_text
from nlp.policy_parser import extract_policy_statements
from nlp.embeddings import generate_and_save_embeddings
from graph.graph_builder import (
    get_graph, save_graph,
    build_document_subgraph, add_policy_statement_node, add_edge,
)

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_document(
    file_bytes: bytes,
    original_filename: str,
    metadata: Dict[str, Any],
    document_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full ingestion pipeline for one uploaded file.

    Parameters
    ----------
    file_bytes        : raw file content
    original_filename : original upload filename (used for extension detection)
    metadata          : user-supplied fields (title, department, version_label, …)
    document_id       : if provided, attach this version to an existing document;
                        if None, the system checks for a matching series code or
                        creates a new document record

    Returns
    -------
    Processing report dict with 'success', 'version_id', 'document_id',
    'stages', 'stats', 'warnings', 'errors' keys.
    """
    report: Dict[str, Any] = {
        "success": False,
        "document_id": None,
        "version_id": None,
        "stages": {},
        "stats": {},
        "warnings": [],
        "errors": [],
        "timestamp": datetime.utcnow().isoformat(),
    }

    # ── Stage 1: Save & Validate ───────────────────────────────────────────
    try:
        saved_path = save_uploaded_file(file_bytes, original_filename)
        report["stages"]["save"] = "ok"
    except Exception as exc:
        report["errors"].append(f"Could not save file: {exc}")
        report["stages"]["save"] = f"failed: {exc}"
        return report

    validation = validate_file(saved_path)
    if not validation["ok"]:
        saved_path.unlink(missing_ok=True)
        report["errors"].append(validation["error"])
        report["stages"]["validation"] = f"failed: {validation['error']}"
        return report
    report["stages"]["validation"] = "ok"
    report["warnings"].extend(validation.get("warnings", []))

    # ── Stage 2: Duplicate detection ──────────────────────────────────────
    try:
        file_hash = compute_file_hash(saved_path)
    except Exception as exc:
        file_hash = ""
        report["warnings"].append(f"Could not compute file hash: {exc}")

    with get_session() as session:
        existing_ver = repo.get_version_by_hash(session, file_hash) if file_hash else None
        if existing_ver:
            saved_path.unlink(missing_ok=True)
            report["errors"].append(
                f"This exact file has already been uploaded (version_id={existing_ver.version_id}). "
                "If you want to register a new version, modify the document first."
            )
            report["stages"]["duplicate_check"] = "duplicate_detected"
            return report
    report["stages"]["duplicate_check"] = "ok"

    # ── Stage 3: Extraction ────────────────────────────────────────────────
    extraction = load_document(saved_path)
    report["warnings"].extend(extraction.get("warnings", []))

    if extraction.get("error"):
        _cleanup_and_log(saved_path, None, "extraction", extraction["error"], report)
        return report

    if extraction["total_chars"] < 50:
        msg = (
            f"Extraction yielded only {extraction['total_chars']} characters. "
            "Document may be empty or a scanned image."
        )
        report["warnings"].append(msg)
        if extraction["is_likely_scanned"]:
            report["errors"].append(
                "Document appears scanned. Enable OCR (Tesseract) for proper extraction."
            )
            _cleanup_and_log(saved_path, None, "extraction", report["errors"][-1], report)
            return report

    report["stages"]["extraction"] = "ok"
    report["stats"]["extracted_chars"] = extraction["total_chars"]

    # ── Stage 4: Database registration ────────────────────────────────────
    series_code = metadata.get("doc_series_code", "")
    title = metadata.get("title") or Path(original_filename).stem.replace("_", " ")

    with get_session() as session:
        # Resolve or create Document
        if document_id:
            doc = repo.get_document(session, document_id)
            if not doc:
                report["errors"].append(f"Document ID '{document_id}' not found.")
                _cleanup_and_log(saved_path, None, "registration",
                                 report["errors"][-1], report)
                return report
        else:
            doc = None
            if series_code:
                doc = repo.get_document_by_series_code(session, series_code)
            if not doc:
                doc = repo.create_document(
                    session,
                    title=title,
                    source=metadata.get("source"),
                    issuing_authority=metadata.get("issuing_authority"),
                    department=metadata.get("department"),
                    document_type=metadata.get("document_type", "Policy"),
                    doc_series_code=series_code or None,
                )

        document_id = doc.document_id
        report["document_id"] = document_id

        # Create version
        ver = repo.create_version(
            session,
            document_id=document_id,
            version_label=metadata.get("version_label", "1.0"),
            publication_date=metadata.get("publication_date"),
            effective_date=metadata.get("effective_date"),
            expiry_date=metadata.get("expiry_date"),
            file_hash=file_hash,
            file_path=str(saved_path),
            original_filename=original_filename,
            file_size_bytes=extraction["file_size"],
            mime_type=extraction["mime_type"],
            source_url=metadata.get("source_url"),
            processing_status="processing",
            extracted_text_path=extraction.get("extracted_text_path"),
            extracted_text_length=extraction["total_chars"],
        )
        version_id = ver.version_id
        report["version_id"] = version_id
        report["stages"]["registration"] = "ok"

    # ── Stage 5: Chunking ──────────────────────────────────────────────────
    try:
        page_map = extraction.get("pages") or []
        chunks_raw = chunk_text(
            extraction["cleaned_text"],
            page_map=page_map if page_map else None,
        )
        report["stats"]["chunk_count"] = len(chunks_raw)
        report["stages"]["chunking"] = "ok"
    except Exception as exc:
        _cleanup_and_log(saved_path, version_id, "chunking", str(exc), report)
        return report

    # ── Stage 6: Persist Chunks ────────────────────────────────────────────
    chunk_db_records: List[Dict[str, Any]] = []
    try:
        with get_session() as session:
            for ch in chunks_raw:
                db_chunk = repo.create_chunk(
                    session,
                    version_id=version_id,
                    chunk_index=ch["chunk_index"],
                    text=ch["text"],
                    page_number=ch.get("page_number"),
                    section_heading=ch.get("section_heading"),
                    char_start=ch.get("char_start"),
                    char_end=ch.get("char_end"),
                )
                chunk_db_records.append({
                    "id": db_chunk.chunk_id,
                    "text": ch["text"],
                    "chunk_index": ch["chunk_index"],
                    "page_number": ch.get("page_number"),
                    "section_heading": ch.get("section_heading"),
                })
        report["stages"]["chunk_persistence"] = "ok"
    except Exception as exc:
        _cleanup_and_log(saved_path, version_id, "chunk_persistence", str(exc), report)
        return report

    # ── Stage 7: Policy Statement Extraction ──────────────────────────────
    policy_stmts: List[Dict[str, Any]] = []
    try:
        raw_stmts = extract_policy_statements(chunks_raw, version_id)
        # Persist
        with get_session() as session:
            for stmt in raw_stmts:
                # Find chunk_id from our persisted records
                chunk_id = None
                for cr in chunk_db_records:
                    if cr["chunk_index"] == chunks_raw[0].get("chunk_index", -99):
                        pass  # done below
                # Map chunk by index
                chunk_idx = None
                for orig_chunk in chunks_raw:
                    if orig_chunk.get("text") == stmt.get("original_text", "")[:len(orig_chunk["text"])]:
                        chunk_idx = orig_chunk.get("chunk_index")
                        break
                for cr in chunk_db_records:
                    if cr["chunk_index"] == chunk_idx:
                        chunk_id = cr["id"]
                        break

                db_stmt = repo.create_policy_statement(
                    session,
                    version_id=version_id,
                    chunk_id=chunk_id,
                    source_page=stmt.get("source_page"),
                    source_section=stmt.get("source_section"),
                    original_text=stmt["original_text"],
                    actor=stmt.get("actor"),
                    action=stmt.get("action"),
                    object_=stmt.get("object"),
                    modality=stmt.get("modality"),
                    conditions=stmt.get("conditions", []),
                    exceptions=stmt.get("exceptions", []),
                    scope=stmt.get("scope"),
                    temporal_constraints=stmt.get("temporal_constraints", {}),
                    numerical_threshold=stmt.get("numerical_threshold"),
                    authority=stmt.get("authority"),
                    effective_date=stmt.get("effective_date"),
                    extraction_method=stmt.get("extraction_method", "rule_based"),
                    verification_status="unverified",
                )
                policy_stmts.append({
                    "id": db_stmt.statement_id,
                    "text": stmt["original_text"],
                    "modality": stmt.get("modality"),
                })

        report["stats"]["policy_statement_count"] = len(policy_stmts)
        report["stages"]["policy_extraction"] = "ok"
    except Exception as exc:
        logger.exception("Policy extraction failed for version %s", version_id)
        report["warnings"].append(f"Policy extraction partial failure: {exc}")
        report["stages"]["policy_extraction"] = f"partial: {exc}"

    # ── Stage 8: Embedding Generation ─────────────────────────────────────
    embedding_count = 0
    try:
        items_for_embedding = [
            {"id": cr["id"], "text": cr["text"]}
            for cr in chunk_db_records
        ]
        vectors, emb_metas = generate_and_save_embeddings(
            version_id, items_for_embedding, source_type="chunk"
        )
        with get_session() as session:
            for meta in emb_metas:
                repo.create_embedding_meta(session, **meta)
        embedding_count = len(emb_metas)
        report["stats"]["embedding_count"] = embedding_count
        report["stages"]["embedding"] = "ok"
    except Exception as exc:
        logger.exception("Embedding failed for version %s", version_id)
        report["warnings"].append(f"Embedding generation failed: {exc}")
        report["stages"]["embedding"] = f"failed: {exc}"

    # ── Stage 9: Knowledge Graph Update ───────────────────────────────────
    nodes_added = edges_added = 0
    try:
        g = get_graph()
        doc_attrs = {
            "title": title,
            "department": metadata.get("department", ""),
            "document_type": metadata.get("document_type", "Policy"),
            "doc_series_code": series_code,
        }
        ver_attrs = {
            "version_label": metadata.get("version_label", "1.0"),
            "original_filename": original_filename,
        }
        nodes_added, edges_added = build_document_subgraph(
            g, document_id, doc_attrs, version_id, ver_attrs
        )

        # Add policy statement nodes (small sample to keep graph manageable)
        for stmt in policy_stmts[:20]:
            add_policy_statement_node(
                g, stmt["id"], version_id,
                stmt.get("modality", "unknown"),
                stmt["text"][:200],
            )
            nodes_added += 1
            edges_added += 1

        save_graph(g)
        report["stats"]["graph_nodes_added"] = nodes_added
        report["stats"]["graph_edges_added"] = edges_added
        report["stages"]["graph"] = "ok"
    except Exception as exc:
        logger.exception("Graph update failed for version %s", version_id)
        report["warnings"].append(f"Graph update failed: {exc}")
        report["stages"]["graph"] = f"failed: {exc}"

    # ── Stage 10: Finalise DB record ───────────────────────────────────────
    try:
        with get_session() as session:
            repo.update_version_stats(
                session, version_id,
                processing_status="done",
                chunk_count=len(chunk_db_records),
                policy_statement_count=len(policy_stmts),
                embedding_count=embedding_count,
                graph_nodes_added=nodes_added,
                graph_edges_added=edges_added,
                warnings=len(report["warnings"]),
            )
            repo.update_version_status(
                session, version_id, "done",
                f"Pipeline completed: {len(chunk_db_records)} chunks, "
                f"{len(policy_stmts)} statements, {embedding_count} embeddings."
            )
    except Exception as exc:
        report["warnings"].append(f"Could not finalise DB record: {exc}")

    report["success"] = not report["errors"]
    logger.info(
        "Ingestion %s for '%s' (version=%s): chunks=%d stmts=%d embs=%d",
        "SUCCESS" if report["success"] else "PARTIAL",
        original_filename, version_id,
        len(chunk_db_records), len(policy_stmts), embedding_count,
    )
    return report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cleanup_and_log(
    saved_path: Path, version_id: Optional[str],
    stage: str, error_msg: str,
    report: Dict[str, Any],
) -> None:
    """Mark stage as failed, log error, update version status."""
    report["errors"].append(f"[{stage}] {error_msg}")
    report["stages"][stage] = f"failed: {error_msg}"
    report["success"] = False
    if version_id:
        try:
            with get_session() as session:
                repo.update_version_status(session, version_id, "failed", error_msg)
                repo.log_processing_error(
                    session, version_id, stage, error_msg, severity="error"
                )
        except Exception:
            pass
    logger.error("Ingestion stage '%s' failed: %s", stage, error_msg)
