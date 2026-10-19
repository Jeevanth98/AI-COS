"""
AI-COS — Dataset Seeding Utility
Batch ingests all synthetic documents from Dataset/AI-COS_Dataset/documents/
and populates the Knowledge Graph with Dataset/AI-COS_Dataset/relationships.json.
"""

from pathlib import Path
import json
import logging
import sys

# Ensure root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.db import init_db, get_session
from database import repositories as repo
from services.ingestion_service import ingest_document
from graph.graph_builder import get_graph, save_graph, add_dataset_relationships, load_graph
from config.settings import DATASET_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed_dataset")


def seed_dataset(dry_run: bool = False):
    init_db()
    docs_dir = DATASET_DIR / "documents"
    meta_dir = DATASET_DIR / "metadata"
    rel_file = DATASET_DIR / "relationships.json"

    if not docs_dir.exists():
        logger.error("Documents directory not found at: %s", docs_dir)
        return

    doc_files = sorted(list(docs_dir.glob("*.docx")) + list(docs_dir.glob("*.pdf")) + list(docs_dir.glob("*.txt")))
    logger.info("Found %d documents to ingest in %s", len(doc_files), docs_dir)

    success_count = 0
    duplicate_count = 0
    failed_count = 0

    for doc_path in doc_files:
        code_guess = doc_path.stem.split("_")[0]
        meta_json_path = meta_dir / f"{code_guess}.json"
        
        # Load associated metadata JSON if present
        meta_from_file = {}
        if meta_json_path.exists():
            try:
                meta_from_file = json.loads(meta_json_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Could not read metadata JSON %s: %e", meta_json_path.name, e)

        title = meta_from_file.get("title", doc_path.stem.replace("_", " "))
        department = meta_from_file.get("department", "General")
        doc_type = meta_from_file.get("document_type", "Policy")
        version_label = meta_from_file.get("version", "1.0")
        series_code = meta_from_file.get("document_id", code_guess)

        metadata = {
            "title": title,
            "department": department,
            "document_type": doc_type,
            "doc_series_code": series_code,
            "version_label": version_label,
            "issuing_authority": department,
        }

        if dry_run:
            logger.info("[Dry Run] Would ingest: %s (%s)", title, doc_path.name)
            continue

        file_bytes = doc_path.read_bytes()
        report = ingest_document(file_bytes, doc_path.name, metadata)

        if report["success"]:
            success_count += 1
            logger.info("✅ Ingested: %s (version: %s, chunks: %d, statements: %d)",
                        title, report["version_id"][:8],
                        report["stats"].get("chunk_count", 0),
                        report["stats"].get("policy_statement_count", 0))
        elif any("already uploaded" in e.lower() for e in report.get("errors", [])):
            duplicate_count += 1
            logger.info("ℹ️ Skipped duplicate: %s", title)
        else:
            failed_count += 1
            logger.error("❌ Failed to ingest %s: %s", title, report.get("errors"))

    # Load dataset relationships into graph
    if rel_file.exists() and not dry_run:
        try:
            rel_data = json.loads(rel_file.read_text(encoding="utf-8"))
            g = load_graph()
            added_edges = add_dataset_relationships(g, rel_data)
            save_graph(g)
            logger.info("🕸️ Loaded %d relationships into Knowledge Graph from %s", added_edges, rel_file.name)
        except Exception as exc:
            logger.error("Could not load relationships: %s", exc)

    logger.info("--- Ingestion Complete ---")
    logger.info("Successfully ingested: %d", success_count)
    logger.info("Duplicates skipped: %d", duplicate_count)
    logger.info("Failed: %d", failed_count)


if __name__ == "__main__":
    seed_dataset()
