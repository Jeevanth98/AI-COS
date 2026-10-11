import json
from pathlib import Path

import numpy as np

from phase1_ai_cos_pipeline import (
    build_document_registry,
    build_knowledge_graph,
    build_structured_rules,
    extract_document_text,
    find_dataset_root,
    generate_document_embeddings,
    validate_dataset,
)


DATASET_ROOT = find_dataset_root(Path.cwd())
processed = DATASET_ROOT / "processed"
processed.mkdir(exist_ok=True)


def test_dataset_loading_and_validation():
    report = validate_dataset(DATASET_ROOT)
    assert report["summary"]["total_documents_found"] == 17
    assert report["summary"]["total_metadata_records"] == 17
    assert report["summary"]["total_relationships"] == 27
    assert report["summary"]["duplicate_metadata_ids"] == []
    assert report["errors"] == []


def test_docx_extraction():
    path = DATASET_ROOT / "documents" / "HR_001_Remote_Work_Policy.docx"
    text = extract_document_text(path)
    assert "REMOTE WORK POLICY" in text
    assert "Document ID: HR-001" in text


def test_json_loading_and_validation():
    rel_path = DATASET_ROOT / "relationships.json"
    relationships = json.loads(rel_path.read_text(encoding="utf-8"))
    assert isinstance(relationships, list)
    assert len(relationships) >= 1

    scenario_path = DATASET_ROOT / "scenarios" / "SIPA-001.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    assert scenario["scenario_id"] == "SIPA-001"


def test_structured_rules_output():
    report = validate_dataset(DATASET_ROOT)
    registry = build_document_registry(DATASET_ROOT, report)
    rules = build_structured_rules(registry)
    assert len(rules) > 0
    first = rules[0]
    assert "document_id" in first
    assert "source_text" in first
    assert "extraction_method" in first


def test_embedding_generation_and_mapping():
    report = validate_dataset(DATASET_ROOT)
    registry = build_document_registry(DATASET_ROOT, report)
    info = generate_document_embeddings(registry, processed)
    assert info["embedding_count"] == len(registry)
    save_path = processed / "document_embeddings.npy"
    vectors = np.load(save_path)
    assert vectors.shape[0] == len(registry)
    assert np.isfinite(vectors).all()
    index_path = processed / "embedding_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(index) == len(registry)


def test_knowledge_graph_node_creation_and_relationships():
    report = validate_dataset(DATASET_ROOT)
    registry = build_document_registry(DATASET_ROOT, report)
    graph = build_knowledge_graph(DATASET_ROOT, registry, report)
    assert graph["node_count"] == 17
    assert graph["edge_count"] == 27
    assert graph["directed"] is True


def test_invalid_relationship_detection():
    report = validate_dataset(DATASET_ROOT)
    invalid = report["summary"].get("invalid_relationship_endpoints", [])
    assert invalid == []


if __name__ == "__main__":
    test_dataset_loading_and_validation()
    test_docx_extraction()
    test_json_loading_and_validation()
    test_structured_rules_output()
    test_embedding_generation_and_mapping()
    test_knowledge_graph_node_creation_and_relationships()
    test_invalid_relationship_detection()
    print("All Phase 1 tests passed.")
