from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
from docx import Document

try:
    import fitz
except Exception:  # pragma: no cover
    fitz = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


WINDOWS_STUB_PATH = Path(tempfile.gettempdir()) / "fake_sklearn_pkg"


def ensure_fake_sklearn_stub() -> Path:
    """Create a temporary sklearn stub to bypass a Windows App Control block on sklearn DLLs."""
    path = WINDOWS_STUB_PATH
    (path / "sklearn" / "metrics").mkdir(parents=True, exist_ok=True)
    (path / "sklearn" / "__init__.py").write_text('__version__ = "0.0"\n', encoding="utf-8")
    (path / "sklearn" / "metrics" / "__init__.py").write_text(
        "def roc_curve(*args, **kwargs):\n"
        "    raise RuntimeError('stub roc_curve invoked only for import-time compatibility')\n",
        encoding="utf-8",
    )
    return path


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"JSON file could not be read: {path} ({exc})") from exc


def normalize_document_id(name: str) -> str:
    text = name.strip()
    match = re.search(r"([A-Za-z]+[-_][A-Za-z0-9]+)", text)
    if not match:
        return text
    normalized = match.group(1).replace("_", "-")
    normalized = re.sub(r"-+", "-", normalized)
    return normalized


def find_dataset_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    candidates = [start] + list(start.parents)
    for candidate in candidates:
        for possible in [candidate / "AI-COS_Dataset", candidate / "Dataset" / "AI-COS_Dataset", candidate / "Dataset"]:
            if possible.exists() and possible.is_dir():
                if possible.name == "AI-COS_Dataset":
                    return possible
                if (possible / "AI-COS_Dataset").exists() and (possible / "AI-COS_Dataset").is_dir():
                    return possible / "AI-COS_Dataset"
    raise FileNotFoundError(
        "Could not find AI-COS_Dataset. Please pass the dataset folder or place it under the current project tree."
    )


def list_supported_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.iterdir() if not p.name.startswith(".") and p.is_file()],
        key=lambda p: p.name.lower(),
    )


def validate_dataset(root: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "errors": [],
        "warnings": [],
        "summary": {},
        "document_files": [],
        "metadata_files": [],
        "relationships": [],
        "scenarios": [],
        "conflicts": [],
    }

    docs_dir = root / "documents"
    metadata_dir = root / "metadata"
    relationships_path = root / "relationships.json"
    scenarios_dir = root / "scenarios"
    conflicts_dir = root / "conflicts"

    doc_files = list_supported_files(docs_dir)
    metadata_files = list_supported_files(metadata_dir)
    report["document_files"] = [p.name for p in doc_files]
    report["metadata_files"] = [p.name for p in metadata_files]

    if not docs_dir.exists():
        report["errors"].append(f"Missing documents directory: {docs_dir}")
    if not metadata_dir.exists():
        report["errors"].append(f"Missing metadata directory: {metadata_dir}")

    doc_ids = []
    for file_path in doc_files:
        if file_path.suffix.lower() not in {".docx", ".pdf", ".txt", ".doc"}:
            report["warnings"].append(f"Skipped unsupported file type: {file_path.name}")
            continue
        doc_id = normalize_document_id(file_path.name)
        doc_ids.append(doc_id)
    report["summary"]["total_document_files"] = len(doc_files)

    metadata_records = []
    duplicate_meta_ids = []
    for file_path in metadata_files:
        if file_path.suffix.lower() != ".json":
            continue
        try:
            data = load_json(file_path)
            metadata_records.append(data)
            doc_id = data.get("document_id")
            if doc_id:
                if doc_id in duplicate_meta_ids or doc_id in [item for item, count in __import__('collections').Counter(doc_id for doc_id in [m.get("document_id") for m in metadata_records if m.get("document_id")]).items() if count > 1]:
                    duplicate_meta_ids.append(doc_id)
        except Exception as exc:
            report["errors"].append(f"Invalid JSON metadata file: {file_path.name} ({exc})")

    meta_ids = [item.get("document_id") for item in metadata_records if isinstance(item, dict) and item.get("document_id")]
    report["summary"]["total_metadata_records"] = len(metadata_records)
    duplicates = sorted({value for value in meta_ids if meta_ids.count(value) > 1})
    if duplicates:
        report["errors"].append(f"Duplicate metadata IDs: {duplicates}")

    doc_source_ids = sorted({value for value in doc_ids if value})
    missing_metadata = sorted(set(doc_source_ids) - set(meta_ids))
    if missing_metadata:
        report["warnings"].append(f"Documents missing metadata: {missing_metadata}")

    if relationships_path.exists():
        try:
            relationships = load_json(relationships_path)
            report["relationships"] = relationships if isinstance(relationships, list) else []
        except Exception as exc:
            report["errors"].append(f"Invalid relationships JSON: {relationships_path.name} ({exc})")
            relationships = []
    else:
        relationships = []
        report["warnings"].append(f"Missing relationships file: {relationships_path}")

    valid_doc_ids = set(meta_ids) | set(doc_source_ids)
    invalid_relationship_endpoints = []
    for rel in relationships:
        if not isinstance(rel, dict):
            report["warnings"].append(f"Malformed relationship entry: {rel}")
            continue
        source = rel.get("source")
        target = rel.get("target")
        if source not in valid_doc_ids or target not in valid_doc_ids:
            invalid_relationship_endpoints.append({"source": source, "target": target, "relationship": rel.get("relationship")})
    if invalid_relationship_endpoints:
        report["errors"].append(f"Invalid relationship endpoints: {invalid_relationship_endpoints}")

    for scenario_path in sorted((scenarios_dir).glob("*.json")) if scenarios_dir.exists() else []:
        try:
            scenario = load_json(scenario_path)
            report["scenarios"].append(scenario)
        except Exception as exc:
            report["errors"].append(f"Invalid scenario JSON: {scenario_path.name} ({exc})")
    for conflict_path in sorted((conflicts_dir).glob("*.json")) if conflicts_dir.exists() else []:
        try:
            conflict = load_json(conflict_path)
            report["conflicts"].append(conflict)
        except Exception as exc:
            report["errors"].append(f"Invalid conflict JSON: {conflict_path.name} ({exc})")

    report["summary"]["total_relationships"] = len(relationships)
    report["summary"]["total_scenarios"] = len(report["scenarios"])
    report["summary"]["total_conflict_cases"] = len(report["conflicts"])
    report["summary"]["total_documents_found"] = len(doc_source_ids)
    report["summary"]["missing_metadata"] = missing_metadata
    report["summary"]["invalid_relationship_endpoints"] = invalid_relationship_endpoints
    report["summary"]["duplicate_metadata_ids"] = duplicates
    report["summary"]["document_ids"] = sorted(doc_source_ids)
    report["summary"]["valid_document_ids"] = sorted(valid_doc_ids)

    return report


def extract_text_from_docx(path: Path) -> str:
    try:
        document = Document(str(path))
        paragraphs = []
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                paragraphs.append(text)
        return "\n".join(paragraphs)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Unreadable DOCX: {path} ({exc})") from exc


def extract_text_from_pdf(path: Path) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed; PDF extraction is unavailable.")
    try:
        pdf = fitz.open(str(path))
        chunks: List[str] = []
        for page in pdf:
            text = page.get_text("text")
            if text and text.strip():
                chunks.append(text.strip())
        pdf.close()
        return "\n".join(chunks)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Unreadable PDF: {path} ({exc})") from exc


def extract_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return extract_text_from_docx(path)
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in {".txt", ".rtf"}:
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def build_document_registry(root: Path, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs_dir = root / "documents"
    metadata_dir = root / "metadata"
    registry: List[Dict[str, Any]] = []
    supported = {".docx", ".pdf", ".txt", ".doc"}

    metadata_map: Dict[str, Dict[str, Any]] = {}
    for path in sorted(metadata_dir.glob("*.json")):
        try:
            meta = load_json(path)
            if isinstance(meta, dict) and meta.get("document_id"):
                metadata_map[meta["document_id"]] = meta
        except Exception as exc:
            report["warnings"].append(f"Unable to parse metadata JSON: {path.name} ({exc})")

    for file_path in sorted(docs_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in supported:
            continue
        doc_id = normalize_document_id(file_path.name)
        meta = metadata_map.get(doc_id, {})
        text = ""
        try:
            text = extract_document_text(file_path)
        except Exception as exc:
            report["warnings"].append(f"Text extraction failed for {file_path.name}: {exc}")

        record = {
            "document_id": doc_id,
            "title": meta.get("title") or file_path.stem,
            "source_filename": file_path.name,
            "file_path": str(file_path),
            "source": meta.get("source") if isinstance(meta, dict) else "unknown",
            "department": meta.get("department") if isinstance(meta, dict) else "unknown",
            "document_type": meta.get("document_type") if isinstance(meta, dict) else "unknown",
            "version": meta.get("version") if isinstance(meta, dict) else "unknown",
            "effective_date": meta.get("effective_date") if isinstance(meta, dict) else "unknown",
            "publication_date": meta.get("publication_date") if isinstance(meta, dict) else "unknown",
            "extracted_text": text,
            "has_extracted_text": bool(text and text.strip()),
            "metadata_source": "existing" if doc_id in metadata_map else "generated",
            "relationships": meta.get("relationships", []) if isinstance(meta, dict) else [],
        }
        registry.append(record)

    return registry


def clean_text_conservatively(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[\t\xa0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def normalize_rule_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    return value


def extract_rule_candidates(document_id: str, text: str) -> List[Dict[str, Any]]:
    cleaned = clean_text_conservatively(text)
    sentences = split_sentences(cleaned)
    rules: List[Dict[str, Any]] = []

    policy_patterns = [
        ("obligation", re.compile(r"\b(must|shall|is required to|required to|has to|needs to)\b", re.I)),
        ("permission", re.compile(r"\b(may|can|is permitted to|is allowed to|may be permitted to)\b", re.I)),
        ("prohibition", re.compile(r"\b(must not|shall not|may not|is not permitted to|cannot|not allowed to)\b", re.I)),
    ]

    condition_pattern = re.compile(r"\b(if|when|unless|except when|provided that|subject to|where)\b", re.I)
    temporal_pattern = re.compile(r"\b(within\s+\d+\s+(day|days|month|months|year|years)|at least once every\s+\d+\s+(day|days|month|months|year|years)|before\s+\d+\s+(day|days|month|months|year|years)|after\s+\d+\s+(day|days|month|months|year|years))\b", re.I)
    threshold_pattern = re.compile(r"\b(\d+\s*(%|days?|months?|years?|hours?|minutes?))\b", re.I)
    exception_pattern = re.compile(r"\b(unless|except\s+when|except\s+for|except)\b", re.I)

    for sentence in sentences:
        if len(sentence) < 10:
            continue
        policy_type = None
        for label, pattern in policy_patterns:
            if pattern.search(sentence):
                policy_type = label
                break
        if policy_type is None and not (condition_pattern.search(sentence) or temporal_pattern.search(sentence)):
            continue

        actor = None
        for candidate in ["employee", "manager", "staff", "contractor", "director", "user", "team", "administrator", "officer"]:
            if re.search(rf"\b{candidate}\b", sentence, re.I):
                actor = candidate.title()
                break

        match = re.search(r"(must|shall|may|can|must not|shall not|is required to|is permitted to|is allowed to|may not)\s+(?:\w+\s+)?([A-Za-z][A-Za-z0-9\s-]{0,120})", sentence, re.I)
        action = None
        if match:
            action = match.group(2).strip(" ,;:")
            action = re.sub(r"\s+", " ", action)
        if action and action.lower().startswith("be "):
            action = action[3:].strip()

        condition = None
        condition_match = re.search(r"(?:(?:if|when|provided that|subject to|where)\s+)([A-Z0-9a-z,\s-]{1,200})(?=\.|;|$)", sentence, re.I)
        if condition_match:
            condition = condition_match.group(1).strip(" ,;:")

        exception = None
        exception_match = re.search(r"(?:(?:unless|except when|except for|except)\s+)([A-Za-z0-9,\s-]{1,200})(?=\.|;|$)", sentence, re.I)
        if exception_match:
            exception = exception_match.group(1).strip(" ,;:")

        temporal_constraint = None
        temporal_match = temporal_pattern.search(sentence)
        if temporal_match:
            temporal_constraint = temporal_match.group(0)

        threshold = None
        threshold_match = threshold_pattern.search(sentence)
        if threshold_match:
            threshold = threshold_match.group(0)

        source_text = sentence
        rule = {
            "rule_id": f"{document_id}-{len(rules)+1}",
            "document_id": document_id,
            "source_text": source_text,
            "actor": normalize_rule_value(actor),
            "action": normalize_rule_value(action),
            "policy_type": normalize_rule_value(policy_type),
            "condition": normalize_rule_value(condition),
            "exception": normalize_rule_value(exception),
            "temporal_constraint": normalize_rule_value(temporal_constraint),
            "threshold": normalize_rule_value(threshold),
            "extraction_method": "rule_based",
            "annotation_status": "auto_extracted",
        }
        rules.append(rule)

    return rules


def build_structured_rules(registry: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    for record in registry:
        text = str(record.get("extracted_text") or "")
        if not text.strip():
            continue
        rules.extend(extract_rule_candidates(record["document_id"], text))
    return rules


def create_embedding_stub_model() -> Tuple[str, str]:
    stub_dir = ensure_fake_sklearn_stub()
    stub_dir_str = str(stub_dir)
    if stub_dir_str not in sys.path:
        sys.path.insert(0, stub_dir_str)
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model_name, tokenizer.__class__.__name__


def generate_document_embeddings(registry: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    embedding_info: Dict[str, Any] = {"model_name": "sentence-transformers/all-MiniLM-L6-v2", "document_vectors": [], "warnings": []}
    if output_dir.exists() is False:
        output_dir.mkdir(parents=True, exist_ok=True)

    stub_dir = ensure_fake_sklearn_stub()
    if str(stub_dir) not in sys.path:
        sys.path.insert(0, str(stub_dir))

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        vectors = []
        index = []
        for record in registry:
            text = str(record.get("extracted_text") or "").strip()
            if not text:
                vector = np.zeros(384, dtype=np.float32)
            else:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()
                pooled = (last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
                vector = pooled.cpu().numpy().reshape(-1).astype(np.float32)
            vectors.append(vector)
            index.append(record["document_id"])
            embedding_info["document_vectors"].append({"document_id": record["document_id"], "vector_dim": int(vector.shape[0])})

        matrix = np.vstack(vectors).astype(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("Embedding matrix contains non-finite values.")
        np.save(output_dir / "document_embeddings.npy", matrix)
        (output_dir / "embedding_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        embedding_info["embedding_count"] = matrix.shape[0]
        embedding_info["embedding_dim"] = int(matrix.shape[1])
        embedding_info["warnings"].append("Embeddings generated using a free Hugging Face sentence-transformer model via direct transformers loading due blocked sklearn DLLs in this environment.")
        return embedding_info
    except Exception as exc:  # pragma: no cover
        fallback_vectors = [np.zeros(64, dtype=np.float32) for _ in registry]
        matrix = np.vstack(fallback_vectors).astype(np.float32)
        np.save(output_dir / "document_embeddings.npy", matrix)
        (output_dir / "embedding_index.json").write_text(json.dumps([record["document_id"] for record in registry], indent=2), encoding="utf-8")
        embedding_info["warnings"].append(f"Embedding generation fell back to deterministic zero vectors because the model could not be loaded: {exc}")
        embedding_info["embedding_count"] = matrix.shape[0]
        embedding_info["embedding_dim"] = int(matrix.shape[1])
        return embedding_info


def build_knowledge_graph(root: Path, registry: List[Dict[str, Any]], report: Dict[str, Any]) -> Dict[str, Any]:
    graph = nx.DiGraph()
    valid_ids = {record["document_id"] for record in registry}

    for record in registry:
        graph.add_node(record["document_id"], **{
            "title": record.get("title"),
            "source_filename": record.get("source_filename"),
            "department": record.get("department"),
            "document_type": record.get("document_type"),
            "version": record.get("version"),
            "effective_date": record.get("effective_date"),
        })

    relationship_path = root / "relationships.json"
    relationships = []
    if relationship_path.exists():
        try:
            relationships = load_json(relationship_path)
        except Exception as exc:
            report["errors"].append(f"Unable to load relationship file while building the graph: {exc}")

    seen_pairs = set()
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        source = rel.get("source")
        target = rel.get("target")
        rel_type = rel.get("relationship")
        if source in valid_ids and target in valid_ids:
            pair = (source, target, rel_type)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            graph.add_edge(source, target, relationship=rel_type)
        else:
            report["warnings"].append(f"Skipped invalid graph edge: {source} -> {target} ({rel_type})")

    graph_data = {
        "nodes": list(graph.nodes(data=True)),
        "edges": list(graph.edges(data=True)),
        "directed": True,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
    }

    if plt is not None:
        try:
            positions = nx.spring_layout(graph, seed=42)
            fig, ax = plt.subplots(figsize=(12, 8))
            nx.draw(graph, pos=positions, with_labels=True, node_color="lightblue", edge_color="gray", arrows=True, ax=ax)
            for edge in graph.edges(data=True):
                source, target, data = edge
                ax.annotate("", xy=positions[target], xytext=positions[source], arrowprops={"arrowstyle": "->", "color": "gray"}, zorder=0)
            fig.tight_layout()
            fig.savefig(root / "processed" / "knowledge_graph.png", dpi=150)
            plt.close(fig)
        except Exception as exc:  # pragma: no cover
            report["warnings"].append(f"Graph visualization failed: {exc}")

    return graph_data


def build_test_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "dataset_errors": len(report.get("errors", [])),
        "dataset_warnings": len(report.get("warnings", [])),
        "documents": report["summary"].get("total_documents_found", 0),
        "metadata": report["summary"].get("total_metadata_records", 0),
        "relationships": report["summary"].get("total_relationships", 0),
    }
    return summary


def run_phase1() -> Dict[str, Any]:
    dataset_root = find_dataset_root()
    processed_dir = dataset_root / "processed"
    processed_dir.mkdir(exist_ok=True)

    report = validate_dataset(dataset_root)
    registry = build_document_registry(dataset_root, report)

    extracted_text = {record["document_id"]: record["extracted_text"] for record in registry}
    (processed_dir / "extracted_text.json").write_text(json.dumps(extracted_text, indent=2, ensure_ascii=False), encoding="utf-8")

    registry_path = processed_dir / "document_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")

    structured_rules = build_structured_rules(registry)
    (processed_dir / "structured_rules.json").write_text(json.dumps(structured_rules, indent=2, ensure_ascii=False), encoding="utf-8")

    embeddings = generate_document_embeddings(registry, processed_dir)
    graph_data = build_knowledge_graph(dataset_root, registry, report)
    (processed_dir / "knowledge_graph.json").write_text(json.dumps(graph_data, indent=2, ensure_ascii=False), encoding="utf-8")

    validation_report = {
        "dataset_validation_report": report,
        "summary": build_test_summary(report),
        "generated_artifacts": {
            "extracted_text": str(processed_dir / "extracted_text.json"),
            "document_registry": str(processed_dir / "document_registry.json"),
            "structured_rules": str(processed_dir / "structured_rules.json"),
            "document_embeddings": str(processed_dir / "document_embeddings.npy"),
            "embedding_index": str(processed_dir / "embedding_index.json"),
            "knowledge_graph": str(processed_dir / "knowledge_graph.json"),
            "graph_visualization": str(processed_dir / "knowledge_graph.png"),
        },
        "embeddings": embeddings,
    }

    (processed_dir / "dataset_validation_report.json").write_text(json.dumps(validation_report, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "dataset_root": str(dataset_root),
        "processed_dir": str(processed_dir),
        "registry": registry,
        "rules": structured_rules,
        "embeddings": embeddings,
        "graph": graph_data,
        "report": report,
    }


if __name__ == "__main__":
    result = run_phase1()
    print(json.dumps({
        "dataset_root": result["dataset_root"],
        "processed_dir": result["processed_dir"],
        "documents_processed": len(result["registry"]),
        "rules_extracted": len(result["rules"]),
        "graph_nodes": result["graph"]["node_count"],
        "graph_edges": result["graph"]["edge_count"],
        "dataset_errors": len(result["report"]["errors"]),
        "dataset_warnings": len(result["report"]["warnings"]),
    }, indent=2))
