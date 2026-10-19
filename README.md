# AI-COS — Artificial Intelligence Cognitive Organizational System

> **Document Intelligence, Policy Parsing, Knowledge Graphs & Semantic Reasoning Platform**

---

## 📌 Project Overview

**AI-COS** (*Artificial Intelligence Cognitive Organizational System*) is an intelligent document analysis platform engineered for organizational governance, policy lifecycle management, and regulatory compliance. It processes unstructured organizational policies, manuals, and regulations, structures their normative contents into machine-processable rules, maps dependencies into an organizational Knowledge Graph, and supports semantic retrieval.

---

## 🏛️ System Architecture (Phase 1)

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Streamlit Interactive Dashboard                   │
│  (Overview, Document Library, Text & Rules Inspection, Semantic Search,│
│              Knowledge Graph Explorer, Dataset Scenarios)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
               ┌────────────────────┴───────────────────┐
               ▼                                        ▼
    ┌──────────────────────┐                ┌───────────────────────┐
    │  Ingestion Service   │                │   Document Service    │
    └──────────┬───────────┘                └───────────┬───────────┘
               │                                        │
    ┌──────────┴───────────────┐                        │
    ▼                          ▼                        ▼
┌──────────────┐     ┌──────────────────┐    ┌──────────────────────┐
│ Extractors   │     │  NLP Processing  │    │ Persistent Storage   │
│ (PDF, DOCX,  │────▶│ (Chunker, spaCy, │───▶│ - SQLite (Metadata)  │
│  Text Normal)│     │  Rule Parser)    │    │ - NumPy (.npy vecs)  │
└──────────────┘     └─────────┬────────┘    │ - NetworkX (JSON KG) │
                               │             └──────────────────────┘
                               ▼
                     ┌──────────────────┐
                     │ MiniLM Embeddings│
                     │ (384-d Cosine)   │
                     └──────────────────┘
```

### 1. Ingestion & Preprocessing
- **Multi-Format Extraction**: Native support for **PDF** (`PyMuPDF`), **DOCX** (`python-docx`), and plain **TXT**.
- **Sanitization & Quality Assurance**: Unicode NFC normalization, whitespace compaction, page-artifact stripping, minimum character threshold checks, and SHA-256 hash collision checks to prevent duplicate uploads.
- **Semantic Chunking**: Paragraph-aware chunking with customizable overlap window (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`) preserving character offsets for full auditability.

### 2. Neuro-Symbolic NLP & Policy Parsing
- **Modality Extraction**: Rule-based regex parser identifying deontic modality:
  - `Obligation` (*must, shall, required to*)
  - `Permission` (*may, can, permitted to*)
  - `Prohibition` (*must not, shall not, prohibited*)
  - `Recommendation` (*should, ought to, advisable*)
- **Entity & Temporal Parsing**: spaCy-powered entity recognition (`ORG`, `PERSON`, `LAW`, `DATE`) and regex extraction of actors, conditions, numerical thresholds, and recurrence.

### 3. Vector Embeddings & Semantic Search
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Storage Strategy**: Per-version `.npy` vector files stored under `data/embeddings/`, indexed in SQLite via `EmbeddingMetadata`.
- **Search Engine**: Exact dot-product cosine similarity over normalized vectors with top-k ranking and threshold gating.

### 4. Knowledge Graph (NetworkX)
- **Node Taxonomy**: `document`, `version`, `department`, `policy_statement`.
- **Edge Semantics**: `AFFECTS`, `REFERENCES`, `CONTROLS`, `INFLUENCES`, `VERSION_OF`, `BELONGS_TO`.
- **Cross-Session Reload**: Serialized to JSON node-link format under `data/graphs/knowledge_graph.json`.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10 – 3.12
- Virtual environment (recommended)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/Jeevanth98/AI-COS.git
cd AI-COS

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### 3. Database Initialization & Seeding
```powershell
# Ingest the 17 synthetic policy documents and graph relationships
python scripts/seed_dataset.py
```

### 4. Running the Dashboard
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 🧪 Testing & Verification

Run the full automated test suite (25 unit and integration tests):
```powershell
pytest tests/ -v
```

### Test Coverage Highlights:
- **`tests/test_database.py`**: SQLite models, cascade deletes, transaction rollback, and error logging.
- **`tests/test_extractors.py`**: DOCX/PDF extractors, character boundary verification, and text normalization.
- **`tests/test_nlp.py`**: Tokenization, modality detection, actor identification, and policy parsing.
- **`tests/test_embeddings.py`**: Vector generation shape, L2 normalization, and `.npy` disk persistence.
- **`tests/test_graph.py`**: NetworkX DiGraph creation, typed edges, and JSON persistence reload.
- **`tests/test_ingestion_pipeline.py`**: End-to-end ingestion, duplicate rejection, and status tracking.
- **`tests/test_restart_persistence.py`**: Simulation of complete app termination and cross-session data restoration.

---

## 📂 Project Structure

```
AI-COS/
├── app.py                     # Streamlit multi-page dashboard
├── requirements.txt           # Project dependencies
├── .env.example               # Environment configuration template
├── config/
│   └── settings.py            # Centralized settings and path configurations
├── database/
│   ├── models.py              # SQLAlchemy ORM models
│   ├── db.py                  # Database connection, WAL mode, session factory
│   └── repositories.py        # Data access objects / CRUD helpers
├── ingestion/
│   ├── document_loader.py     # Ingestion entry point & validation
│   ├── pdf_extractor.py       # PyMuPDF extractor with scanned document check
│   ├── docx_extractor.py      # python-docx parser for paragraphs & tables
│   ├── text_cleaner.py        # Unicode & whitespace cleaner
│   └── chunker.py             # Paragraph-aware semantic chunker
├── nlp/
│   ├── preprocessing.py       # spaCy NER and tokenization
│   ├── policy_parser.py       # Rule-based policy statement extractor
│   └── embeddings.py          # SentenceTransformer & vector search
├── graph/
│   ├── graph_builder.py       # NetworkX DiGraph & JSON persistence
│   └── graph_visualizer.py    # Plotly interactive network graph
├── services/
│   ├── ingestion_service.py   # 10-stage transactional pipeline orchestrator
│   └── document_service.py    # Read/write queries for dashboard pages
├── scripts/
│   └── seed_dataset.py        # Batch-ingestion and graph seeder
├── tests/                     # 25 automated unit & integration tests
└── Dataset/
    └── AI-COS_Dataset/        # Synthetic documents, metadata, conflicts, scenarios
```

---

## 🗺️ Roadmap: Subsequent Phases

- **Phase 2**: Semantic Comparison, Reasoning, SIPA (Semantic Impact Prediction Analysis), and ASCD (Automated Semantic Conflict Detection).
- **Phase 3**: Actionable Recommendation Engine, Impact Ranking Visualizations, and Executive Reports.
- **Phase 4**: Real-World Document Validation (CBSE/academic policies), Benchmark Evaluation, and Hardening.
