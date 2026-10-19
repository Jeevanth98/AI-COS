"""
AI-COS Configuration Settings
All application-wide configuration is centralised here.
Values are read from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
GRAPHS_DIR = DATA_DIR / "graphs"
EXPORTS_DIR = DATA_DIR / "exports"
BACKUPS_DIR = DATA_DIR / "backups"
DATABASE_DIR = BASE_DIR / "database"
DATASET_DIR = BASE_DIR / "Dataset" / "AI-COS_Dataset"

# Ensure runtime directories exist
for _d in [UPLOADS_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, GRAPHS_DIR, EXPORTS_DIR, BACKUPS_DIR, DATABASE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Database ─────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_DIR / 'aicos.db'}")

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM: int = 384          # all-MiniLM-L6-v2 output dimension
EMBEDDING_BATCH_SIZE: int = 32

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))          # chars
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))     # chars

# ── Semantic search ───────────────────────────────────────────────────────────
SEMANTIC_TOP_K: int = int(os.getenv("SEMANTIC_TOP_K", "10"))
SEMANTIC_SIMILARITY_THRESHOLD: float = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.35"))

# ── File upload ───────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")) * 1024 * 1024
ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".txt"}

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_ENABLED: bool = os.getenv("OCR_ENABLED", "0") == "1"
TESSERACT_PATH: str = os.getenv("TESSERACT_PATH", "")

# ── Knowledge graph ───────────────────────────────────────────────────────────
GRAPH_FILE: Path = GRAPHS_DIR / "aicos_graph.json"
GRAPH_SEMANTIC_EDGE_THRESHOLD: float = float(os.getenv("GRAPH_SEMANTIC_EDGE_THRESHOLD", "0.60"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── Application ───────────────────────────────────────────────────────────────
APP_NAME: str = "AI-COS"
APP_VERSION: str = "1.0.0-phase1"
APP_ENV: str = os.getenv("APP_ENV", "development")
