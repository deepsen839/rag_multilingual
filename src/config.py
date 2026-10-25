import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


# ============================================================
# DIRECTORIES
# ============================================================

PDF_DIR = BASE_DIR / "data" / "pdfs"

PROCESSED_DIR = BASE_DIR / "data" / "processed"

STORAGE_DIR = BASE_DIR / "storage"

BM25_PATH = STORAGE_DIR / "bm25.pkl"

REGISTRY_PATH = STORAGE_DIR / "documents.json"

METRICS_PATH = STORAGE_DIR / "metrics.jsonl"


for p in [
    PDF_DIR,
    PROCESSED_DIR,
    STORAGE_DIR,
]:
    p.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# QDRANT
# ============================================================

# IMPORTANT:
# Inside Docker Compose, use the Qdrant service name.
# Locally without Docker, localhost is used.

QDRANT_HOST = os.getenv(
    "QDRANT_HOST",
    "localhost"
)

QDRANT_PORT = int(
    os.getenv(
        "QDRANT_PORT",
        "6333"
    )
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "pdf_chunks"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# EMBEDDING BATCH SIZE
# ============================================================

# CPU-only machine:
# 16 or 32 is safer than a very large batch.

BATCH_SIZE = int(
    os.getenv(
        "BATCH_SIZE",
        "16"
    )
)


# ============================================================
# CHUNKING
# ============================================================

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "800"
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "120"
    )
)


# ============================================================
# OCR
# ============================================================

OCR_DPI = int(
    os.getenv(
        "OCR_DPI",
        "300"
    )
)

MIN_NATIVE_CHARS = int(
    os.getenv(
        "MIN_NATIVE_CHARS",
        "50"
    )
)


# ============================================================
# RETRIEVAL
# ============================================================

DENSE_TOP_K = int(
    os.getenv(
        "DENSE_TOP_K",
        "20"
    )
)

BM25_TOP_K = int(
    os.getenv(
        "BM25_TOP_K",
        "20"
    )
)

FINAL_TOP_K = int(
    os.getenv(
        "FINAL_TOP_K",
        "8"
    )
)

RRF_K = int(
    os.getenv(
        "RRF_K",
        "60"
    )
)


# ============================================================
# OCR GENERIC LANGUAGE
# ============================================================

OCR_GENERIC_LANG = os.getenv(
    "OCR_GENERIC_LANG",
    "eng"
)


# ============================================================
# GROQ
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    ""
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "glm-ocr"
)

OLLAMA_TIMEOUT = int(
    os.getenv("OLLAMA_TIMEOUT", "180")
)