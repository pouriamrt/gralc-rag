"""
GraLC-RAG configuration: paths, model settings, and environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

UMLS_API_KEY: str = os.getenv("UMLS_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]  # repo root

DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CORPUS_DIR: Path = DATA_DIR / "raw"
PUBMEDQA_DIR: Path = DATA_DIR / "pubmedqa"

for _dir in (DATA_DIR, RESULTS_DIR, CORPUS_DIR, PUBMEDQA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embedding settings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384

# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------
MAX_TOKENS_PER_CHUNK: int = 512
CHUNK_OVERLAP: int = 64

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
TOP_K_RETRIEVAL: int = 5

# ---------------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------------
OPENAI_MODEL: str = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Knowledge-graph fusion weight (beta)
# ---------------------------------------------------------------------------
KG_WEIGHT_BETA: float = 0.7
