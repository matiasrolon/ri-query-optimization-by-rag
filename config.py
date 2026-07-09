# -*- coding: utf-8 -*-
"""
Configuration module.
Loads settings from .env file and defines project-wide constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Indexing method ───────────────────────────────────────────────────────────
# Valid values: "pyterrier", "pisa"
INDEXING_METHOD = os.getenv("INDEXING_METHOD", "pyterrier").lower().strip()

if INDEXING_METHOD not in ("pyterrier", "pisa"):
    raise ValueError(
        f"INDEXING_METHOD inválido: '{INDEXING_METHOD}'. "
        "Debe ser 'pyterrier' o 'pisa'."
    )

# ─── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "./data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(OUTPUT_DIR, "index"))

# Dataset files (relative to DATA_DIR)
COLLECTION_FILE = os.path.join(DATA_DIR, "collection.tsv")
QUERIES_DIR = os.path.join(DATA_DIR, "queries")
QRELS_FILE = os.path.join(DATA_DIR, "qrels.dev.tsv")

# ─── Ensure required directories exist ─────────────────────────────────────────
for _dir in (DATA_DIR, OUTPUT_DIR, INDEX_DIR, QUERIES_DIR):
    os.makedirs(_dir, exist_ok=True)

# ─── Dataset parameters ───────────────────────────────────────────────────────
TOTAL_PASSAGES = 8_841_823  # Total passages in MS MARCO passage collection

SAMPLE_FRACTION = float(os.getenv("SAMPLE_FRACTION", "1.0"))
SAMPLE_SIZE = int(TOTAL_PASSAGES * SAMPLE_FRACTION)

# ─── System ────────────────────────────────────────────────────────────────────
THREADS = os.cpu_count()

# ─── Index reuse ───────────────────────────────────────────────────────────────
# Set to "true" in .env to force a full re-index even when an index already
# exists on disk. By default the program will reuse a previously built index.
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "false").lower().strip() == "true"
