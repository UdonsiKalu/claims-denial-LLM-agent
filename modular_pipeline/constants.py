import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file if present

# === Base Directory ===
BASE_DIR = "/media/udonsi-kalu/New Volume/denials/denials/modular_pipeline"

# === Manual & Index Paths ===
CMS_MANUAL_PATH = os.path.join(BASE_DIR, "manuals")
FAISS_INDEX_PATH = os.environ.get("INDEX_PATH", os.path.join(BASE_DIR, "cms_entropy_faiss_index"))

# === Chapter Mapping ===
PRIORITY_CHAPTERS = {
    "Chapter 1": "General Billing Requirements",
    "Chapter 12": "Physicians Nonphysician Practitioners",
    "Chapter 23": "Fee Schedule Administration",
    "Chapter 30": "Financial Liability Protections"
}

# === Chunking Defaults ===
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 10000))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 2000))

# === Embedding Model ===
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "nomic-embed-text-v1")

# === Vector DB Backend Switch (FAISS or QDRANT) ===
USE_QDRANT = os.environ.get("USE_QDRANT", "0") == "1"  # set USE_QDRANT=1 in .env to activate

# === Qdrant Configuration ===
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "cms_chunks")
QDRANT_URL = os.environ.get("QDRANT_URL", f"http://{QDRANT_HOST}:{QDRANT_PORT}")
