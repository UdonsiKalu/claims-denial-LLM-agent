import os

# === Base Directory ===
BASE_DIR = "/media/udonsi-kalu/New Volume/denials/denials"

# === Manual & Index Paths ===
CMS_MANUAL_PATH = "/media/udonsi-kalu/New Volume/denials/denials/manuals/"
FAISS_INDEX_PATH = os.environ.get("INDEX_PATH", os.path.join(BASE_DIR, "cms_entropy_faiss_index"))

# === Chapter Mapping ===
PRIORITY_CHAPTERS = {
    "12": "Physicians and NPP Services",
    "13": "Radiology Services and Other Diagnostic Procedures",
    "15": "Covered Medical and Other Health Services",
    "16": "Laboratory Services",
    "30": "Clinical Laboratory Services"
}

# === Chunking Defaults ===
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_CHUNK_OVERLAP = 2000

# === Embedding Model ===
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
