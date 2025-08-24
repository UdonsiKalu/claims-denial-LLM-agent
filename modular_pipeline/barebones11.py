# === TOP-LEVEL IMPORTS ===
import os
import shutil
import time
import streamlit as st
from datetime import datetime
from chunking import load_and_chunk_manuals
from langchain_ollama import OllamaEmbeddings
from faiss_gpu_entropy import CMSDenialAnalyzer
from langchain.schema import Document

# === Default Embeddings ===
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# === STREAMLIT PAGE ===
st.set_page_config(page_title="Retrieval Studio (Test)", layout="wide")
st.title("Retrieval Studio: Chunking & Retriever Test")

# === SESSION INIT ===
for key in ["chunks", "retriever", "analyzer"]:
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state.setdefault("rebuild_index", False)
st.session_state.setdefault("retriever_initialized", False)
st.session_state.setdefault("config_history", [])

# === TRANSIENT TERMINAL BOX (bottom HUD) ===
terminal_box = st.empty()

def log_status(message, level="info", keep=False):
    """Show transient status HUD at bottom (replaced unless keep=True)."""
    box = terminal_box
    if level == "success":
        box.success(message)
    elif level == "warning":
        box.warning(message)
    elif level == "error":
        box.error(message)
    else:
        box.info(message)
    if not keep:
        time.sleep(0.8)
        box.empty()

# === STRATEGY MAP ===
STRAT_MAP = {
    "Fixed": "Fixed-size",
    "Recursive": "Recursive",
    "Header-aware": "Header-aware",
    "Semantic": "Semantic",
    "By-page": "By-page"
}

# === SIDEBAR UI ===
with st.sidebar:
    st.header("Configuration Panel")

    # --- File Upload ---
    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Upload additional PDFs or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    # --- Chunking Settings ---
    with st.expander("Chunking Settings", expanded=True):
        ui_chunking_strategy = st.selectbox(
            "Chunking Strategy",
            ["Fixed", "Recursive", "Header-aware", "Semantic", "By-page"],
            index=0,
        )
        ui_chunk_size = st.number_input("Chunk Size", 1000, 20000, 10000, step=500)
        ui_chunk_overlap = st.number_input("Chunk Overlap", 0, 5000, 2000, step=200)

        ui_header_levels = None
        if ui_chunking_strategy == "Header-aware":
            ui_header_levels = st.slider("Header Levels", 1, 6, 3)

        ui_semantic_threshold = None
        if ui_chunking_strategy == "Semantic":
            ui_semantic_threshold = st.slider("Semantic Threshold", 0.0, 1.0, 0.5)

    # --- Retriever Settings ---
    with st.expander("Retriever Settings", expanded=True):
        ui_faiss_k = st.slider("FAISS k", 1, 50, 5)
        ui_bm25_k = st.slider("BM25 k", 1, 20, 3)
        ui_faiss_fetch_k = st.slider("FAISS Fetch k", 10, 200, 50, step=10)
        ui_weights = st.slider("Ensemble Weights (FAISS vs BM25)", 0.0, 1.0, (0.5, 0.5))

    # --- Actions ---
    if st.button("Rechunk & Reindex", use_container_width=True):
        frozen = {
            "chunking_strategy": ui_chunking_strategy,
            "chunk_size": ui_chunk_size,
            "chunk_overlap": ui_chunk_overlap,
            "header_levels": ui_header_levels,
            "semantic_threshold": ui_semantic_threshold,
            "faiss_k": ui_faiss_k,
            "bm25_k": ui_bm25_k,
            "faiss_fetch_k": ui_faiss_fetch_k,
            "weights": ui_weights,
        }
        st.session_state["config_history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": frozen
        })
        st.session_state["frozen_params"] = frozen
        st.session_state["uploaded_files"] = uploaded_files  # save session copy
        st.session_state["rebuild_index"] = True
        st.rerun()

    if st.button("Reset Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    if st.button("Clear Disk Cache", use_container_width=True):
        shutil.rmtree(".cache/chunks", ignore_errors=True)
        os.makedirs(".cache/chunks", exist_ok=True)
        log_status("Disk cache cleared!", "success", keep=True)

    # --- Config History Log ---
    with st.expander("Config History", expanded=False):
        if not st.session_state["config_history"]:
            st.info("No configs applied yet.")
        else:
            for entry in reversed(st.session_state["config_history"]):
                st.markdown(f"**{entry['timestamp']}**")
                st.json(entry["params"])

# === LIVE CONFIG (mirrors sidebar) ===
live_params = {
    "chunking_strategy": ui_chunking_strategy,
    "chunk_size": ui_chunk_size,
    "chunk_overlap": ui_chunk_overlap,
    "header_levels": ui_header_levels,
    "semantic_threshold": ui_semantic_threshold,
    "faiss_k": ui_faiss_k,
    "bm25_k": ui_bm25_k,
    "faiss_fetch_k": ui_faiss_fetch_k,
    "weights": ui_weights,
}
st.subheader("Live Config (Preview)")
st.json(live_params)

# === Build/Rebuild Analyzer ===
if st.session_state["rebuild_index"]:
    params = st.session_state["frozen_params"]

    log_status("Rebuilding retriever... please wait.", keep=True)

    os.environ["FORCE_REBUILD"] = "1"

    # Persistent progress bar
    progress = st.progress(0, text="Initializing rebuild...")

    # Default hardwired docs
    default_docs = [
        "CMS_MANUAL_PATH/default_manual1.pdf",
        "CMS_MANUAL_PATH/default_manual2.pdf",
        "CMS_MANUAL_PATH/default_manual3.pdf"
    ]

    # Save uploaded docs if any
    extra_docs = []
    if st.session_state.get("uploaded_files"):
        os.makedirs(".cache/uploads", exist_ok=True)
        for uf in st.session_state["uploaded_files"]:
            file_path = os.path.join(".cache/uploads", uf.name)
            with open(file_path, "wb") as f:
                f.write(uf.getbuffer())
            extra_docs.append(file_path)

    all_docs = default_docs + extra_docs

        # --- Step 1: Chunking ---
    for pct in range(0, 40, 5):
        progress.progress(pct, text="Step 1/3: Chunking documents...")
        time.sleep(0.05)

    # If you uploaded files, pass them; otherwise use defaults inside load_and_chunk_manuals
    if all_docs:
        chunks = load_and_chunk_manuals(
            input_files=all_docs,
            chunking_strategy=params.get("chunking_strategy", "Fixed-size"),
            chunk_size=params.get("chunk_size", 1000),
            chunk_overlap=params.get("chunk_overlap", 200),
            header_levels=params.get("header_levels"),
            semantic_threshold=params.get("semantic_threshold", 0.5),
            embeddings=embeddings,
            enforce_max_size=True,   # ✅ make sure max size respected
        )
    else:
        chunks = load_and_chunk_manuals(
            chunking_strategy=params.get("chunking_strategy", "Fixed-size"),
            chunk_size=params.get("chunk_size", 1000),
            chunk_overlap=params.get("chunk_overlap", 200),
            header_levels=params.get("header_levels"),
            semantic_threshold=params.get("semantic_threshold", 0.5),
            embeddings=embeddings,
            enforce_max_size=True,
        )

    # ✅ Ensure chunks is always a list
    if not chunks:
        chunks = []
    elif isinstance(chunks[0], dict):
        chunks = [Document(page_content=c["content"], metadata=c["metadata"]) for c in chunks]

    st.session_state["chunks"] = chunks
    progress.progress(40, text=f"Step 1 complete: {len(chunks)} chunks created.")

    # --- Step 2: Embedding (simulate progress) ---
    for pct in range(41, 70, 3):
        progress.progress(pct, text="Step 2/3: Preparing embeddings...")
        time.sleep(0.05)
    progress.progress(70, text="Step 2 complete: Embeddings ready.")

    # --- Step 3: Retriever/Analyzer ---
    for pct in range(71, 99, 2):
        progress.progress(pct, text="Step 3/3: Building analyzer + retriever...")
        time.sleep(0.05)

        analyzer = CMSDenialAnalyzer(
        exclude_tokens=params.get("exclude_tokens", []),
        faiss_k=params.get("faiss_k", 5),
        bm25_k=params.get("bm25_k", 3),
        faiss_fetch_k=params.get("faiss_fetch_k", 20),
        weights=params.get("weights", (0.4, 0.6)),
        chunking_strategy=params.get("chunking_strategy", "Fixed-size"),
        chunk_size=params.get("chunk_size"),
        chunk_overlap=params.get("chunk_overlap"),
        header_levels=params.get("header_levels"),
        semantic_threshold=params.get("semantic_threshold"),
        embeddings=params.get("embeddings"),
        chunks=chunks,   # <--- pass precomputed chunks here
        )



    st.session_state["analyzer"] = analyzer
    st.session_state["retriever"] = analyzer.retrieval["retriever"]
    st.session_state["retriever_initialized"] = True
    st.session_state["rebuild_index"] = False

    progress.progress(100, text="Rebuild complete: Retriever initialized.")

    # Clear HUD + progress after finish
    time.sleep(0.5)
    terminal_box.empty()
    progress.empty()

    # show sample chunks
    st.subheader("Sample Chunks")
    for i, c in enumerate(st.session_state["chunks"][:3]):
        st.markdown(f"**Chunk {i+1}** — {len(c.page_content)} chars")
        st.text(c.page_content[:400] + "...\n")

        st.text(c.page_content[:400] + "...\n")
