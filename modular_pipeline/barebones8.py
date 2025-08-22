# === TOP-LEVEL IMPORTS ===
import os
import shutil
import streamlit as st
from chunking import load_and_chunk_manuals
from langchain_ollama import OllamaEmbeddings  
from faiss_gpu_entropy import CMSDenialAnalyzer  # ✅ your analyzer
from langchain.schema import Document   # ✅ normalize dict → Document

# === Default Embeddings (used by semantic chunker) ===
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# === STREAMLIT PAGE ===
st.set_page_config(page_title="Retrieval Studio (Test)", layout="wide")
st.title("Retrieval Studio: Chunking & Retriever Test")

# === SESSION CLEANUP AT STARTUP ===
# Ensure nothing carries over unless explicitly rebuilt
for key in ["chunks", "retriever", "analyzer", "frozen_params"]:
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state.setdefault("rebuild_index", False)
st.session_state.setdefault("retriever_initialized", False)

# === Default Frozen Params ===
DEFAULT_FROZEN_PARAMS = {
    "chunking_strategy": "Header-aware",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "header_levels": 3,
    "semantic_threshold": None,
    "faiss_k": 5,
    "bm25_k": 3,
    "faiss_fetch_k": 50,
    "weights": (0.5, 0.5),
}
st.session_state.setdefault("frozen_params", DEFAULT_FROZEN_PARAMS.copy())

# === STATUS PLACEHOLDER (so we can clear messages later) ===
status_box = st.empty()

# === STATUS INDICATOR ===
if st.session_state["rebuild_index"]:
    status_box.info("⏳ Rebuilding retriever... please wait.")
elif st.session_state["retriever_initialized"]:
    status_box.success("✅ Retriever initialized and ready.")
else:
    status_box.warning("❌ Retriever not yet initialized. Click **Rechunk & Reindex** to build.")

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

    # --- Chunking Settings ---
    with st.expander("⚙️ Chunking Settings", expanded=True):
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
    with st.expander("🔍 Retriever Settings", expanded=True):
        ui_faiss_k = st.slider("FAISS k", 1, 50, 5)
        ui_bm25_k = st.slider("BM25 k", 1, 20, 3)
        ui_faiss_fetch_k = st.slider("FAISS Fetch k", 10, 200, 50, step=10)
        ui_weights = st.slider("Ensemble Weights (FAISS vs BM25)", 0.0, 1.0, (0.5, 0.5))

    # --- Actions ---
    if st.button("🚀 Rechunk & Reindex", use_container_width=True):
        st.session_state["frozen_params"] = {
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
        st.session_state["rebuild_index"] = True
        st.rerun()  # force rebuild on same run

    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.clear()
        st.experimental_rerun()

    if st.button("🗑️ Clear Disk Cache", use_container_width=True):
        shutil.rmtree(".cache/chunks", ignore_errors=True)
        os.makedirs(".cache/chunks", exist_ok=True)
        st.success("Disk cache cleared!")

# === MAIN CONTENT ===
st.subheader("Current Frozen Params")
if st.session_state["frozen_params"]:
    st.json(st.session_state["frozen_params"])
else:
    st.info("⚠️ No parameters frozen yet. Use the sidebar to Rechunk & Reindex.")

# === Build/Rebuild Analyzer ===
if st.session_state["rebuild_index"]:
    params = st.session_state["frozen_params"]

    with st.spinner("⏳ Rebuilding retriever... please wait."):
        os.environ["FORCE_REBUILD"] = "1"

        # ✅ force re-chunk
        chunks = load_and_chunk_manuals(
            chunking_strategy=params.get("chunking_strategy", "Fixed-size"),
            chunk_size=params.get("chunk_size", 1000),
            chunk_overlap=params.get("chunk_overlap", 200),
            header_levels=params.get("header_levels"),
            semantic_threshold=params.get("semantic_threshold", 0.5),
            embeddings=embeddings,
        )
        # 🔄 normalize dicts → Documents
        if chunks and isinstance(chunks[0], dict):
            chunks = [
                Document(page_content=c["content"], metadata=c["metadata"])
                for c in chunks
            ]

        st.session_state["chunks"] = chunks

        analyzer = CMSDenialAnalyzer(
            exclude_tokens=params.get("exclude_tokens", []),
            faiss_k=params.get("faiss_k", 5),
            bm25_k=params.get("bm25_k", 3),
            faiss_fetch_k=params.get("faiss_fetch_k", 25),
            weights=params.get("weights", (0.5, 0.5)),
            chunking_strategy=params.get("chunking_strategy", "Fixed"),
            chunk_size=params.get("chunk_size", 512),
            chunk_overlap=params.get("chunk_overlap", 64),
            header_levels=params.get("header_levels", 3),
            semantic_threshold=params.get("semantic_threshold", 0.5),
            embeddings=embeddings
        )

        st.session_state["analyzer"] = analyzer
        st.session_state["retriever"] = analyzer.retrieval["retriever"]
        st.session_state["retriever_initialized"] = True
        st.session_state["rebuild_index"] = False

    st.success("✅ Fresh analyzer + retriever built (chunks & Qdrant rebuilt).")

    # show sample chunks
    st.subheader("📑 Sample Chunks")
    for i, c in enumerate(st.session_state["chunks"][:3]):
        st.markdown(f"**Chunk {i+1}** — {len(c.page_content)} chars")
        st.text(c.page_content[:400] + "...\n")
