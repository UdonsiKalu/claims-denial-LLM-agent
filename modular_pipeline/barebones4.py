# === TOP-LEVEL IMPORTS ===
import os
import streamlit as st
from langchain_ollama import OllamaEmbeddings  
from faiss_gpu_entropy import CMSDenialAnalyzer  # ✅ your analyzer

# === Default Embeddings (used by semantic chunker) ===
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# === STREAMLIT PAGE ===
st.set_page_config(page_title="Retrieval Studio (Test)", layout="wide")
st.title("Retrieval Studio: Chunking & Retriever Test")

# === STATE DEFAULTS ===
st.session_state.setdefault("frozen_params", {})
st.session_state.setdefault("rebuild_index", False)
st.session_state.setdefault("retriever_initialized", False)

# === STATUS PLACEHOLDER (so we can clear messages later) ===
status_box = st.empty()

# === STATUS INDICATOR ===
if st.session_state["rebuild_index"]:
    status_box.info("⏳ Rebuilding retriever... please wait.")
elif st.session_state["retriever_initialized"]:
    status_box.success("✅ Retriever initialized and ready.")
else:
    status_box.warning("❌ Retriever not yet initialized. Click **Rechunk & Reindex** to build.")

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

    # --- Rebuild Button ---
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


# === MAIN CONTENT ===
st.subheader("Current Frozen Params")
st.json(st.session_state["frozen_params"])

# === Build/Rebuild Analyzer ===
if st.session_state["rebuild_index"]:   # ✅ only rebuild when button pressed
    params = st.session_state.get("frozen_params", {})

    # Safe defaults in case params is empty
    chunking_strategy = params.get("chunking_strategy", "fixed")
    chunk_size = params.get("chunk_size", 512)
    chunk_overlap = params.get("chunk_overlap", 64)

    with st.spinner("🔄 Rebuilding CMSDenialAnalyzer with new settings..."):
        os.environ["FORCE_REBUILD"] = "1"
        collection_name = f"cms_{chunking_strategy}_c{chunk_size}_o{chunk_overlap}"
        os.environ["QDRANT_COLLECTION"] = collection_name

        analyzer = CMSDenialAnalyzer(
            exclude_tokens=params.get("exclude_tokens", []),
            faiss_k=params.get("faiss_k", 5),
            bm25_k=params.get("bm25_k", 3),
            faiss_fetch_k=params.get("faiss_fetch_k", 25),
            weights=params.get("weights", (0.5, 0.5)),
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            header_levels=params.get("header_levels", 3),
            semantic_threshold=params.get("semantic_threshold", 0.5),
            embeddings=embeddings
        )

        st.session_state["analyzer"] = analyzer
        st.session_state["retriever"] = analyzer.retrieval["retriever"]
        st.session_state["retriever_initialized"] = True
        st.session_state["rebuild_index"] = False

    # ✅ Overwrite the placeholder so temporary messages are cleared
    status_box.success("✅ Retriever initialized and ready.")


# === Quick Sanity Check ===
if st.session_state["retriever_initialized"]:
    st.info("Retriever is ready. You can now test queries via analyzer.")
