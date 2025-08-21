#barebones app
import streamlit as st

# === STATE DEFAULTS ===
st.session_state.setdefault("frozen_params", {
    "chunking_strategy": "Fixed-size",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "faiss_k": 5,
    "bm25_k": 3,
    "faiss_fetch_k": 20,
    "weights": (0.5, 0.5),
})

st.set_page_config(page_title="Chunking & Retriever Controls", layout="wide")
st.title("📦 Minimal Retrieval Studio")

with st.sidebar:
    st.header("⚙️ Fixed Chunking Settings")
    ui_chunk_size = st.number_input("Chunk Size", 128, 4096, st.session_state["frozen_params"]["chunk_size"], step=128)
    ui_chunk_overlap = st.number_input("Chunk Overlap", 0, 1024, st.session_state["frozen_params"]["chunk_overlap"], step=16)

    st.header("🔍 Retriever Settings")
    ui_faiss_k = st.slider("FAISS k", 1, 50, st.session_state["frozen_params"]["faiss_k"])
    ui_bm25_k = st.slider("BM25 k", 1, 20, st.session_state["frozen_params"]["bm25_k"])
    ui_faiss_fetch_k = st.slider("FAISS Fetch k", 10, 200, st.session_state["frozen_params"]["faiss_fetch_k"], step=10)
    ui_weights = st.slider("Ensemble Weights (FAISS vs BM25)", 0.0, 1.0, st.session_state["frozen_params"]["weights"])

    # 🚀 The ONLY trigger
    if st.button("🚀 Rechunk & Reindex", use_container_width=True):
        st.session_state["frozen_params"] = {
            "chunking_strategy": "Fixed-size",
            "chunk_size": ui_chunk_size,
            "chunk_overlap": ui_chunk_overlap,
            "faiss_k": ui_faiss_k,
            "bm25_k": ui_bm25_k,
            "faiss_fetch_k": ui_faiss_fetch_k,
            "weights": ui_weights,
        }
        st.success("✅ Parameters updated. Rechunking & Reindexing started!")

# === MAIN CONTENT ===
st.subheader("📋 Current Parameters")
st.json(st.session_state["frozen_params"])
