# === TOP-LEVEL IMPORTS ===
import os
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


    # === Preview Chunks After Rebuild ===
    if "analyzer" in st.session_state:
        analyzer = st.session_state["analyzer"]
        st.markdown("### 📑 Chunk Preview")

        st.success(f"Loaded {len(analyzer.chunks)} chunks "
                   f"(strategy={analyzer.chunking_strategy}, "
                   f"size={analyzer.chunk_size}, overlap={analyzer.chunk_overlap})")

        # Show first 3 chunks as a sanity check
        for i, chunk in enumerate(analyzer.chunks[:3]):
            st.text_area(
                f"Chunk {i+1} (source={chunk.metadata.get('source', 'unknown')}, page={chunk.metadata.get('page', 'N/A')})",
                chunk.page_content[:500],  # show first 500 characters
                height=150
            )


# === MAIN CONTENT ===
st.subheader("Current Frozen Params")
st.json(st.session_state["frozen_params"])

# === Build/Rebuild Analyzer ===
if st.session_state["rebuild_index"] or not st.session_state["retriever_initialized"]:
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
