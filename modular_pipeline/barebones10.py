# === TOP-LEVEL IMPORTS ===
import os
import shutil
import streamlit as st
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
st.session_state.setdefault("log_lines", [])
st.session_state.setdefault("last_live_config", {})

# === HUD HELPERS ===
def pretty_config(cfg: dict) -> str:
    """Format config dict into HUD-style readable block."""
    lines = []
    for k, v in cfg.items():
        if isinstance(v, (list, tuple)):
            v_str = ", ".join(str(x) for x in v)
            lines.append(f"{k:<20}: [{v_str}]")
        else:
            lines.append(f"{k:<20}: {v}")
    return "\n".join(lines)

def box_text(title: str, content: str) -> str:
    """Wrap text in ASCII HUD box."""
    border = "=" * 60
    return f"{border}\n{title}\n{border}\n{content}\n{border}"

# === LOGGING TERMINAL HELPER ===
def append_log(message, box=False):
    """Append log lines into terminal view."""
    if box:
        line = f"{message}"
    else:
        line = message

    st.session_state["log_lines"].append(line)
    log_box.markdown(
        f"""
        <div style="
            background-color:#000000;
            color:#00FF00;
            padding:1rem;
            font-family:monospace;
            height:500px;
            overflow-y:auto;
            border-radius:8px;
            white-space: pre-wrap;
        ">{'<br>'.join(st.session_state['log_lines'][-80:])}</div>
        """,
        unsafe_allow_html=True,
    )

def log_config_changes(live_params):
    prev = st.session_state.get("last_live_config", {})
    for k, v in live_params.items():
        if prev.get(k) != v:
            append_log(f"{k} → {v}")
    st.session_state["last_live_config"] = live_params.copy()

# === STATUS PLACEHOLDER ===
status_box = st.empty()
log_box = st.empty()
progress_bar = st.progress(0)

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
        st.session_state["config_history"].append({"params": frozen})
        st.session_state["frozen_params"] = frozen
        st.session_state["rebuild_index"] = True
        st.rerun()

    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.clear()
        st.experimental_rerun()

    if st.button("🗑️ Clear Disk Cache", use_container_width=True):
        shutil.rmtree(".cache/chunks", ignore_errors=True)
        os.makedirs(".cache/chunks", exist_ok=True)
        st.success("Disk cache cleared!")

    # --- Config History Log ---
    with st.expander("📜 Config History", expanded=False):
        if not st.session_state["config_history"]:
            st.info("No configs applied yet.")
        else:
            for entry in reversed(st.session_state["config_history"]):
                st.json(entry["params"])

# === LIVE CONFIG TRACKING ===
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
log_config_changes(live_params)

# === Build/Rebuild Analyzer ===
if st.session_state["rebuild_index"]:
    params = st.session_state["frozen_params"]

    os.environ["FORCE_REBUILD"] = "1"
    append_log(box_text("Rebuild Triggered", "Starting full pipeline..."), box=True)

    # --- STEP 1: RECHUNKING ---
    append_log(box_text("Step 1: Rechunking", "Splitting documents..."), box=True)
    progress_bar.progress(15)
    chunks = load_and_chunk_manuals(
        chunking_strategy=params.get("chunking_strategy", "Fixed-size"),
        chunk_size=params.get("chunk_size", 1000),
        chunk_overlap=params.get("chunk_overlap", 200),
        header_levels=params.get("header_levels"),
        semantic_threshold=params.get("semantic_threshold", 0.5),
        embeddings=embeddings,
    )
    if chunks and isinstance(chunks[0], dict):
        chunks = [Document(page_content=c["content"], metadata=c["metadata"]) for c in chunks]
    append_log(f"Chunks created: {len(chunks)}")
    st.session_state["chunks"] = chunks
    progress_bar.progress(40)

    # --- STEP 2: EMBEDDING ---
    append_log(box_text("Step 2: Embedding", "Generating embeddings..."), box=True)
    append_log("Ollama embeddings ready.")
    progress_bar.progress(70)

    # --- STEP 3: REINDEXING ---
    append_log(box_text("Step 3: Reindexing", "Building FAISS/Qdrant index..."), box=True)
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
    progress_bar.progress(100)

    # --- SUMMARY ---
    append_log(box_text("Final Frozen Config", pretty_config(params)), box=True)

    chunk_lines = []
    for i, c in enumerate(chunks[:3]):
        chunk_lines.append(f"Chunk {i+1} ({len(c.page_content)} chars)\n{c.page_content[:200].replace('\n',' ')}...")
    append_log(box_text("Sample Chunks", "\n\n".join(chunk_lines)), box=True)

    st.success("✅ Fresh analyzer + retriever built.")
