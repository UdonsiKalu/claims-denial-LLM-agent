import os
import sys
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from faiss_gpu_entropy import CMSDenialAnalyzer
from code_extractor import extract_codes_from_chunks

st.set_page_config(page_title="Retrieval Studio", layout="wide")
st.title("📊 Retrieval Studio: Entropy Map & Chunking")

# === SIDEBAR ===
st.sidebar.header("⚙️ Chunking Settings")
chunking_strategy = st.sidebar.selectbox("Chunking Strategy", ["Fixed-size", "Header-aware", "Semantic"])
chunk_size = st.sidebar.slider("Chunk Size", 1000, 20000, 10000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 5000, 2000, 200)

st.sidebar.header("🔧 Retriever Settings")
faiss_k = st.sidebar.slider("FAISS Top-k", 1, 20, 5)
bm25_k = st.sidebar.slider("BM25 Top-k", 1, 20, 3)
faiss_fetch_k = st.sidebar.slider("FAISS Fetch_k (MMR)", faiss_k, 100, 50)
faiss_weight = st.sidebar.slider("Weight: FAISS", 0.0, 1.0, 0.6, 0.05)
bm25_weight = 1.0 - faiss_weight
weights = (bm25_weight, faiss_weight)

st.sidebar.markdown("### 🔍 Exclude Chunks by Pattern")
exclude_pages = st.sidebar.text_area("Patterns to exclude (comma-separated)", "Page_0,Introduction,Scope,Purpose")
exclude_tokens = [x.strip().lower() for x in exclude_pages.split(",") if x.strip()]

if st.sidebar.button("🔁 Rechunk & Reindex"):
    st.session_state["rebuild_index"] = True

@st.cache_resource(show_spinner=True)
def load_analyzer(strategy, chunk_size, chunk_overlap, exclude_tokens, force_rebuild,
                  faiss_k, bm25_k, faiss_fetch_k, weights):
    os.environ["CHUNK_STRATEGY"] = strategy
    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
    os.environ["FORCE_REBUILD"] = "1" if force_rebuild else "0"
    os.environ["INDEX_PATH"] = f"faiss_index_{strategy}_c{chunk_size}_o{chunk_overlap}"
    return CMSDenialAnalyzer(
        exclude_tokens=exclude_tokens,
        faiss_k=faiss_k,
        bm25_k=bm25_k,
        faiss_fetch_k=faiss_fetch_k,
        weights=weights
    )

if "rebuild_index" not in st.session_state:
    st.session_state["rebuild_index"] = False

if st.session_state["rebuild_index"]:
    with st.spinner("Rebuilding retrieval system..."):
        analyzer = load_analyzer(chunking_strategy, chunk_size, chunk_overlap, exclude_tokens, True,
                                 faiss_k, bm25_k, faiss_fetch_k, weights)
        st.session_state["retriever"] = analyzer.retrieval["retriever"]
        st.session_state["analyzer"] = analyzer
        st.success("Reindexed!")
        st.session_state["rebuild_index"] = False
elif "retriever" not in st.session_state:
    with st.spinner("Loading retrieval system..."):
        analyzer = load_analyzer(chunking_strategy, chunk_size, chunk_overlap, exclude_tokens, False,
                                 faiss_k, bm25_k, faiss_fetch_k, weights)
        st.session_state["retriever"] = analyzer.retrieval["retriever"]
        st.session_state["analyzer"] = analyzer
        st.success("Retriever ready!")

retriever = st.session_state["retriever"]

# === CLAIM FILE UPLOAD ===
st.subheader("📁 Upload Synthetic Claims File")
claims_file = st.file_uploader("Upload JSONL file with claims", type="jsonl")
if claims_file:
    try:
        claims = [json.loads(line) for line in claims_file]
        st.success(f"Loaded {len(claims)} claims")
    except Exception as e:
        st.error(f"Failed to load claims: {e}")
        claims = []

    if st.button("🚀 Generate Entropy Map") and claims:
        retrieval_log = defaultdict(int)
        token_data = []

        with st.spinner("Running retrieval and frequency analysis..."):
            for claim in claims:
                try:
                    docs = retriever.invoke(claim["cpt_code"])
                    for doc in docs:
                        source = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "0")
                        chunk_id = f"{source}::Page_{page}"
                        retrieval_log[chunk_id] += 1
                except Exception as e:
                    print(f"Error on {claim.get('cpt_code')}: {e}")

            entropy_df = pd.DataFrame([
                {"chunk_id": k, "retrieval_count": v} for k, v in retrieval_log.items()
            ])
            entropy_df.sort_values(by="retrieval_count", ascending=False, inplace=True)
            st.session_state["entropy_df"] = entropy_df
            st.session_state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            analyzer = st.session_state.get("analyzer")
            if analyzer:
                analyzer.compute_token_frequencies()

                for chunk in analyzer.chunks:
                    freqs = chunk.metadata.get("token_frequencies", {})
                    for token, count in freqs.items():
                        token_data.append({
                            "source": chunk.metadata.get("source", "unknown"),
                            "page": int(chunk.metadata.get("page", 0)),
                            "token": token,
                            "count": count,
                            "chunk_id": f"{chunk.metadata.get('source', 'unknown')}::Page_{chunk.metadata.get('page', 0)}"
                        })

                token_freq_df = pd.DataFrame(token_data)
                st.session_state["token_freq_df"] = token_freq_df

# === INTERACTIVE ENTROPY HISTOGRAM ===
if "entropy_df" in st.session_state:
    st.subheader("📈 Entropy Map Results")
    st.markdown(f"_Last run: {st.session_state['last_run']}_")

    df = st.session_state["entropy_df"]
    chart_data = df.reset_index()

    selection = alt.selection_point(fields=["chunk_id"])

    bar = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("chunk_id:N", sort="-y"),
        y="retrieval_count:Q",
        tooltip=["chunk_id", "retrieval_count"],
        color=alt.condition(selection, alt.value("orange"), alt.value("steelblue"))
    ).properties(height=300).add_params(selection)

    st.altair_chart(bar, use_container_width=True)

    selected = None
    if selection and selection.name in st.session_state:
        selected = st.session_state[selection.name].get("chunk_id")
        if selected:
            st.session_state["selected_chunk"] = selected

# === FREQUENCY MAP FOR SELECTED CHUNK ===
if "token_freq_df" in st.session_state:
    st.subheader("🧠 Token Frequency Map")
    token_freq_df = st.session_state["token_freq_df"]
    chunk_id = st.session_state.get("selected_chunk")

    if chunk_id:
        chunk_df = token_freq_df[token_freq_df["chunk_id"] == chunk_id]
        if not chunk_df.empty:
            st.markdown(f"### 🔍 Frequency Heatmap for Selected Chunk: `{chunk_id}`")
            pivot = chunk_df.pivot_table(index="token", values="count", aggfunc="sum")
            fig, ax = plt.subplots(figsize=(6, max(2, len(pivot) // 2)))
            sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)
        else:
            st.info("No token frequency data for selected chunk.")
    else:
        st.info("Click a bar in the entropy histogram to view token breakdown.")
