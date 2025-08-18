# === TOP-LEVEL IMPORTS ===
import os
import sys
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# === MODULE IMPORTS ===
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from faiss_gpu_entropy import CMSDenialAnalyzer
import fitz  # PyMuPDF
from langchain_community.vectorstores.faiss import FAISS
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.schema import Document

# === STREAMLIT PAGE ===
st.set_page_config(page_title="Retrieval Studio", layout="wide")
st.title("📊 Retrieval Studio: Entropy Map & Chunking")

# === SIDEBAR SETTINGS ===
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
    st.session_state.pop("retriever", None)
    st.session_state.pop("analyzer", None)

# === LOAD ANALYZER ===
@st.cache_resource(show_spinner=True)
def load_analyzer(strategy, chunk_size, chunk_overlap, exclude_tokens, force_rebuild,
                  faiss_k, bm25_k, faiss_fetch_k, weights):

    # Set chunking and control strategy
    os.environ["CHUNK_STRATEGY"] = strategy
    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)

    # Force re-embedding / reindexing
    os.environ["FORCE_REBUILD"] = "1" if force_rebuild else "0"

    # Optional: dynamically name Qdrant collection (if you want separate ones per strategy)
    os.environ["QDRANT_COLLECTION"] = f"qdrant_{strategy}_c{chunk_size}_o{chunk_overlap}"

    # Load analyzer (which internally builds chunks + retriever)
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
#elif "retriever" not in st.session_state:
    #with st.spinner("Loading retrieval system..."):
   #    analyzer = load_analyzer(chunking_strategy, chunk_size, chunk_overlap, exclude_tokens, False,
    #                             faiss_k, bm25_k, faiss_fetch_k, weights)
    #    st.session_state["retriever"] = analyzer.retrieval["retriever"]
     #   st.session_state["analyzer"] = analyzer
     #   st.success("Retriever ready!")

retriever = st.session_state.get("retriever")

if not retriever:
    st.warning("⚠️ Please click **'🔁 Rechunk & Reindex'** to initialize the retrieval system.")
    st.stop()

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
                            "count": count
                        })
                token_freq_df = pd.DataFrame(token_data)
                st.session_state["token_freq_df"] = token_freq_df

# === ENTROPY MAP DISPLAY ===
if "entropy_df" in st.session_state:
    st.subheader("📈 Entropy Map Results")
    st.markdown(f"_Last run: {st.session_state['last_run']}_")

    filter_val = st.slider("📉 Filter: Min Retrieval Count", 1, 100, 1)
    df = st.session_state["entropy_df"]
    filtered_df = df[df["retrieval_count"] >= filter_val]

    if exclude_tokens:
        filtered_df = filtered_df[
            ~filtered_df["chunk_id"].str.lower().apply(
                lambda cid: any(tok in cid for tok in exclude_tokens)
            )
        ]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Bar Chart")
        st.bar_chart(filtered_df.set_index("chunk_id"))

    with col2:
        st.markdown("#### 🧭 Query vs Chunk Entropy Map")

        # Build matrix
        query_chunk_matrix = defaultdict(lambda: defaultdict(int))
        query_ids = []
        chunk_ids = set()

        for i, claim in enumerate(claims):
            query_id = f"Query_{i}"
            query_ids.append(query_id)
            try:
                docs = retriever.invoke(claim["cpt_code"])
                for doc in docs:
                    chunk_id = doc.metadata.get("chunk_id") or f"{doc.metadata.get('source', 'unknown')}::Page_{doc.metadata.get('page', '0')}"
                    
                    # 🧹 Skip if chunk matches any excluded token
                    if any(tok in chunk_id.lower() for tok in exclude_tokens):
                        continue

                    query_chunk_matrix[query_id][chunk_id] += 1
                    chunk_ids.add(chunk_id)
            except Exception as e:
                print(f"Error during matrix build on {claim.get('cpt_code')}: {e}")

        # Report excluded chunks (optional)
        excluded_chunks = set()
        for chunk_id in chunk_ids:
            if any(tok in chunk_id.lower() for tok in exclude_tokens):
                excluded_chunks.add(chunk_id)

        if excluded_chunks:
            st.info(f"🧹 {len(excluded_chunks)} chunks were excluded from the entropy matrix.")

        chunk_ids = sorted(chunk_ids)
        matrix_df = pd.DataFrame(0, index=query_ids, columns=chunk_ids)

        for q in query_ids:
            for c in query_chunk_matrix[q]:
                matrix_df.at[q, c] = query_chunk_matrix[q][c]

        # Optional: cap large values for better visualization
        matrix_display_df = matrix_df.copy()
        max_val = matrix_display_df.values.max()
        if max_val > 10:
            matrix_display_df = matrix_display_df.clip(upper=10)

        # Interactive matrix rendering with square cell sizes
        fig = px.imshow(
            matrix_display_df,
            labels=dict(x="Chunks", y="Queries", color="Retrieval Count"),
            color_continuous_scale="YlGnBu",
            aspect="auto",
            height=600,
        )

        fig.update_layout(
            title="Entropy Map: Query vs Chunk Matrix",
            xaxis=dict(
                title="Chunks",
                tickmode='linear',
                dtick=max(1, len(matrix_display_df.columns) // 10),
                tickangle=45,
            ),
            yaxis=dict(
                title="Queries",
                tickmode='linear',
                dtick=max(1, len(matrix_display_df.index) // 10),
            ),
            margin=dict(l=80, r=50, t=50, b=100),
        )

        if matrix_df.shape[0] < 30 and matrix_df.shape[1] < 30:
            fig.update_xaxes(showticklabels=True)
            fig.update_yaxes(showticklabels=True)

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📄 View Query-Chunk Matrix"):
            st.dataframe(matrix_df, use_container_width=True)

        csv_data = matrix_df.reset_index().rename(columns={"index": "Query"}).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Query vs Chunk Matrix CSV", csv_data, file_name="query_chunk_matrix.csv")




# === FREQUENCY MAP ===
if "token_freq_df" in st.session_state and not st.session_state["token_freq_df"].empty:
    st.subheader("🧠 Token Frequency Map")
    freq_df = st.session_state["token_freq_df"]
    token_to_view = st.selectbox("Select Token to View", freq_df["token"].unique())
    filtered_token_df = freq_df[freq_df["token"] == token_to_view]

    pivot_freq = filtered_token_df.pivot_table(
        index="source",
        columns="page",
        values="count",
        fill_value=0
    )

    st.markdown(f"### 🔍 Frequency Heatmap for: `{token_to_view}`")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot_freq, annot=False, cmap="Blues", ax=ax)
    ax.set_title(f"Frequency Map: '{token_to_view}'")
    ax.set_xlabel("Page")
    ax.set_ylabel("Source")
    st.pyplot(fig)
else:
    st.subheader("🧠 Token Frequency Map")
    st.info("Load a claims file and click '🚀 Generate Entropy Map' to compute token frequencies.")

# === DIAGNOSTICS ===
st.subheader("🧠 Automated Diagnostics")
recs = []
if "entropy_df" in st.session_state:
    top_chunks = st.session_state["entropy_df"].head(5)
    if not top_chunks.empty:
        total = st.session_state["entropy_df"]["retrieval_count"].sum()
        top_total = top_chunks["retrieval_count"].sum()
        percent = (top_total / total) * 100

        if percent > 60:
            recs.append(f"⚠️ Top 5 chunks account for {percent:.1f}% of all retrievals. Try semantic splitting.")
        if any("unknown" in cid for cid in top_chunks["chunk_id"]):
            recs.append("📌 Missing metadata: add `source` and `page` fields.")
        if chunk_overlap < 1000:
            recs.append("🔁 Chunk overlap is low. Consider increasing to preserve boundary context.")

if recs:
    for r in recs:
        st.warning(r)
else:
    st.info("✅ Retrieval distribution is healthy.")
