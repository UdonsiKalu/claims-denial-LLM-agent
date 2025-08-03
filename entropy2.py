import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from faiss_gpu_entropy import CMSDenialAnalyzer  # your backend analyzer

# Optional: For PDF chunking
import fitz  # PyMuPDF
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

st.set_page_config(page_title="Retrieval Studio", layout="wide")
st.title("📊 Retrieval Studio: Entropy Map & Chunking")

# === SIDEBAR ===
st.sidebar.header("⚙️ Chunking Settings")
chunking_strategy = st.sidebar.selectbox("Chunking Strategy", ["Fixed-size", "Header-aware", "Semantic"])
chunk_size = st.sidebar.slider("Chunk Size", 1000, 20000, 10000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 5000, 2000, 200)
search_k = st.sidebar.slider("FAISS Top-k", 1, 20, 3)

# Exclude rules
st.sidebar.markdown("### 🔍 Exclude Chunks by Pattern")
exclude_pages = st.sidebar.text_area("Patterns to exclude (comma-separated)", "Page_0,Introduction,Scope,Purpose")
exclude_tokens = [x.strip().lower() for x in exclude_pages.split(",") if x.strip()]

# Rebuild trigger
if st.sidebar.button("🔁 Rechunk & Reindex"):
    st.session_state["rebuild_index"] = True

# === LOAD ANALYZER ===
@st.cache_resource(show_spinner=True)
def load_analyzer(strategy, chunk_size, chunk_overlap, exclude_tokens, force_rebuild=False):
    os.environ["CHUNK_STRATEGY"] = strategy
    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
    os.environ["FORCE_REBUILD"] = "1" if force_rebuild else "0"
    os.environ["INDEX_PATH"] = f"faiss_index_{strategy}_c{chunk_size}_o{chunk_overlap}"
    return CMSDenialAnalyzer(exclude_tokens=exclude_tokens)

if "rebuild_index" not in st.session_state:
    st.session_state["rebuild_index"] = False

if st.session_state["rebuild_index"]:
    with st.spinner("Rebuilding retrieval system..."):
        analyzer = load_analyzer(chunking_strategy, chunk_size, chunk_overlap, exclude_tokens, True)
        st.session_state["retriever"] = analyzer.retrieval["retriever"]
        st.success("Reindexed!")
        st.session_state["rebuild_index"] = False
elif "retriever" not in st.session_state:
    with st.spinner("Loading retrieval system..."):
        analyzer = load_analyzer(chunking_strategy, chunk_size, chunk_overlap, exclude_tokens)
        st.session_state["retriever"] = analyzer.retrieval["retriever"]
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
        with st.spinner("Running retrieval over claims..."):
            for claim in tqdm(claims):
                try:
                    docs = retriever.invoke(claim["cpt_code"])
                    for doc in docs:
                        source = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "0")
                        chunk_id = f"{source}::Page_{page}"
                        retrieval_log[chunk_id] += 1
                except Exception as e:
                    print(f"Error on {claim.get('cpt_code')}: {e}")

        df = pd.DataFrame([
            {"chunk_id": k, "retrieval_count": v} for k, v in retrieval_log.items()
        ])
        df.sort_values(by="retrieval_count", ascending=False, inplace=True)
        st.session_state["entropy_df"] = df
        st.session_state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success("Entropy map generated!")

# === ENTROPY MAP ===
if "entropy_df" in st.session_state:
    st.subheader("📈 Entropy Map Results")
    st.markdown(f"_Last run: {st.session_state['last_run']}_")

    filter_val = st.slider(
        "📉 Filter: Min Retrieval Count",
        1, 100, 1,
        help="Only show chunks retrieved this many times or more."
    )

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
        st.markdown("#### 🗺️ Heatmap")

        def parse_chunk_id(chunk_id):
            parts = chunk_id.split("::Page_")
            return parts[0], int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

        filtered_df[["source", "page"]] = filtered_df["chunk_id"].apply(
            lambda x: pd.Series(parse_chunk_id(x))
        )
        pivot = filtered_df.pivot_table(index="source", columns="page", values="retrieval_count", fill_value=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(pivot, annot=False, cmap="YlOrRd", ax=ax)
        ax.set_title("Heatmap: Retrieval Frequency")
        ax.set_xlabel("Page")
        ax.set_ylabel("Source")
        st.pyplot(fig)

    with st.expander("📄 View Data Table"):
        st.dataframe(filtered_df, use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, file_name="entropy_map.csv")

# === PDF UPLOAD AND MANUAL CHUNKING ===
st.subheader("📄 Upload and Chunk a CMS PDF")

pdf_file = st.file_uploader("Upload CMS PDF", type="pdf")
manual_chunk_size = st.slider("Manual Chunk Size (Fixed only)", 500, 5000, 1000, 100)

if pdf_file:
    temp_path = Path("temp_uploaded.pdf")
    temp_path.write_bytes(pdf_file.read())
    doc = fitz.open(str(temp_path))
    chunked_docs = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if not text:
            continue
        chunks = [text[i:i + manual_chunk_size] for i in range(0, len(text), manual_chunk_size)]
        for idx, chunk in enumerate(chunks):
            chunked_docs.append({
                "chunk_id": f"{pdf_file.name}::Page_{page_num}_Chunk_{idx}",
                "source": pdf_file.name,
                "page": page_num,
                "text": chunk
            })

    st.success(f"Chunked into {len(chunked_docs)} segments")

    preview_df = pd.DataFrame([{
        "chunk_id": r["chunk_id"],
        "page": r["page"],
        "chars": len(r["text"]),
        "preview": r["text"][:150].replace("\n", " ")
    } for r in chunked_docs])
    st.dataframe(preview_df, use_container_width=True)

    jsonl_bytes = "\n".join(json.dumps(r) for r in chunked_docs).encode("utf-8")
    st.download_button("📥 Download JSONL", jsonl_bytes, file_name="chunks_preview.jsonl")

    if st.checkbox("📚 Index these chunks now"):
        docs = [
            Document(
                page_content=chunk["text"],
                metadata={"chunk_id": chunk["chunk_id"], "page": chunk["page"], "source": chunk["source"]}
            ) for chunk in chunked_docs
        ]
        with st.spinner("Embedding & indexing..."):
            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embedding=embed)
            st.session_state["retriever"] = vectorstore.as_retriever(search_kwargs={"k": search_k})
            st.success("🔎 Custom retriever ready!")

# === DIAGNOSTICS ===
st.subheader("🧠 Automated Diagnostics")
recs = []

if "entropy_df" in st.session_state:
    top_chunks = filtered_df.head(5)
    if not top_chunks.empty:
        total = filtered_df["retrieval_count"].sum()
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
