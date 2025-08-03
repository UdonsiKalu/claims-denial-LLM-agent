import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from faiss_gpu_entropy import CMSDenialAnalyzer  # adjust this if needed
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Retrieval Studio: Entropy Map", layout="wide")
st.title("📊 Retrieval Studio: Entropy Map Generator")

# Sidebar configuration
st.sidebar.header("⚙️ Chunking & Index Settings")
chunk_size = st.sidebar.slider("Chunk Size", 1000, 20000, 10000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 5000, 2000, 200)
search_k = st.sidebar.slider("FAISS Top-k", 1, 20, 3)

# Manual exclusion configuration
st.sidebar.markdown("### 🔍 Exclude Pages by Pattern")
exclude_pages = st.sidebar.text_area(
    "Enter substrings or page numbers to filter out (comma-separated):",
    value="Page_0,Introduction,Scope,Purpose"
)
exclude_tokens = [token.strip().lower() for token in exclude_pages.split(",") if token.strip()]

if st.sidebar.button("🔁 Rechunk & Reindex"):
    st.session_state["rebuild_index"] = True

# Analyzer loader
@st.cache_resource(show_spinner=True)
def load_analyzer(chunk_size, chunk_overlap, exclude_tokens, force_rebuild=False):
    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
    os.environ["FORCE_REBUILD"] = "1" if force_rebuild else "0"
    os.environ["INDEX_PATH"] = f"faiss_index_c{chunk_size}_o{chunk_overlap}"
    return CMSDenialAnalyzer(exclude_tokens=exclude_tokens)

# Load analyzer
if "rebuild_index" not in st.session_state:
    st.session_state["rebuild_index"] = False

if st.session_state["rebuild_index"]:
    with st.spinner("Rebuilding and initializing retrieval engine..."):
        analyzer = load_analyzer(chunk_size, chunk_overlap, exclude_tokens, True)
        st.session_state["retriever"] = analyzer.retrieval["retriever"]
        st.success("Retriever reindexed!")
        st.session_state["rebuild_index"] = False
elif "retriever" not in st.session_state:
    with st.spinner("Initializing retrieval engine..."):
        analyzer = load_analyzer(chunk_size, chunk_overlap, exclude_tokens, False)
        st.session_state["retriever"] = analyzer.retrieval["retriever"]
        st.success("Retriever ready!")

retriever = st.session_state["retriever"]

# Upload claims
st.subheader("📁 Upload Synthetic Claims File")
claims_file = st.file_uploader("Upload JSONL file with claims", type="jsonl")

if claims_file:
    try:
        claims = [json.loads(line) for line in claims_file]
        st.success(f"Loaded {len(claims)} claims")
    except Exception as e:
        st.error(f"Error loading JSONL file: {e}")
        claims = []

    if st.button("🚀 Generate Entropy Map") and claims:
        with st.spinner("Running retrieval and tracking frequency..."):
            retrieval_log = defaultdict(int)
            for claim in tqdm(claims, desc="Claims"):
                try:
                    docs = retriever.invoke(claim["cpt_code"])
                    for doc in docs:
                        source = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "0")
                        chunk_id = f"{source}::Page_{page}"
                        retrieval_log[chunk_id] += 1
                except Exception as e:
                    print(f"Error on claim {claim.get('cpt_code', 'UNKNOWN')}: {e}")

            df = pd.DataFrame([
                {"chunk_id": k, "retrieval_count": v} for k, v in retrieval_log.items()
            ])
            df.sort_values(by="retrieval_count", ascending=False, inplace=True)
            st.session_state["entropy_df"] = df
            st.session_state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("Entropy map generated!")

# === DISPLAY ENTROPY RESULTS ===
if "entropy_df" in st.session_state:
    st.subheader("📈 Entropy Map Results")
    st.markdown(f"_Last generated: {st.session_state['last_run']}_")

    filter_val = st.slider(
    "📉 Filter: Show Only Frequently Retrieved Chunks",
    min_value=1, max_value=100, value=1,
    help="Only include chunks that were retrieved at least this many times across all claims. Helps hide rarely used or noisy chunks."
    )
    df = st.session_state["entropy_df"]

# Filter by retrieval count
    filtered_df = df[df["retrieval_count"] >= filter_val]

    # Also exclude by user-defined page patterns
    if exclude_tokens:
        filtered_df = filtered_df[
            ~filtered_df["chunk_id"].str.lower().apply(
                lambda cid: any(token in cid for token in exclude_tokens)
            )
        ]


    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Bar Chart (Top Chunks)")
        st.bar_chart(filtered_df.set_index("chunk_id"))

    with col2:
        st.markdown("#### 🗺️ Heatmap View")

        def parse_chunk_id(chunk_id):
            parts = chunk_id.split("::Page_")
            return parts[0], int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

        filtered_df[["source", "page"]] = filtered_df["chunk_id"].apply(
            lambda x: pd.Series(parse_chunk_id(x))
        )

        pivot = filtered_df.pivot_table(
            index="source", columns="page", values="retrieval_count", fill_value=0
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(pivot, annot=False, cmap="YlOrRd", ax=ax, linewidths=0.5)
        ax.set_title("Heatmap: Retrieval Frequency (Source vs Page)")
        ax.set_xlabel("Page Number")
        ax.set_ylabel("Source File")

        st.pyplot(fig)

    with st.expander("📄 View Table"):
        st.dataframe(filtered_df, use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Entropy CSV", csv, file_name="entropy_map.csv")



    # === PDF Upload and Manual Chunking Interface ===
    st.subheader("📄 Upload a CMS PDF for Transparent Chunking")

    pdf_file = st.file_uploader("Upload a CMS PDF file", type="pdf")
    chunking_strategy = st.selectbox("Choose Chunking Strategy", ["Fixed-size (default)", "Header-aware (coming soon)"])
    uploaded_chunk_size = st.slider("Manual Chunk Size", 500, 5000, 1000, 100)

    if pdf_file:
        st.info(f"Using **{chunking_strategy}** chunking method")

        if st.button("🔪 Chunk Uploaded PDF"):
            import fitz  # PyMuPDF
            from pathlib import Path

            temp_path = Path("temp_uploaded.pdf")
            temp_path.write_bytes(pdf_file.read())

            doc = fitz.open(str(temp_path))
            chunked_docs = []
            for page_num, page in enumerate(doc):
                text = page.get_text().strip()
                if not text:
                    continue
                # Fixed-size splitter
                chunks = [text[i:i+uploaded_chunk_size] for i in range(0, len(text), uploaded_chunk_size)]
                for idx, chunk in enumerate(chunks):
                    chunked_docs.append({
                        "chunk_id": f"{pdf_file.name}::Page_{page_num}_Chunk_{idx}",
                        "source": pdf_file.name,
                        "page": page_num,
                        "text": chunk
                    })

            st.success(f"Generated {len(chunked_docs)} chunks from uploaded PDF.")

            preview_df = pd.DataFrame([{
                "chunk_id": r["chunk_id"],
                "page": r["page"],
                "chars": len(r["text"]),
                "preview": r["text"][:200].replace("\n", " ")
            } for r in chunked_docs])
            st.dataframe(preview_df, use_container_width=True)

            # Download JSONL version
            jsonl_bytes = "\n".join(json.dumps(c) for c in chunked_docs).encode("utf-8")
            st.download_button("📥 Download Chunked JSONL", jsonl_bytes, file_name="uploaded_chunks.jsonl")

            # Optional index from uploaded content
            if st.checkbox("📚 Index these chunks for retrieval"):
                from langchain.vectorstores import FAISS
                from langchain.embeddings import HuggingFaceEmbeddings
                from langchain.schema import Document

                docs = [
                    Document(
                        page_content=chunk["text"],
                        metadata={"source": chunk["source"], "page": chunk["page"], "chunk_id": chunk["chunk_id"]}
                    ) for chunk in chunked_docs
                ]

                with st.spinner("Embedding and indexing chunks..."):
                    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vectorstore = FAISS.from_documents(docs, embedding=embed)
                    st.session_state["retriever"] = vectorstore.as_retriever(search_kwargs={"k": search_k})
                    st.success("Custom retriever created from uploaded PDF.")



    # === DIAGNOSTICS TAB ===
    st.subheader("🧠 Automated Diagnostics")
    recommendations = []

    top_chunks = filtered_df.head(5)
    if not top_chunks.empty:
        total = filtered_df["retrieval_count"].sum()
        top_total = top_chunks["retrieval_count"].sum()
        percent = (top_total / total) * 100

        if percent > 60:
            recommendations.append(
                f"⚠️ Top 5 chunks account for {percent:.1f}% of all retrievals. Consider decreasing chunk size or applying semantic splitting."
            )

        if any("unknown" in cid for cid in top_chunks["chunk_id"]):
            recommendations.append("📌 Some chunks are missing metadata (e.g., source name). Add source/page metadata to chunks.")

        if chunk_overlap < 1000:
            recommendations.append("🔁 Chunk overlap is low. Consider increasing to preserve context across boundaries.")

    if recommendations:
        for rec in recommendations:
            st.warning(rec)
    else:
        st.info("✅ No major entropy problems detected. Your retrieval distribution is balanced.")