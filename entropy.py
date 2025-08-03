import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from faiss_gpu import CMSDenialAnalyzer  # make sure this is your correct filename/module

st.set_page_config(page_title="Retrieval Studio: Entropy Map", layout="wide")
st.title("📊 Retrieval Studio: Entropy Map Generator")

# Sidebar configuration
st.sidebar.header("⚙️ Chunking & Index Settings")
chunk_size = st.sidebar.slider("Chunk Size", 1000, 20000, 10000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 5000, 2000, 200)
search_k = st.sidebar.slider("FAISS Top-k", 1, 20, 3)

if st.sidebar.button("🔁 Rechunk & Reindex"):
    st.session_state["rebuild_index"] = True

# Analyzer section
@st.cache_resource(show_spinner=True)
def load_analyzer(chunk_size, chunk_overlap, force_rebuild=False):
    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
    os.environ["FORCE_REBUILD"] = "1" if force_rebuild else "0"
    return CMSDenialAnalyzer()

# Load analyzer with current settings
if "rebuild_index" not in st.session_state:
    st.session_state["rebuild_index"] = False

with st.spinner("Initializing retrieval engine..."):
    analyzer = load_analyzer(chunk_size, chunk_overlap, st.session_state["rebuild_index"])
    retriever = analyzer.retrieval["retriever"]
    st.success("Retriever ready!")
    st.session_state["rebuild_index"] = False

# Upload claims
st.subheader("📁 Upload Synthetic Claims File")
claims_file = st.file_uploader("Upload JSONL file with claims", type=["jsonl"])

if claims_file is not None:
    try:
        claims = [json.loads(line) for line in claims_file]
        st.success(f"Loaded {len(claims)} claims")

        if st.button("🚀 Generate Entropy Map"):
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
                        print(f"Error on claim {claim['cpt_code']}: {e}")

                # Create entropy dataframe
                df = pd.DataFrame([
                    {"chunk_id": k, "retrieval_count": v} for k, v in retrieval_log.items()
                ])
                df.sort_values(by="retrieval_count", ascending=False, inplace=True)
                st.session_state["entropy_df"] = df
                st.session_state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("Entropy map generated!")
    except Exception as e:
        st.error(f"Error loading JSONL file: {e}")

# Display results
if "entropy_df" in st.session_state:
    st.subheader("📈 Entropy Map Results")
    st.markdown(f"_Last generated: {st.session_state['last_run']}_")

    filter_val = st.slider("Minimum retrieval count to show", 1, 100, 1)
    filtered_df = st.session_state["entropy_df"][
        st.session_state["entropy_df"]["retrieval_count"] >= filter_val
    ]

    st.bar_chart(filtered_df.set_index("chunk_id"))

    with st.expander("📄 View Table"):
        st.dataframe(filtered_df, use_container_width=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("📅 Download Entropy CSV", csv, file_name="entropy_map.csv")
