# ✅ REVISED: Retrieval Studio with Frozen Params and Button-Gated Reloading (Text Inputs)

# === TOP-LEVEL IMPORTS ===
import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from strategy_optimizer import grid_search, random_search, bayesian_search
from faiss_gpu_entropy import CMSDenialAnalyzer
from strategy_profiler import (profile_objective,suggest_weights_from_variance,dynamic_objective,dynamic_objective_banded,)
from qdrant_client import QdrantClient
from retriever_qdrant import create_retrievers  # ⬅️ must import this
import inspect

# === GET DEFAULTS FROM create_retrievers ===
defaults = inspect.signature(create_retrievers).parameters
faiss_k_default = defaults["faiss_k"].default
bm25_k_default = defaults["bm25_k"].default
faiss_fetch_k_default = defaults["faiss_fetch_k"].default
weights_default = defaults["weights"].default

# === SIDEBAR CONTROLS ===
st.sidebar.header("Retriever Settings")
faiss_k = st.sidebar.slider("FAISS k", 1, 100, faiss_k_default)
bm25_k = st.sidebar.slider("BM25 k", 1, 100, bm25_k_default)
faiss_fetch_k = st.sidebar.slider("FAISS fetch_k", 1, 200, faiss_fetch_k_default)
weights = (
    st.sidebar.number_input("BM25 weight", 0.0, 1.0, weights_default[0], step=0.1),
    st.sidebar.number_input("FAISS weight", 0.0, 1.0, weights_default[1], step=0.1),
)

if st.sidebar.checkbox("Force Rebuild Vector DB", key="force_rebuild"):
    if not st.session_state.get("db_rebuilt", False):
        client = QdrantClient(host="localhost", port=6333)
        try:
            client.delete_collection("cms_chunks")
            st.warning("Deleted 'cms_chunks'. Will rebuild on next run.")
        except Exception as e:
            st.error(f"Could not delete collection: {e}")
        st.session_state.db_rebuilt = True



# === Define filter_chunks function ===
def filter_chunks(claims, exclude_tokens):
    """
    Filters out chunks that match any of the exclude tokens.
    Args:
    - claims: List of claims data
    - exclude_tokens: List of tokens to exclude from the analysis
    
    Returns:
    - Filtered list of claims excluding the specified tokens
    """
    filtered_claims = []
    
    for claim in claims:
        # Assume 'chunks' is a list in each claim that contains chunk IDs
        filtered_chunks = [
            chunk for chunk in claim.get('chunks', [])
            if not any(excluded_token in chunk.lower() for excluded_token in exclude_tokens)
        ]
        # If the claim has valid chunks left after filtering, add it to the result
        if filtered_chunks:
            claim['chunks'] = filtered_chunks  # Replace with filtered chunks
            filtered_claims.append(claim)
    
    return filtered_claims



# === Strategy Visualization Function ===

def plot_optimizer_results(df, bubble_cols=("score","final_score","query_entropy","improvement"),
                           normalize_if_negative=True):
    st.subheader("📊 Strategy Scores")
    fig_bar = px.bar(
        df,
        x="strategy_id",
        y="score",
        title="Score per Strategy",
        labels={"strategy_id": "Strategy ID", "score": "Score"},
        text="score",
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_bar.update_layout(yaxis_range=[0, 1], height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- pick a bubble-size column (falls back to None if not present)
    bubble_col = next((c for c in bubble_cols if c in df.columns), None)

    def _safe_sizes(series: pd.Series):
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if normalize_if_negative and s.min() < 0:
            s = s - s.min() + 1e-6  # shift up only if negatives exist
        # cap outliers to avoid giant bubbles
        s = np.clip(s, 0, np.nanpercentile(s, 99))
        # avoid all-zeros (invisible)
        if np.allclose(s, 0):
            s = np.ones_like(s)
        return s

    # === Entropy vs Coverage Plot ===
    st.subheader("🎯 Entropy vs Coverage")
    if {"entropy", "coverage"}.issubset(df.columns):
        plot_df = df.copy()
        if bubble_col:
            plot_df["_size"] = _safe_sizes(plot_df[bubble_col])
            fig_scatter = px.scatter(
                plot_df,
                x="entropy",
                y="coverage",
                color="score" if "score" in plot_df.columns else None,
                size="_size",
                hover_data=["strategy_id"] if "strategy_id" in plot_df.columns else plot_df.columns,
                color_continuous_scale="Viridis",
                title="Entropy vs Coverage (bubble size ~ normalized metric)",
                labels={"entropy": "Entropy", "coverage": "Coverage"},
            )
            fig_scatter.update_traces(marker=dict(sizemode="area"), selector=dict(mode="markers"))
            fig_scatter.update_traces(marker_sizemode="area", marker_sizemin=4, marker_sizeref=None)
        else:
            fig_scatter = px.scatter(
                plot_df,
                x="entropy",
                y="coverage",
                color="score" if "score" in plot_df.columns else None,
                hover_data=["strategy_id"] if "strategy_id" in plot_df.columns else plot_df.columns,
                color_continuous_scale="Viridis",
                title="Entropy vs Coverage",
                labels={"entropy": "Entropy", "coverage": "Coverage"},
            )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # === 📉 Gini vs Score Plot ===
    st.subheader("📉 Gini vs Score")
    if {"gini","score"}.issubset(df.columns):
        fig_gini = px.scatter(
            df,
            x="gini",
            y="score",
            color="score",
            hover_data=["strategy_id"] if "strategy_id" in df.columns else df.columns,
            title="Gini Coefficient vs Score",
            labels={"gini": "Gini Coefficient", "score": "Score"},
            color_continuous_scale="Plasma"
        )
        fig_gini.update_layout(height=400)
        st.plotly_chart(fig_gini, use_container_width=True)

    # === 🔥 Strategy Score Heatmap ===
    st.subheader("🔥 Strategy Score Heatmap")
    if "score" in df.columns and "strategy_id" in df.columns:
        heat_df = df[["strategy_id", "score"]].copy()
        heat_df["score_normalized"] = (
            (heat_df["score"] - heat_df["score"].min()) /
            (heat_df["score"].max() - heat_df["score"].min() + 1e-6)
        )
        fig_heatmap = px.imshow(
            [heat_df["score_normalized"].tolist()],
            labels=dict(x="Strategy ID", color="Score"),
            x=heat_df["strategy_id"].tolist(),
            y=["Score Intensity"],
            color_continuous_scale="Inferno",
            aspect="auto",
        )
        fig_heatmap.update_layout(height=200, xaxis_showgrid=False, yaxis_showgrid=False)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # === 📊 Max Chunk Frequency vs Score ===
    st.subheader("📊 Max Chunk Frequency vs Score")
    if {"max_freq","score"}.issubset(df.columns):
        fig_maxfreq = px.scatter(
            df,
            x="max_freq",
            y="score",
            color="score",
            hover_data=["strategy_id"] if "strategy_id" in df.columns else df.columns,
            title="Max Chunk Frequency vs Score",
            labels={"max_freq": "Max Chunk Frequency", "score": "Score"},
            color_continuous_scale="Cividis"
        )
        fig_maxfreq.update_layout(height=400)
        st.plotly_chart(fig_maxfreq, use_container_width=True)



# === MODULE IMPORTS ===
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from retrieval_optimizer import optimize_retrieval  # ✅ REQUIRED FOR OPTIMIZER MODE
import fitz  # PyMuPDF
from langchain_community.vectorstores.faiss import FAISS
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.schema import Document

# === STREAMLIT PAGE ===
st.set_page_config(page_title="Retrieval Studio", layout="wide")
st.title("Retrieval Studio: Entropy Map & Chunking")

# === Logging State Initialization ===
if "log_lines" not in st.session_state:
    st.session_state["log_lines"] = []

# === Add placeholder containers ===
progress_container = st.container()
log_container = st.container()

# === Progress Bar in its own container ===
with progress_container:
    progress_bar = st.progress(0.0, text="Starting optimization...")

# === Define log_box only now (AFTER progress bar) ===
with log_container:
    log_box = st.empty()
    st.session_state["log_box"] = log_box

# === Logging Function ===
def append_log(text):
    st.session_state["log_lines"].append(text)
    log_text = "\n".join(st.session_state["log_lines"])
    
    # Improved terminal styling
    st.session_state["log_box"].markdown(
        f"""
        <div style="
            background-color: #0a0a0a;
            color: #00ff00;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            border-radius: 0.5rem;
            border: 1px solid #00aa00;
            height: 400px;
            overflow-y: auto;
            white-space: pre;
            line-height: 1.3;
        ">
            {log_text}
        </div>
        """,
        unsafe_allow_html=True
    )

if "log_lines" not in st.session_state:
    st.session_state["log_lines"] = []

def append_log(text):
    st.session_state["log_lines"].append(text)
    log_text = "\n".join(st.session_state["log_lines"])
    st.session_state["log_box"].markdown(
        f"""
        <div style="background-color:#000000; color:#00FF00; padding:1em; font-family:monospace;
                    height:400px; overflow-y:scroll; border-radius:8px;">
            <pre>{log_text}</pre>
        </div>
        """,
        unsafe_allow_html=True
    )

# === STATE DEFAULTS ===
st.session_state.setdefault("retriever_initialized", False)
st.session_state.setdefault("rebuild_index", False)

# === FROZEN PARAMS ===
if not st.session_state["retriever_initialized"]:
    st.session_state["frozen_params"] = {
        "chunking_strategy": "Fixed-size",
        "chunk_size": 10000,
        "chunk_overlap": 2000,
        "faiss_k": 5,
        "bm25_k": 3,
        "faiss_fetch_k": 50,
        "weights": (0.4, 0.6),
        "exclude_tokens": ["page_0", "introduction", "scope", "purpose"]
    }

# === SIDEBAR UI ===
with st.sidebar:
    st.header("Configuration Panel")

    # Mode Switch
    mode = st.radio("🔀 App Mode", ["Manual", "Optimizer"], horizontal=True)
    st.session_state["app_mode"] = mode

    if mode == "Manual":
        with st.expander("Chunking Settings", expanded=True):
            ui_chunking_strategy = st.selectbox("Chunking Strategy", ["Fixed-size", "Header-aware", "Semantic"])
            ui_chunk_size = st.number_input("Chunk Size", min_value=1000, max_value=20000, value=10000, step=500)
            ui_chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=5000, value=2000, step=100)

        with st.expander("Retriever Settings", expanded=True):
            ui_faiss_k = st.number_input("FAISS Top-k", min_value=1, max_value=20, value=5)
            ui_bm25_k = st.number_input("BM25 Top-k", min_value=1, max_value=20, value=3)
            ui_faiss_fetch_k = st.number_input("FAISS Fetch_k (MMR)", min_value=ui_faiss_k, max_value=100, value=50)
            ui_faiss_weight = st.number_input("Weight: FAISS", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
            ui_weights = (1.0 - ui_faiss_weight, ui_faiss_weight)

        with st.expander("Exclusion Settings"):
            exclude_pages = st.text_area("Patterns to exclude (comma-separated)", "Page_0,Introduction,Scope,Purpose")
            exclude_tokens = [x.strip().lower() for x in exclude_pages.split(",") if x.strip()]
            st.session_state["exclude_tokens_runtime"] = exclude_tokens

        if st.button("🚀 Rechunk & Reindex", use_container_width=True):
            st.session_state["frozen_params"] = {
                "chunking_strategy": ui_chunking_strategy,
                "chunk_size": ui_chunk_size,
                "chunk_overlap": ui_chunk_overlap,
                "faiss_k": ui_faiss_k,
                "bm25_k": ui_bm25_k,
                "faiss_fetch_k": ui_faiss_fetch_k,
                "weights": ui_weights,
                "exclude_tokens": exclude_tokens
            }
            st.session_state["rebuild_index"] = True
            st.rerun()

# === MAIN CONTENT ===
if mode == "Optimizer":
    st.header("🤖 Retrieval Strategy Optimizer")
    st.markdown("Evaluate multiple strategies to match a defined entropy and coverage profile.")

    # === File Upload ===
    claims_file = st.file_uploader("Upload Synthetic Claims (JSONL)", type="jsonl", key="opt_claims")

    claims = []
    if claims_file:
        try:
            claims = [json.loads(line) for line in claims_file]
            st.success(f"✅ Loaded {len(claims)} claims")
        except Exception as e:
            st.error(f"❌ Failed to parse file: {e}")
            claims = []

    # === Target Profile Sliders (Always Visible) ===
    st.subheader("🎯 Define Your Retrieval Target Profile")
    target_profile = {
        "query_entropy_range": st.slider("Query Entropy Range", 0.0, 1.0, (0.7, 0.9)),
        "max_chunk_frequency": st.slider("Max Chunk Frequency", 0.01, 0.5, 0.1),
        "gini_threshold": st.slider("Gini Coefficient", 0.0, 1.0, 0.4),
        "required_code_coverage": st.slider("Code Coverage Threshold", 0.0, 1.0, 0.95)
    }

    # === Choose Search Method ===
    search_method = st.selectbox("🔍 Select Optimization Method", ["Grid Search", "Random Search", "Bayesian Optimization"])

    if search_method == "Random Search":
        n_iter = st.slider("Random Iterations", 5, 50, 10)
    elif search_method == "Bayesian Optimization":
        n_calls = st.slider("Bayesian Calls", 5, 30, 15)

    # ✅ Define the param grid (weights must be strings)
    param_grid = {
        "chunk_size": [5000, 7500, 10000, 15000],
        "chunk_overlap": [500, 1000, 2000, 3000],
        "faiss_k": list(range(3, 8)),
        "bm25_k": list(range(2, 6)),
        "fetch_k": [25, 50],
        "weights": ["0.4,0.6", "0.5,0.5", "0.6,0.4"]
    }

    # === Run Optimizer ===
    if st.button("🚀 Run Optimizer") and claims:
        st.session_state["log_lines"] = []  # Clear logs
        append_log("🚀 Starting optimization run...\n")

        # Display excluded chunk information
        excluded_chunks = len([claim for claim in claims if not any(excluded_token in chunk.lower() for chunk in claim.get('chunks', []) for excluded_token in exclude_tokens)])
        st.write(f"Excluding {excluded_chunks} chunks based on the provided tokens.")

        with st.spinner("Evaluating strategy combinations..."):

            def evaluate_strategy(params, claims, target_profile, idx=None, total=None):
                # --- progress log
                if idx is not None and total is not None:
                    append_log(f"⚙️ Evaluating strategy {idx}/{total}: {params}")

                # 1) Parse weights safely
                raw_weights = params["weights"]
                if isinstance(raw_weights, str):
                    weights = tuple(map(float, raw_weights.split(",")))
                else:
                    weights = raw_weights

                # 2) Cast chunking params and set env so analyzer rebuilds index & uses unique collection
                chunk_size = int(params.get("chunk_size", 10000))
                chunk_overlap = int(params.get("chunk_overlap", 2000))
                os.environ["CHUNK_STRATEGY"] = "Fixed-size"  # or whatever you support
                os.environ["CHUNK_SIZE"] = str(chunk_size)
                os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
                os.environ["FORCE_REBUILD"] = "1"  # ensure reindex per strategy
                os.environ["QDRANT_COLLECTION"] = f"qdrant_c{chunk_size}_o{chunk_overlap}"

                # 3) Strategy id (for display)
                strategy_id = (
                    f"faiss{params['faiss_k']}_bm25{params['bm25_k']}"
                    f"_fetch{params['fetch_k']}_w{params['weights']}"
                    f"_c{chunk_size}_o{chunk_overlap}"
                )
                params["strategy_id"] = strategy_id

                # 4) (Optional) apply runtime excludes to CLAIMS (not required; keep if you really use it)
                exclude_tokens_rt = st.session_state.get("exclude_tokens_runtime", [])
                filtered_claims = claims  # default: use original claims
                # If you truly need to drop claims based on some token logic, do it here.
                # Otherwise leave as-is to ensure all queries run every time.

                # 5) Build analyzer for this strategy (this triggers chunking + Qdrant rebuild per env)
                analyzer = CMSDenialAnalyzer(
                    exclude_tokens=[],            # analyzer-level content filtering; keep [] unless you need it
                    faiss_k=int(params["faiss_k"]),
                    bm25_k=int(params["bm25_k"]),
                    faiss_fetch_k=int(params["fetch_k"]),
                    weights=weights
                )

                # === Override retriever with sidebar params ===
                analyzer.retrieval["retriever"] = create_retrievers(
                    analyzer.retrieval["vectorstore"],
                    analyzer.chunks,
                    faiss_k=faiss_k,
                    bm25_k=bm25_k,
                    faiss_fetch_k=faiss_fetch_k,
                    weights=weights
                )

                # === Run Entropy Analysis ===
                results = analyzer.evaluate_entropy_score(claims, target_profile)


                # 6) ACTUALLY RUN ALL CLAIM QUERIES for this strategy
                append_log(f"🔎 Running queries for {len(filtered_claims)} claims...")
                result = analyzer.evaluate_entropy_score(
                    filtered_claims,
                    target_profile,
                    log_fn=append_log,
                    idx=idx,
                    total=total,
                )

                # 7) Extra logging
                append_log(
                    "📊 Strategy metrics: "
                    f"entropy={result.get('entropy'):.4f}, "
                    f"max_freq={result.get('max_freq'):.4f}, "
                    f"gini={result.get('gini'):.4f}, "
                    f"coverage={result.get('coverage'):.4f}, "
                    f"final_score={result.get('score'):.4f}"
                )

                # 8) Make sure the strategy_id is visible on the result
                result["strategy_id"] = strategy_id
                result["chunk_size"] = chunk_size
                result["chunk_overlap"] = chunk_overlap
                return result



            def apply_best_config_to_session(config):
                st.session_state["frozen_params"] = {
                    "chunking_strategy": "Fixed-size",
                    "chunk_size": config.get("chunk_size", 10000),
                    "chunk_overlap": config.get("chunk_overlap", 2000),
                    "faiss_k": config["faiss_k"],
                    "bm25_k": config["bm25_k"],
                    "faiss_fetch_k": config["fetch_k"],
                    "weights": tuple(map(float, config["weights"].split(","))),
                    "exclude_tokens": st.session_state.get("exclude_tokens_runtime", [])
                }
                st.session_state["rebuild_index"] = True


#############################################################################
# === Inside the Optimizer mode ===
        if mode == "Optimizer":
            # === Optimizer section ===
            st.header("🤖 Retrieval Strategy Optimizer")
            st.markdown("Evaluate multiple strategies to match a defined entropy and coverage profile.")

            # Initialize full_history as an empty list
            full_history = []


            # ---- PROFILING & DYNAMIC OBJECTIVE SELECTION ----
            append_log("🧪 Running pre-optimization profiling...")

            def score_fn_raw(params):
                # Single eval (no idx/total). We reuse your evaluate_strategy to compute metrics.
                return evaluate_strategy(params, claims, target_profile)

            # Quick probe over the declared param_grid
            profile = profile_objective(param_grid, score_fn_raw, n=12)
            append_log(f"📊 Profiling stats:\n{json.dumps(profile['stats'], indent=2)}")

            # Decide which scorer to use for the optimizer calls
            if profile.get("score_flat"):
                append_log("⚠️ Detected flat objective. Switching to data-driven dynamic objective.")
                # Suggest weights from variance of component metrics
                weights_suggested = suggest_weights_from_variance(profile["stats"])
                append_log(f"🧭 Suggested metric weights: {json.dumps(weights_suggested, indent=2)}")

                USE_BANDED = True  # set False to use simple dynamic_objective
                entropy_band = target_profile.get("query_entropy_range", (0.7, 0.9))

                def scorer_for_optimizer(p, i=None, t=None):
                    r = evaluate_strategy(p, claims, target_profile, idx=i, total=t)
                    # Replace r['score'] with a dynamic one:
                    if USE_BANDED:
                        r["score"] = dynamic_objective_banded(r, weights_suggested, entropy_band)
                    else:
                        r["score"] = dynamic_objective(r, weights_suggested)
                    return r
            else:
                append_log("✅ Objective shows usable variance. Proceeding with existing score.")
                def scorer_for_optimizer(p, i=None, t=None):
                    return evaluate_strategy(p, claims, target_profile, idx=i, total=t)
            # ---- END PROFILING ----


            # Run the optimization process based on the selected search method
            if search_method == "Grid Search":
                full_history = grid_search(param_grid, lambda p: scorer_for_optimizer(p), log_fn=append_log)

            elif search_method == "Random Search":
                full_history = random_search(param_grid, lambda p: scorer_for_optimizer(p), n_iter=n_iter, log_fn=append_log)

            elif search_method == "Bayesian Optimization":
                full_history = bayesian_search(
                    param_grid,
                    lambda p, i=None, t=None: scorer_for_optimizer(p, i, t),
                    n_calls=n_calls,
                    log_fn=append_log
                )

            # === After optimization is complete ===
            # Get the best strategy based on the optimization results
            best_config = max(full_history, key=lambda x: x["score"])

            # === Display top strategy ===
            st.success(f"✅ Best Score: {best_config['score']:.4f}")
            st.json(best_config)

            # === Apply Best Config and Trigger Re-chunking/Re-indexing ===
            
            if st.button("📥 Apply Best Config and Rebuild Index"):
                # 1) Save best into frozen params (you already do this)
                apply_best_config_to_session(best_config)

                # 2) Make env reflect the best config
                bp = st.session_state["frozen_params"]
                os.environ["CHUNK_STRATEGY"]   = bp["chunking_strategy"]
                os.environ["CHUNK_SIZE"]       = str(bp["chunk_size"])
                os.environ["CHUNK_OVERLAP"]    = str(bp["chunk_overlap"])
                # Use a unique collection name so Qdrant doesn't silently reuse the old one
                os.environ["QDRANT_COLLECTION"] = (
                    f"qdrant_{bp['chunking_strategy']}_c{bp['chunk_size']}_o{bp['chunk_overlap']}"
                )
                os.environ["FORCE_REBUILD"]    = "1"

                # 3) Clear the cached analyzer so we get a fresh build
                try:
                    load_analyzer.clear()  # clears @st.cache_resource
                except Exception:
                    pass

                # 4) Rebuild analyzer now (or set a flag + st.rerun())
                with st.spinner("Rebuilding index with best configuration..."):
                    analyzer = load_analyzer(bp, force_rebuild=True)  # will re-chunk + re-embed + rebuild retriever
                    st.session_state["retriever"] = analyzer.retrieval["retriever"]
                    st.session_state["analyzer"]  = analyzer
                    st.success("✅ Index rebuilt with best configuration.")

                # Optional sanity check: re-run the same claims on the fresh retriever
                if claims:
                    with st.spinner("Validating best config by re-running queries..."):
                        result = analyzer.evaluate_entropy_score(
                            claims,
                            target_profile,
                            log_fn=append_log
                        )
                        st.info("Validation (best config):")
                        st.code(json.dumps(result, indent=2))


            # === Plot Results ===
            result_df = pd.DataFrame(full_history)
            if not result_df.empty:
                plot_optimizer_results(result_df)

            # === Show top 5 strategies ===
            st.markdown("---")
            st.subheader("📜 Top 5 Strategies")
            for i, entry in enumerate(sorted(full_history, key=lambda x: x['score'], reverse=True)[:5]):
                st.markdown(f"### 🔹 Strategy #{i+1} — Score: {entry['score']:.4f}")
                safe_entry = {k: (
                    int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v
                ) for k, v in entry.items()}
                st.code(json.dumps(safe_entry, indent=2))  # Show the selected strategies




elif mode == "Manual":
    tab1, tab2, tab3 = st.tabs(["📊 Entropy Analysis", "🔍 Token Frequencies", "⚙️ Diagnostics"])

# === ANALYZER LOADER ===
@st.cache_resource(show_spinner=True)
def load_analyzer(params, force_rebuild=False):
    os.environ["CHUNK_STRATEGY"] = params["chunking_strategy"]
    os.environ["CHUNK_SIZE"] = str(params["chunk_size"])
    os.environ["CHUNK_OVERLAP"] = str(params["chunk_overlap"])
    os.environ["FORCE_REBUILD"] = "1" if force_rebuild else "0"
    os.environ["QDRANT_COLLECTION"] = f"qdrant_{params['chunking_strategy']}_c{params['chunk_size']}_o{params['chunk_overlap']}"
    return CMSDenialAnalyzer(
        exclude_tokens=params["exclude_tokens"],
        faiss_k=params["faiss_k"],
        bm25_k=params["bm25_k"],
        faiss_fetch_k=params["faiss_fetch_k"],
        weights=params["weights"]
    )


# === MAIN CONTENT AREA ===
tab1, tab2, tab3 = st.tabs(["📊 Entropy Analysis", "🔍 Token Frequencies", "⚙️ Diagnostics"])

# === LOAD ANALYZER ===
if not st.session_state["retriever_initialized"] or st.session_state["rebuild_index"]:
    with st.spinner("Initializing or Rebuilding Retriever..."):
        frozen_params = st.session_state["frozen_params"]
        analyzer = load_analyzer(frozen_params, force_rebuild=st.session_state["rebuild_index"])
        st.session_state["retriever"] = analyzer.retrieval["retriever"]
        st.session_state["analyzer"] = analyzer
        st.session_state["retriever_initialized"] = True
        st.session_state["rebuild_index"] = False
        st.success("Retriever is ready.")

retriever = st.session_state.get("retriever")
if not retriever:
    st.warning("Please click **'Rechunk & Reindex'** to initialize the retriever.")
    st.stop()

# === TAB 1: ENTROPY ANALYSIS ===
with tab1:
    st.subheader("📁 Upload Synthetic Claims File")
    claims_file = st.file_uploader("Upload JSONL file with claims", type="jsonl")

    if claims_file:
        try:
            claims = [json.loads(line) for line in claims_file]
            st.success(f"Loaded {len(claims)} claims")
        except Exception as e:
            st.error(f"Failed to load claims: {e}")
            claims = []

        if st.button("🚀 Generate Entropy Map", key="generate_entropy") and claims:
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

    if "entropy_df" in st.session_state:
        st.subheader("📈 Entropy Map Results")
        st.markdown(f"_Last run: {st.session_state['last_run']}_")

        col1, col2 = st.columns([1, 2])
        
        with col1:
            filter_val = st.slider("📉 Filter: Min Retrieval Count", 1, 100, 1, key="retrieval_filter")
            df = st.session_state["entropy_df"]
            filtered_df = df[df["retrieval_count"] >= filter_val]

            runtime_excludes = st.session_state.get("exclude_tokens_runtime", [])
            if runtime_excludes:
                filtered_df = filtered_df[
                    ~filtered_df["chunk_id"].str.lower().apply(
                        lambda cid: any(tok in cid for tok in runtime_excludes)
                    )
                ]

            st.markdown("#### 📊 Retrieval Frequency")
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
                        
                        runtime_excludes = st.session_state.get("exclude_tokens_runtime", [])
                        if any(tok in chunk_id.lower() for tok in runtime_excludes):

                            continue

                        query_chunk_matrix[query_id][chunk_id] += 1
                        chunk_ids.add(chunk_id)
                except Exception as e:
                    print(f"Error during matrix build on {claim.get('cpt_code')}: {e}")

            chunk_ids = sorted(chunk_ids)
            matrix_df = pd.DataFrame(0, index=query_ids, columns=chunk_ids)

            for q in query_ids:
                for c in query_chunk_matrix[q]:
                    matrix_df.at[q, c] = query_chunk_matrix[q][c]

            matrix_display_df = matrix_df.copy()
            max_val = matrix_display_df.values.max()
            if max_val > 10:
                matrix_display_df = matrix_display_df.clip(upper=10)

            # Create figure with no x-axis labels
            fig = px.imshow(
                matrix_display_df,
                labels=dict(x="", y="Queries"),  # Empty string for x-axis label
                color_continuous_scale="YlGnBu",
                aspect="auto",
                height=400,
            )
            
            # Remove colorbar and x-axis ticks/labels
            fig.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=50, r=50, t=30, b=50),
                xaxis=dict(
                    showticklabels=False,  # Hide x-axis labels completely
                    showgrid=False,
                    title=""  # Ensure no title appears
                ),
                yaxis=dict(
                    showgrid=False,
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📄 View Raw Matrix Data"):
                st.dataframe(matrix_df, use_container_width=True)
                csv_data = matrix_df.reset_index().rename(columns={"index": "Query"}).to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Matrix CSV", csv_data, file_name="query_chunk_matrix.csv", use_container_width=True)

# === TAB 2: TOKEN FREQUENCIES ===
with tab2:
    if "token_freq_df" in st.session_state and not st.session_state["token_freq_df"].empty:
        st.subheader("🧠 Token Frequency Analysis")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            token_to_view = st.selectbox("Select Token to View", st.session_state["token_freq_df"]["token"].unique())
            filtered_token_df = st.session_state["token_freq_df"][st.session_state["token_freq_df"]["token"] == token_to_view]
            
            st.metric("Total Occurrences", filtered_token_df["count"].sum())
            st.metric("Unique Documents", filtered_token_df["source"].nunique())
            
            top_sources = filtered_token_df.groupby("source")["count"].sum().nlargest(5)
            st.write("**Top Sources:**")
            st.dataframe(top_sources.reset_index(), hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown(f"### 🔍 Frequency Heatmap for: `{token_to_view}`")
            pivot_freq = filtered_token_df.pivot_table(
                index="source",
                columns="page",
                values="count",
                fill_value=0
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_freq, annot=False, cmap="Blues", ax=ax)
            ax.set_title(f"Frequency Map: '{token_to_view}'")
            ax.set_xlabel("Page")
            ax.set_ylabel("Source")
            st.pyplot(fig)
    else:
        st.info("Load a claims file and generate entropy maps to view token frequencies.")

# === TAB 3: DIAGNOSTICS ===
with tab3:
    st.subheader("⚙️ System Diagnostics")
    
    if "entropy_df" in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Retrieval Distribution")
            top_chunks = st.session_state["entropy_df"].head(5)
            if not top_chunks.empty:
                total = st.session_state["entropy_df"]["retrieval_count"].sum()
                top_total = top_chunks["retrieval_count"].sum()
                percent = (top_total / total) * 100
                
                st.metric("Top 5 Chunk Share", f"{percent:.1f}%")
                st.progress(percent / 100)
                
                if percent > 60:
                    st.warning("Top chunks dominate retrievals. Consider semantic splitting.")
                else:
                    st.success("Retrieval distribution is balanced.")
        
        with col2:
            st.markdown("### Configuration Check")
            params = st.session_state["frozen_params"]
            
            st.write(f"**Chunk Size:** {params['chunk_size']}")
            st.write(f"**Chunk Overlap:** {params['chunk_overlap']}")
            st.write(f"**FAISS Weight:** {params['weights'][1]:.1f}")
            
            if params["chunk_overlap"] < 1000:
                st.warning("Low chunk overlap may lose boundary context")
            if params["weights"][1] > 0.8:
                st.warning("High FAISS weight may reduce diversity")
    
    st.markdown("### Recommendations")
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
            if st.session_state.get("frozen_chunk_overlap", 2000) < 1000:
                recs.append("🔁 Chunk overlap is low. Consider increasing to preserve boundary context.")

    if recs:
        for r in recs:
            st.warning(r)
    else:
        st.info("✅ Retrieval distribution is healthy.")


