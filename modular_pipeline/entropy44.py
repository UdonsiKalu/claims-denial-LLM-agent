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
from fft_diag import rowwise_power, spectrum_features, detect_rois
from strategy_optimizer import grid_search, random_search, bayesian_search
from faiss_gpu_entropy import CMSDenialAnalyzer
from strategy_profiler import (profile_objective,suggest_weights_from_variance,dynamic_objective,dynamic_objective_banded,)
from langchain_ollama import OllamaEmbeddings  
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# === Default Embeddings (used by semantic chunker) ===
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# === OPTIMIZATION SPACES (place right after imports) ===
exclude_tokens = []
claims = []

# Used by Grid / Random
param_grid = {
    # Retriever knobs
    "faiss_k": [5, 10, 20, 32],
    "bm25_k": [5, 10, 20, 32],
    "faiss_fetch_k": [20, 40, 60, 80],
    "weights": [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)],

    # Chunking knobs
    "chunking_strategy": ["fixed", "recursive", "header", "semantic", "by_page"],
    "chunk_size": [256, 384, 512],
    "chunk_overlap": [32, 64, 96],
    "header_levels": [1, 2, 3],
    "semantic_threshold": [0.3, 0.5, 0.7],

    # Optional reranker (uncomment if wired)
    # "rerank_top_k": [10, 20, 50],
    # "rerank_weight": [0.2, 0.4, 0.6],
}

# Used by Bayesian
search_space = {
    "faiss_k": [5, 10, 20, 32],
    "bm25_k": [5, 10, 20, 32],
    "fetch_k": [20, 40, 60, 80],
    "weights": [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)],

    "chunking_strategy": ["fixed", "recursive", "header", "semantic", "by_page"],
    "chunk_size": [256, 384, 512],
    "chunk_overlap": [32, 64, 96],
    "header_levels": [1, 2, 3],
    "semantic_threshold": [0.3, 0.5, 0.7],

    # "rerank_top_k": [10, 20, 50],
    # "rerank_weight": [0.2, 0.4, 0.6],
}



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

def plot_optimizer_results(
    df,
    bubble_cols=("score", "final_score", "query_entropy", "improvement"),
    normalize_if_negative=True
):
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


def _to_serializable(obj):
    """Recursively convert NumPy/Pandas scalars & arrays to plain Python for json.dumps."""
    if isinstance(obj, np.generic):         # np.int64, np.float32, etc.
        return obj.item()
    if isinstance(obj, np.ndarray):         # arrays -> lists
        return obj.tolist()
    if isinstance(obj, dict):               # dict -> recurse
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):      # list/tuple -> recurse
        return [_to_serializable(v) for v in obj]
    return obj

# === Logging Function ===
def append_log(text):
    # If you pass dicts/lists directly, pretty-print them as JSON (NumPy-safe)
    if isinstance(text, (dict, list)):
        safe = _to_serializable(text)
        entry = f"```json\n{json.dumps(safe, indent=4)}\n```"
    else:
        t = str(text)
        ts = t.lstrip()
        # Keep your JSON-looking heuristic for plain strings
        if (ts.startswith("{") and ts.endswith("}")) or (ts.startswith("[") and ts.endswith("]")):
            # Try to pretty-print if it parses; else just fence it as-is
            try:
                parsed = json.loads(ts)
                entry = f"```json\n{json.dumps(_to_serializable(parsed), indent=4)}\n```"
            except Exception:
                entry = f"```json\n{t}\n```"
        else:
            entry = t

    st.session_state["log_lines"].append(entry)
    log_text = "\n\n".join(st.session_state["log_lines"])
    st.session_state["log_box"].markdown(
        f"""
        <div style="
            background-color:#000;
            color:#0f0;
            padding:1rem;
            font-family:monospace;
            height:400px;
            overflow-y:auto;
            border-radius:8px;
            white-space: pre-wrap;
        ">
{log_text}
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

            # ✅ Preview chunks
            if "analyzer" in st.session_state and getattr(st.session_state["analyzer"], "chunks", None):
                st.subheader("📑 Preview of First 3 Chunks")
                for i, chunk in enumerate(st.session_state["analyzer"].chunks[:3]):
                    st.text_area(f"Chunk {i+1}", chunk.page_content[:500], height=120)

                # ✅ Download JSONL
                import json
                chunks_jsonl = "\n".join(json.dumps({
                    "id": f"chunk_{i}",
                    "text": chunk.page_content,
                    "metadata": chunk.metadata
                }) for i, chunk in enumerate(st.session_state["analyzer"].chunks))

                st.download_button(
                    "⬇️ Download Chunks JSONL",
                    data=chunks_jsonl,
                    file_name="chunks.jsonl",
                    mime="application/jsonl"
                )

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
                "exclude_tokens": exclude_tokens,
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
                strategy_raw = str(params.get("chunking_strategy", "fixed")).strip().lower()
                STRAT_MAP = {
                    "fixed": "Fixed-size",
                    "recursive": "Recursive",
                    "header": "Header-aware",
                    "semantic": "Semantic",
                    "by_page": "By-page",
                }
                os.environ["CHUNK_STRATEGY"] = STRAT_MAP.get(strategy_raw, "Fixed-size")
                os.environ["CHUNK_SIZE"] = str(chunk_size)
                os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
                os.environ["FORCE_REBUILD"] = "1"  # ensure reindex per strategy
                os.environ["QDRANT_COLLECTION"] = f"qdrant_{strategy_raw}_c{chunk_size}_o{chunk_overlap}"


                # 3) Strategy id (for display)
                strategy_id = (
                    f"faiss{params['faiss_k']}_bm25{params['bm25_k']}"
                    f"_fetch{params['faiss_fetch_k']}_w{params['weights']}"
                    f"_{strategy_raw}_c{chunk_size}_o{chunk_overlap}"
                )
                params["strategy_id"] = strategy_id

                # 4) (Optional) apply runtime excludes to CLAIMS (not required; keep if you really use it)
                exclude_tokens_rt = st.session_state.get("exclude_tokens_runtime", [])
                filtered_claims = claims  # default: use original claims
                # If you truly need to drop claims based on some token logic, do it here.
                # Otherwise leave as-is to ensure all queries run every time.

                # 5) Build analyzer for this strategy (this triggers chunking + Qdrant rebuild per env)
                analyzer = CMSDenialAnalyzer(
                    exclude_tokens=params.get("exclude_tokens", []),
                    faiss_k=params.get("faiss_k", 5),
                    bm25_k=params.get("bm25_k", 3),
                    faiss_fetch_k=params.get("faiss_fetch_k", 25),
                    weights=params.get("weights", (0.4, 0.6)),
                    chunking_strategy=params.get("chunking_strategy"),
                    chunk_size=params.get("chunk_size"),
                    chunk_overlap=params.get("chunk_overlap"),
                    header_levels=params.get("header_levels"),         # ✅
                    semantic_threshold=params.get("semantic_threshold") # ✅
                )

                # 6) ACTUALLY RUN ALL CLAIM QUERIES for this strategy
                append_log(f"🔎 Running queries for {len(filtered_claims)} claims...")

                # ✅ create 'result' FIRST
                result = analyzer.evaluate_entropy_score(
                    filtered_claims,
                    target_profile,
                    log_fn=append_log,
                    idx=idx,
                    total=total,
                    params=params,
                    search_method=search_method,
                )

                # ---- Pretty JSON log (only when idx/total are provided by the optimizer) ----
                emit_log = (idx is not None) and (total is not None)
                if emit_log:
                    payload = {
                        "strategy_number": idx,
                        "total_strategies": total,
                        "chunking": {
                            "strategy": strategy_raw,
                            "size": chunk_size,
                            "overlap": chunk_overlap,
                            "header_levels": params.get("header_levels"),
                            "semantic_threshold": params.get("semantic_threshold"),
                        },
                        "retriever": {
                            "faiss_fetch_k": int(params["faiss_fetch_k"]),
                            "bm25_k": int(params["bm25_k"]),
                            "fetch_k": int(params["fetch_k"]),
                            "weights": list(weights) if isinstance(weights, tuple) else params["weights"],
                            "rerank_top_k": params.get("rerank_top_k"),
                            "rerank_weight": params.get("rerank_weight"),
                        },
                        "optimization": {
                            "search_method": search_method,
                            "params": params,
                        },
                        "scores": {
                            "entropy_score": result["score"],
                            "metrics": {
                                "query_entropy": result["entropy"],
                                "max_chunk_frequency": result["max_freq"],
                                "gini_coefficient": result["gini"],
                                "code_coverage": result["coverage"],
                            },
                            "penalties": result.get("penalties", {}),
                            "final_strategy_score": result["score"],
                        },
                    }
                    append_log(payload)  # your logger pretty-prints dicts as JSON

                # 7) Extra logging (optional)
                append_log(
                    "📊 Strategy metrics: "
                    f"entropy={result.get('entropy'):.4f}, "
                    f"max_freq={result.get('max_freq'):.4f}, "
                    f"gini={result.get('gini'):.4f}, "
                    f"coverage={result.get('coverage'):.4f}, "
                    f"final_score={result.get('score'):.4f}"
                )

                # 8) Ensure identifiers are on result
                result["strategy_id"] = strategy_id
                result["chunk_size"] = chunk_size
                result["chunk_overlap"] = chunk_overlap
                return result




            def apply_best_config_to_session(config):
                # normalize weights (tuple or "a,b" string → tuple)
                w = config.get("weights", (0.5, 0.5))
                if isinstance(w, str):
                    try:
                        w = tuple(map(float, w.split(",")))
                    except Exception:
                        w = (0.5, 0.5)

                # normalize chunking strategy to the env label your loader expects
                strategy_raw = str(config.get("chunking_strategy", "fixed")).strip().lower()
                STRAT_MAP = {
                    "fixed": "Fixed-size",
                    "recursive": "Recursive",
                    "header": "Header-aware",
                    "semantic": "Semantic",
                    "by_page": "By-page",
                }
                chunk_strategy_env = STRAT_MAP.get(strategy_raw, "Fixed-size")

                st.session_state["frozen_params"] = {
                    "chunking_strategy": chunk_strategy_env,
                    "chunk_size": int(config.get("chunk_size", 10000)),
                    "chunk_overlap": int(config.get("chunk_overlap", 2000)),
                    "faiss_fetch_k": int(config["faiss_fetch_k"]),
                    "bm25_k": int(config["bm25_k"]),
                    "faiss_fetch_k": int(config["fetch_k"]),
                    "weights": w,
                    "exclude_tokens": st.session_state.get("exclude_tokens_runtime", []),
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
                # No idx/total and no logs during profiling; re-use your evaluator
                return evaluate_strategy(params, claims, target_profile)


            # Quick probe over the declared param_grid
            profile = profile_objective(param_grid, score_fn_raw, n=12)
            append_log(f"📊 Profiling stats:\n{json.dumps(profile['stats'], indent=2)}")

            # Decide which scorer to use for the optimizer calls
            if profile.get("score_flat"):
                append_log("⚠️ Detected flat objective. Switching to data-driven dynamic objective.")
                weights_suggested = suggest_weights_from_variance(profile["stats"])
                USE_BANDED = True
                entropy_band = target_profile.get("query_entropy_range", (0.7, 0.9))

                def scorer_for_optimizer(p, i=None, t=None):
                    r = evaluate_strategy(p, claims, target_profile, idx=i, total=t)  # ← passes idx/total
                    if USE_BANDED:
                        r["score"] = dynamic_objective_banded(r, weights_suggested, entropy_band)
                    else:
                        r["score"] = dynamic_objective(r, weights_suggested)
                    return r
            else:
                append_log("✅ Objective shows usable variance. Proceeding with existing score.")
                def scorer_for_optimizer(p, i=None, t=None):
                    # Re-use the same path that builds an analyzer per-params
                    return evaluate_strategy(p, claims, target_profile, idx=i, total=t)



            # ---- END PROFILING ----


            # Run the optimization process based on the selected search method
            if search_method == "Grid Search":
                full_history = grid_search(
                    param_grid,
                    lambda p, i, t: scorer_for_optimizer(p, i, t),   # ← pass i & t through
                    log_fn=append_log
                )

            elif search_method == "Random Search":
                full_history = random_search(
                    param_grid,
                    lambda p, i, t: scorer_for_optimizer(p, i, t),   # ← pass i & t through
                    n_iter=n_iter,
                    log_fn=append_log
                )
            elif search_method == "Bayesian Optimization":
                # ✅ Change this line to use `search_space` instead of `param_grid`
                full_history = bayesian_search(
                    search_space,
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
    # --- Normalize strategy casing ---
    strategy_raw = str(params.get("chunking_strategy", "fixed")).strip().lower()
    STRAT_MAP = {
        "fixed": "Fixed-size",
        "recursive": "Recursive",
        "header": "Header-aware",
        "semantic": "Semantic",
        "by_page": "By-page",
    }
    chunk_strategy_env = STRAT_MAP.get(strategy_raw, "Fixed-size")

    # --- Core env vars ---
    os.environ["CHUNK_STRATEGY"] = chunk_strategy_env
    os.environ["CHUNK_SIZE"] = str(params.get("chunk_size", 10000))
    os.environ["CHUNK_OVERLAP"] = str(params.get("chunk_overlap", 2000))
    os.environ["FORCE_REBUILD"] = "1" if force_rebuild else "0"

    # --- Optional params ---
    if params.get("header_levels") is not None:
        os.environ["HEADER_LEVELS"] = str(params["header_levels"])
    else:
        os.environ.pop("HEADER_LEVELS", None)

    if params.get("semantic_threshold") is not None:
        os.environ["SEMANTIC_THRESHOLD"] = str(params["semantic_threshold"])
    else:
        os.environ.pop("SEMANTIC_THRESHOLD", None)

    # --- Unique Qdrant collection name ---
    collection = f"qdrant_{strategy_raw}_c{params.get('chunk_size',10000)}_o{params.get('chunk_overlap',2000)}"
    if params.get("header_levels"):
        collection += f"_h{params['header_levels']}"
    if params.get("semantic_threshold"):
        collection += f"_st{params['semantic_threshold']}"
    os.environ["QDRANT_COLLECTION"] = collection

    # --- Build Analyzer ---
    return CMSDenialAnalyzer(
        exclude_tokens=params.get("exclude_tokens", []),
        faiss_k=params.get("faiss_k", 5),
        bm25_k=params.get("bm25_k", 3),
        faiss_fetch_k=params.get("faiss_fetch_k", 20),
        weights=params.get("weights", (0.5, 0.5)),
        chunking_strategy=params.get("chunking_strategy", "fixed"),
        chunk_size=params.get("chunk_size", 512),
        chunk_overlap=params.get("chunk_overlap", 64),
        header_levels=params.get("header_levels", 3),
        semantic_threshold=params.get("semantic_threshold", 0.5),
        embeddings=embeddings   # 👈 here’s the fix
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


            # === FFT PERIODICITY ANALYSIS ===
            R = matrix_df.values  # shape (Q, C)

            # --- FFT Analysis (only if matrix_df has data) ---

            # 🔄 Reset FFT state at the top of the block
            fft_results = pd.DataFrame()
            fft_chunks_df, fft_queries_df = pd.DataFrame(), pd.DataFrame()
            P_chunks, P_queries = np.array([]), np.array([])
            features_chunks, features_queries = [], []
            target_idx_chunks, target_idx_queries = None, None

            if matrix_df is None or matrix_df.empty:
                st.warning("⚠️ Retrieval matrix is empty. Try uploading claims or changing chunking strategy.")
            else:
                R = matrix_df.values  # shape (Q, C)

                # Row-wise and column-wise FFT power spectra
                freqs_c, P_chunks = rowwise_power(R, axis=0)   # chunks
                freqs_q, P_queries = rowwise_power(R, axis=1)  # queries

                # === Extract per-row spectral features (safe) ===
                features_chunks = [spectrum_features(freqs_c, row) for row in P_chunks if row.size > 0]
                features_queries = [spectrum_features(freqs_q, row) for row in P_queries if row.size > 0]

                fft_chunks_df = pd.DataFrame(features_chunks) if features_chunks else pd.DataFrame()
                fft_queries_df = pd.DataFrame(features_queries) if features_queries else pd.DataFrame()

                # --- Attach chunk IDs safely ---
                if not fft_chunks_df.empty:
                    chunk_ids = list(matrix_df.columns)
                    n = len(fft_chunks_df)

                    if len(chunk_ids) != n:
                        st.warning(f"⚠️ Chunk ID length mismatch: {len(chunk_ids)} vs {n}. Adjusting.")
                    # Trim or pad to exact length
                    chunk_ids = chunk_ids[:n] + [None] * max(0, n - len(chunk_ids))

                    fft_chunks_df.insert(0, "chunk_id", chunk_ids)

                # Final export table for chunks
                fft_results = fft_chunks_df


                # --- Auto-select most common peak frequency (safe) ---
                peak_idx_chunks = (
                    fft_chunks_df["peak_idx"].dropna().astype(int).values
                    if not fft_chunks_df.empty and "peak_idx" in fft_chunks_df
                    else np.array([])
                )
                peak_idx_queries = (
                    fft_queries_df["peak_idx"].dropna().astype(int).values
                    if not fft_queries_df.empty and "peak_idx" in fft_queries_df
                    else np.array([])
                )

                target_idx_chunks = (
                    int(np.bincount(peak_idx_chunks[peak_idx_chunks > 0]).argmax())
                    if peak_idx_chunks.size > 0 and np.any(peak_idx_chunks > 0)
                    else None
                )
                target_idx_queries = (
                    int(np.bincount(peak_idx_queries[peak_idx_queries > 0]).argmax())
                    if peak_idx_queries.size > 0 and np.any(peak_idx_queries > 0)
                    else None
                )

                # --- Optional debugging ---
                if len(features_chunks) > 0:
                    st.write("Example chunk feature:", features_chunks[0])
                if len(features_queries) > 0:
                    st.write("Example query feature:", features_queries[0])
                if len(features_chunks) == 0 and len(features_queries) == 0:
                    st.info("ℹ️ No valid spectral features extracted.")


            # --- Scan all bins programmatically ---
            if "P_chunks" in locals() and P_chunks is not None and len(P_chunks) > 0 \
               and "P_queries" in locals() and P_queries is not None and len(P_queries) > 0:

                avg_chunk_spectrum = P_chunks.mean(axis=0)
                bin_strength_chunks = (P_chunks > 0.05).sum(axis=0)
                perc_chunks = bin_strength_chunks / P_chunks.shape[0] * 100

                avg_query_spectrum = P_queries.mean(axis=0)
                bin_strength_queries = (P_queries > 0.05).sum(axis=0)
                perc_queries = bin_strength_queries / P_queries.shape[0] * 100

                sig_bins_chunks = np.where(perc_chunks >= 20)[0]
                sig_bins_queries = np.where(perc_queries >= 20)[0]

                st.markdown("### 🔍 Global Bin Scan")
                st.write("Significant bins (chunks):", sig_bins_chunks.tolist())
                st.write("Significant bins (queries):", sig_bins_queries.tolist())

                # --- Build DataFrame for chunk bins ---
                df_bins_chunks = pd.DataFrame({
                    "bin": np.arange(len(avg_chunk_spectrum)),
                    "perc_active_chunks": perc_chunks,
                })
                fig_chunks = px.bar(
                    df_bins_chunks,
                    x="bin",
                    y="perc_active_chunks",
                    title="Bin activity across chunks",
                    labels={"perc_active_chunks": "% rows active"}
                )
                st.plotly_chart(fig_chunks, use_container_width=True)

            else:
                st.warning("⚠️ FFT bin scan skipped — no valid P_chunks/P_queries (empty retrieval or chunking strategy).")

            # --- Build DataFrame for query bins ---

            if "avg_query_spectrum" in locals() and avg_query_spectrum is not None:
                df_bins_queries = pd.DataFrame({
                    "bin": np.arange(len(avg_query_spectrum)),
                    "perc_active_queries": perc_queries,
                })
                fig_queries = px.bar(
                    df_bins_queries,
                    x="bin",
                    y="perc_active_queries",
                    title="Bin activity across queries",
                    labels={"perc_active_queries": "% rows active"}
                )
                st.plotly_chart(fig_queries, use_container_width=True)
            else:
                st.warning("⚠️ Query bin scan skipped — no valid query spectrum.")



                        #############################

            # --- Optional debugging/inspection ---
            if "features_chunks" in locals() and features_chunks:
                st.write("Example chunk feature:", features_chunks[0])

                # Collect peak indices and PNRs (if you need them later)
                peak_indices = [f.get("peak_index") for f in features_chunks if "peak_index" in f]
                pnr_values = [f.get("PNR") for f in features_chunks if "PNR" in f]

                st.write("Collected peak indices (chunks):", peak_indices[:10])
                st.write("Collected PNR values (chunks):", pnr_values[:10])
            else:
                st.warning("⚠️ No chunk features available — maybe empty retrieval matrix or strategy produced no FFT data.")




            # === FFT PERIODICITY ANALYSIS ===
            R = matrix_df.values  # shape (Q, C)

            freqs_c, P_chunks = rowwise_power(R, axis=0)   # chunks
            freqs_q, P_queries = rowwise_power(R, axis=1)  # queries

            feat_chunks = [spectrum_features(freqs_c, P_chunks[c]) for c in range(P_chunks.shape[0])]
            feat_queries = [spectrum_features(freqs_q, P_queries[q]) for q in range(P_queries.shape[0])]

            peak_idx_chunks = np.array([f["peak_idx"] for f in feat_chunks])
            pnr_chunks = np.array([f["pnr_db"] for f in feat_chunks])

            # --- Slider + Highlights ---
            bin_idx = st.slider("FFT Bin Index", min_value=1, max_value=len(freqs_q)-1, value=10, step=1)

            highlight_queries = [q for q, f in enumerate(feat_queries) if f["peak_idx"] == bin_idx]
            highlight_chunks = [c for c, f in enumerate(feat_chunks) if f["peak_idx"] == bin_idx]

            st.write(f"Bin {bin_idx}: {len(highlight_queries)} queries, {len(highlight_chunks)} chunks highlighted")

            import matplotlib.pyplot as plt

            # Queries heatmap
            fig_q, ax_q = plt.subplots(figsize=(8, 4))
            im_q = ax_q.imshow(P_queries, aspect="auto", cmap="viridis", origin="lower")
            ax_q.set_title("FFT Power (Queries)")
            ax_q.set_xlabel("Frequency Bin")
            ax_q.set_ylabel("Query Index")
            fig_q.colorbar(im_q, ax=ax_q, label="Normalized Power")
            for q in highlight_queries:
                ax_q.axhline(q, color="red", lw=1.5, alpha=0.7)
            st.pyplot(fig_q)

            # Chunks heatmap
            fig_c, ax_c = plt.subplots(figsize=(8, 4))
            im_c = ax_c.imshow(P_chunks, aspect="auto", cmap="viridis", origin="lower")
            ax_c.set_title("FFT Power (Chunks)")
            ax_c.set_xlabel("Frequency Bin")
            ax_c.set_ylabel("Chunk Index")
            fig_c.colorbar(im_c, ax=ax_c, label="Normalized Power")
            for c in highlight_chunks:
                ax_c.axhline(c, color="red", lw=1.5, alpha=0.7)
            st.pyplot(fig_c)

            # --- Continue with your summary/diagnostics ---
            st.markdown("### 🔎 FFT Periodicity Diagnostics")


            # --- Clean peak_idx before using it ---
            if not fft_chunks_df.empty and "peak_idx" in fft_chunks_df:
                peak_idx_chunks = (
                    fft_chunks_df["peak_idx"]
                    .dropna()                 # remove NaN
                    .astype(float)            # force numeric
                    .astype(int, errors="ignore")
                    .values
                )
            else:
                peak_idx_chunks = np.array([], dtype=int)

            # --- Safe check ---
            if peak_idx_chunks.size > 0 and np.any(peak_idx_chunks > 0):
                target_idx = int(np.bincount(peak_idx_chunks[peak_idx_chunks > 0]).argmax())
            else:
                target_idx = None


            # Detect contiguous ROIs of periodic chunks
            chunk_rois, query_rois = [], []
            if target_idx is not None:
                chunk_rois = detect_rois(
                    P_chunks, peak_idx_chunks, target_idx,
                    pnr_db=pnr_chunks, min_pnr_db=6.0, min_run=5
                )

                peak_idx_queries = np.array([f["peak_idx"] for f in feat_queries])
                pnr_queries = np.array([f["pnr_db"] for f in feat_queries])
                query_rois = detect_rois(
                    P_queries, peak_idx_queries, target_idx,
                    pnr_db=pnr_queries, min_pnr_db=6.0, min_run=5
                )

            st.markdown("### 🔎 FFT Periodicity Diagnostics")
            st.write(f"Detected {len(chunk_rois)} chunk ROIs with strong periodicity.")
            st.write(f"Detected {len(query_rois)} query ROIs with strong periodicity.")

            # === FFT HEATMAP VISUALIZATIONS ===
            st.markdown("#### 🔥 FFT Power (Chunks)")
            fig_chunks = px.imshow(
                P_chunks.T,
                aspect="auto",
                labels=dict(x="Chunk", y="Frequency Bin", color="Power"),
                color_continuous_scale="Viridis"
            )
            for (start, end) in chunk_rois:
                fig_chunks.add_vrect(start, end, fillcolor="red", opacity=0.25, line_width=0)
            st.plotly_chart(fig_chunks, use_container_width=True)

            st.markdown("#### 🔥 FFT Power (Queries)")
            fig_queries = px.imshow(
                P_queries.T,
                aspect="auto",
                labels=dict(x="Query", y="Frequency Bin", color="Power"),
                color_continuous_scale="Cividis"
            )
            for (start, end) in query_rois:
                fig_queries.add_vrect(start, end, fillcolor="blue", opacity=0.25, line_width=0)
            st.plotly_chart(fig_queries, use_container_width=True)

            # === FREQUENCY BIN SELECTOR ===
            max_bin = P_chunks.shape[1]
            selected_bin = st.slider("🎛️ Select frequency bin", 0, max_bin - 1, value=target_idx or 0)
            st.write(f"Selected bin: {selected_bin}")

            # === CROSS-INTEGRATION WITH ENTROPY MAP ===
            if "entropy_df" in st.session_state:
                df_entropy = st.session_state["entropy_df"].copy()
                df_entropy["is_periodic"] = df_entropy["chunk_id"].apply(
                    lambda cid: any(start <= i <= end for (start, end) in chunk_rois
                                    for i, c in enumerate(matrix_df.columns) if c == cid)
                )
                st.markdown("#### 🌀 Entropy + FFT Cross Map")
                st.dataframe(df_entropy.head(20))


            # === EXPORT FFT RESULTS (safe version) ===
            max_len = max(len(matrix_df.columns), len(peak_idx_chunks), len(pnr_chunks))

            chunk_ids   = list(matrix_df.columns)[:max_len]
            peak_idx    = list(peak_idx_chunks)[:max_len]
            pnr_db_vals = list(pnr_chunks)[:max_len]

            # Pad any shorter lists with None
            while len(chunk_ids) < max_len: chunk_ids.append(None)
            while len(peak_idx) < max_len: peak_idx.append(None)
            while len(pnr_db_vals) < max_len: pnr_db_vals.append(None)

            fft_results = pd.DataFrame({
                "chunk_id": chunk_ids,
                "peak_idx": peak_idx,
                "pnr_db": pnr_db_vals
            })

            st.download_button(
                "⬇️ Download FFT Results",
                fft_results.to_csv(index=False).encode("utf-8"),
                "fft_results.csv",
                "text/csv"
            )




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

            st.json({
                "Chunking": {
                    "Strategy": params.get("chunking_strategy"),
                    "Chunk Size": params.get("chunk_size"),
                    "Chunk Overlap": params.get("chunk_overlap"),
                    "Header Levels": params.get("header_levels"),
                    "Semantic Threshold": params.get("semantic_threshold"),
                },
                "Retriever": {
                    "FAISS k": params.get("faiss_k"),
                    "BM25 k": params.get("bm25_k"),
                    "FAISS Fetch_k": params.get("faiss_fetch_k"),
                    "Weights": params.get("weights"),
                },
                "Exclusions": params.get("exclude_tokens"),
            })

            
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


