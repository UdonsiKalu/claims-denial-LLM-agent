# === STREAMLIT CMS RETRIEVAL OPTIMIZER ===
import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from strategy_optimizer import grid_search, random_search, bayesian_search
from faiss_gpu_entropy import CMSDenialAnalyzer

# =============================================
# === UI CONFIGURATION ===
# =============================================
st.set_page_config(page_title="Retrieval Studio", layout="wide")
st.title("📊 Retrieval Optimization Studio")

# Custom CSS styling
st.markdown("""
<style>
    .main {padding: 2rem 1.5rem 5rem;}
    .sidebar .sidebar-content {padding: 1.5rem;}
    .stPlotlyChart, .stDataFrame {border-radius: 8px;}
    .metric-card {border: 1px solid #333; border-radius: 8px; padding: 1rem;}
    .param-section {border: 1px solid #444; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;}
    .terminal {font-family: monospace; background-color: #0a0a0a; color: #00ff00; 
               padding: 1rem; border-radius: 8px; border: 1px solid #00aa00;}
    .stTabs [data-baseweb="tab-list"] {gap: 0.5rem;}
    .stTabs [data-baseweb="tab"] {padding: 0.5rem 1rem;}
</style>
""", unsafe_allow_html=True)

# =============================================
# === SESSION STATE INITIALIZATION ===
# =============================================
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []
if "retriever_initialized" not in st.session_state:
    st.session_state.retriever_initialized = False
if "rebuild_index" not in st.session_state:
    st.session_state.rebuild_index = False
if not st.session_state.retriever_initialized:
    st.session_state.frozen_params = {
        "chunking_strategy": "Fixed-size",
        "chunk_size": 10000,
        "chunk_overlap": 2000,
        "faiss_k": 5,
        "bm25_k": 3,
        "faiss_fetch_k": 50,
        "weights": (0.4, 0.6),
        "exclude_tokens": ["page_0", "introduction", "scope", "purpose"]
    }

# =============================================
# === CORE FUNCTIONS ===
# =============================================
def append_log(text):
    st.session_state.log_lines.append(text)
    log_text = "\n".join(st.session_state.log_lines)
    st.session_state.log_box.markdown(
        f'<div class="terminal"><pre>{log_text}</pre></div>',
        unsafe_allow_html=True
    )

def plot_optimizer_results(df, bubble_cols=("score","final_score","query_entropy","improvement"),
                         normalize_if_negative=True):
    with st.container():
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

    bubble_col = next((c for c in bubble_cols if c in df.columns), None)

    def _safe_sizes(series: pd.Series):
        s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if normalize_if_negative and s.min() < 0:
            s = s - s.min() + 1e-6
        s = np.clip(s, 0, np.nanpercentile(s, 99))
        if np.allclose(s, 0):
            s = np.ones_like(s)
        return s

    if {"entropy", "coverage"}.issubset(df.columns):
        with st.container():
            st.subheader("🎯 Entropy vs Coverage")
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
                )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

    if {"gini","score"}.issubset(df.columns):
        with st.container():
            st.subheader("📉 Gini vs Score")
            fig_gini = px.scatter(
                df,
                x="gini",
                y="score",
                color="score",
                hover_data=["strategy_id"] if "strategy_id" in df.columns else df.columns,
                title="Gini Coefficient vs Score",
                color_continuous_scale="Plasma"
            )
            fig_gini.update_layout(height=400)
            st.plotly_chart(fig_gini, use_container_width=True)

    if "score" in df.columns and "strategy_id" in df.columns:
        with st.container():
            st.subheader("🔥 Strategy Score Heatmap")
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

    if {"max_freq","score"}.issubset(df.columns):
        with st.container():
            st.subheader("📊 Max Chunk Frequency vs Score")
            fig_maxfreq = px.scatter(
                df,
                x="max_freq",
                y="score",
                color="score",
                hover_data=["strategy_id"] if "strategy_id" in df.columns else df.columns,
                title="Max Chunk Frequency vs Score",
                color_continuous_scale="Cividis"
            )
            fig_maxfreq.update_layout(height=400)
            st.plotly_chart(fig_maxfreq, use_container_width=True)

@st.cache_resource(show_spinner=True)
def load_analyzer(params, force_rebuild=False):
    os.environ.update({
        "CHUNK_STRATEGY": params["chunking_strategy"],
        "CHUNK_SIZE": str(params["chunk_size"]),
        "CHUNK_OVERLAP": str(params["chunk_overlap"]),
        "FORCE_REBUILD": "1" if force_rebuild else "0",
        "QDRANT_COLLECTION": f"qdrant_{params['chunking_strategy']}_c{params['chunk_size']}_o{params['chunk_overlap']}"
    })
    return CMSDenialAnalyzer(
        exclude_tokens=params["exclude_tokens"],
        faiss_k=params["faiss_k"],
        bm25_k=params["bm25_k"],
        faiss_fetch_k=params["faiss_fetch_k"],
        weights=params["weights"]
    )

# =============================================
# === SIDEBAR CONFIGURATION ===
# =============================================
with st.sidebar:
    st.header("⚙️ Configuration Panel")
    mode = st.radio("🔀 App Mode", ["Manual", "Optimizer"], horizontal=True)
    
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
            st.session_state.exclude_tokens_runtime = exclude_tokens

        if st.button("🚀 Rechunk & Reindex", use_container_width=True, type="primary"):
            st.session_state.frozen_params = {
                "chunking_strategy": ui_chunking_strategy,
                "chunk_size": ui_chunk_size,
                "chunk_overlap": ui_chunk_overlap,
                "faiss_k": ui_faiss_k,
                "bm25_k": ui_bm25_k,
                "faiss_fetch_k": ui_faiss_fetch_k,
                "weights": ui_weights,
                "exclude_tokens": exclude_tokens
            }
            st.session_state.rebuild_index = True
            st.rerun()

# =============================================
# === MAIN CONTENT AREA ===
# =============================================
# Initialize analyzer if needed
if not st.session_state.retriever_initialized or st.session_state.rebuild_index:
    with st.spinner("Initializing retriever..."):
        analyzer = load_analyzer(st.session_state.frozen_params, 
                               force_rebuild=st.session_state.rebuild_index)
        st.session_state.retriever = analyzer.retrieval["retriever"]
        st.session_state.analyzer = analyzer
        st.session_state.retriever_initialized = True
        st.session_state.rebuild_index = False

# Progress and log containers
progress_container = st.container()
with progress_container:
    progress_bar = st.progress(0.0, text="Ready for optimization...")

log_container = st.container()
with log_container:
    log_box = st.empty()
    st.session_state.log_box = log_box
    append_log("System initialized and ready")

# Mode-specific content
if mode == "Optimizer":
    st.header("🤖 Retrieval Strategy Optimizer")
    
    with st.container(border=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Input Data")
            claims_file = st.file_uploader("Upload Synthetic Claims (JSONL)", type="jsonl", key="opt_claims")
            claims = []
            if claims_file:
                try:
                    claims = [json.loads(line) for line in claims_file]
                    st.success(f"✅ Loaded {len(claims)} claims")
                except Exception as e:
                    st.error(f"❌ Failed to parse file: {e}")
        
        with col2:
            st.subheader("Target Profile")
            target_profile = {
                "query_entropy_range": st.slider("Query Entropy Range", 0.0, 1.0, (0.7, 0.9)),
                "max_chunk_frequency": st.slider("Max Chunk Frequency", 0.01, 0.5, 0.1),
                "gini_threshold": st.slider("Gini Coefficient", 0.0, 1.0, 0.4),
                "required_code_coverage": st.slider("Code Coverage", 0.0, 1.0, 0.95)
            }

    if claims:
        with st.container(border=True):
            st.subheader("Optimization Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                search_method = st.selectbox("🔍 Method", ["Grid Search", "Random Search", "Bayesian Optimization"])
                if search_method == "Random Search":
                    n_iter = st.slider("Iterations", 5, 50, 10)
                elif search_method == "Bayesian Optimization":
                    n_calls = st.slider("Bayesian Calls", 5, 30, 15)
            
            with col2:
                param_grid = {
                    "chunk_size": [5000, 7500, 10000, 15000],
                    "chunk_overlap": [500, 1000, 2000, 3000],
                    "faiss_k": list(range(3, 8)),
                    "bm25_k": list(range(2, 6)),
                    "fetch_k": [25, 50],
                    "weights": ["0.4,0.6", "0.5,0.5", "0.6,0.4"]
                }
                st.caption("Parameter space to explore")

        if st.button("🚀 Run Optimizer", type="primary", use_container_width=True):
            st.session_state.log_lines = []
            append_log("🚀 Starting optimization run...")

            def evaluate_strategy(params, claims, target_profile, idx=None, total=None):
                if idx is not None and total is not None:
                    append_log(f"⚙️ Evaluating strategy {idx}/{total}: {params}")

                raw_weights = params["weights"]
                weights = tuple(map(float, raw_weights.split(","))) if isinstance(raw_weights, str) else raw_weights

                os.environ["CHUNK_SIZE"] = str(int(params.get("chunk_size", 10000)))
                os.environ["CHUNK_OVERLAP"] = str(int(params.get("chunk_overlap", 2000)))

                analyzer = CMSDenialAnalyzer(
                    exclude_tokens=[],
                    faiss_k=int(params["faiss_k"]),
                    bm25_k=int(params["bm25_k"]),
                    faiss_fetch_k=int(params["fetch_k"]),
                    weights=weights
                )

                result = analyzer.evaluate_entropy_score(
                    claims,
                    target_profile,
                    log_fn=append_log,
                    idx=idx,
                    total=total
                )
                return result

            def apply_best_config_to_session(config):
                st.session_state.frozen_params = {
                    "chunking_strategy": "Fixed-size",
                    "chunk_size": config.get("chunk_size", 10000),
                    "chunk_overlap": config.get("chunk_overlap", 2000),
                    "faiss_k": config["faiss_k"],
                    "bm25_k": config["bm25_k"],
                    "faiss_fetch_k": config["fetch_k"],
                    "weights": tuple(map(float, config["weights"].split(","))),
                    "exclude_tokens": st.session_state.get("exclude_tokens_runtime", [])
                }
                st.session_state.rebuild_index = True

            with st.spinner("Running optimization..."):
                if search_method == "Grid Search":
                    full_history = grid_search(
                        param_grid, 
                        lambda p: evaluate_strategy(p, claims, target_profile),
                        log_fn=append_log
                    )
                elif search_method == "Random Search":
                    full_history = random_search(
                        param_grid,
                        lambda p: evaluate_strategy(p, claims, target_profile),
                        n_iter=n_iter,
                        log_fn=append_log
                    )
                elif search_method == "Bayesian Optimization":
                    full_history = bayesian_search(
                        param_grid,
                        lambda p, i=None, t=None: evaluate_strategy(p, claims, target_profile, idx=i, total=t),
                        n_calls=n_calls,
                        log_fn=append_log
                    )

                st.session_state.results = full_history
                st.success("Optimization complete!")

            with st.container(border=True):
                st.subheader("Results")
                best_config = max(full_history, key=lambda x: x["score"])
                st.metric("Best Score", f"{best_config['score']:.4f}")
                
                if st.button("📥 Apply Best Config", use_container_width=True):
                    apply_best_config_to_session(best_config)
                    st.success("✅ Applied best configuration!")

                st.subheader("Top 5 Strategies")
                for i, entry in enumerate(sorted(full_history, key=lambda x: x['score'], reverse=True)[:5]):
                    with st.expander(f"🔹 Strategy #{i+1} — Score: {entry['score']:.4f}"):
                        safe_entry = {
                            k: (int(v) if isinstance(v, (np.integer, np.int64))
                                else float(v) if isinstance(v, (np.floating, np.float64))
                                else v)
                            for k, v in entry.items()
                        }
                        st.json(safe_entry)

            plot_optimizer_results(pd.DataFrame(full_history))

elif mode == "Manual":
    tab1, tab2, tab3 = st.tabs(["📊 Entropy Analysis", "🔍 Token Frequencies", "⚙️ Diagnostics"])
    
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

            if st.button("🚀 Generate Entropy Map", key="generate_entropy"):
                with st.spinner("Running analysis..."):
                    retrieval_log = defaultdict(int)
                    token_data = []
                    
                    for claim in claims:
                        try:
                            docs = st.session_state.retriever.invoke(claim["cpt_code"])
                            for doc in docs:
                                source = doc.metadata.get("source", "unknown")
                                page = doc.metadata.get("page", "0")
                                chunk_id = f"{source}::Page_{page}"
                                retrieval_log[chunk_id] += 1
                        except Exception as e:
                            print(f"Error on {claim.get('cpt_code')}: {e}")

                    entropy_df = pd.DataFrame([
                        {"chunk_id": k, "retrieval_count": v} for k, v in retrieval_log.items()
                    ]).sort_values("retrieval_count", ascending=False)
                    
                    st.session_state.entropy_df = entropy_df
                    st.session_state.last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    if st.session_state.analyzer:
                        st.session_state.analyzer.compute_token_frequencies()
                        for chunk in st.session_state.analyzer.chunks:
                            freqs = chunk.metadata.get("token_frequencies", {})
                            token_data.extend({
                                "source": chunk.metadata.get("source", "unknown"),
                                "page": int(chunk.metadata.get("page", 0)),
                                "token": token,
                                "count": count
                            } for token, count in freqs.items())
                        st.session_state.token_freq_df = pd.DataFrame(token_data)
                    st.success("Analysis complete!")

        if "entropy_df" in st.session_state:
            st.subheader("📈 Results")
            st.caption(f"Last run: {st.session_state.last_run}")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                filter_val = st.slider("Min Retrieval Count", 1, 100, 1)
                filtered_df = st.session_state.entropy_df[
                    st.session_state.entropy_df["retrieval_count"] >= filter_val
                ]
                if st.session_state.get("exclude_tokens_runtime"):
                    filtered_df = filtered_df[
                        ~filtered_df["chunk_id"].str.lower().apply(
                            lambda cid: any(tok in cid for tok in st.session_state.exclude_tokens_runtime)
                        )
                    ]
                st.bar_chart(filtered_df.set_index("chunk_id"))
            
            with col2:
                st.plotly_chart(
                    px.imshow(
                        pd.DataFrame(0, index=[f"Query_{i}" for i in range(len(claims))],
                        columns=sorted(set(
                            f"{doc.metadata.get('source', 'unknown')}::Page_{doc.metadata.get('page', '0')}"
                            for claim in claims
                            for doc in st.session_state.retriever.invoke(claim["cpt_code"])
                        )),
                        labels=dict(x="", y="Queries"),
                        color_continuous_scale="YlGnBu",
                        height=400
                    ).update_layout(
                        coloraxis_showscale=False,
                        xaxis=dict(showticklabels=False),
                        yaxis=dict(showgrid=False)
                    ),
                    use_container_width=True
                )
    
    with tab2:
        if "token_freq_df" in st.session_state:
            st.subheader("Token Analysis")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                token = st.selectbox("Token", st.session_state.token_freq_df["token"].unique())
                filtered = st.session_state.token_freq_df[
                    st.session_state.token_freq_df["token"] == token
                ]
                st.metric("Total", filtered["count"].sum())
                st.metric("Unique Docs", filtered["source"].nunique())
                st.dataframe(
                    filtered.groupby("source")["count"].sum().nlargest(5).reset_index(),
                    hide_index=True
                )
            
            with col2:
                st.plotly_chart(
                    px.scatter(
                        filtered,
                        x="source",
                        y="count",
                        color="page",
                        size="count",
                        hover_data=["page", "count"]
                    ),
                    use_container_width=True
                )
        else:
            st.info("Generate analysis first")
    
    with tab3:
        st.subheader("System Diagnostics")
        if "entropy_df" in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                total = st.session_state.entropy_df["retrieval_count"].sum()
                top5 = st.session_state.entropy_df.head(5)["retrieval_count"].sum()
                st.metric("Top 5 Share", f"{top5/total:.1%}")
                st.progress(top5/total)
                if top5/total > 0.6:
                    st.warning("Top chunks dominate retrievals")
            
            with col2:
                params = st.session_state.frozen_params
                st.metric("Chunk Size", params["chunk_size"])
                st.metric("Overlap", params["chunk_overlap"])
                st.metric("FAISS Weight", f"{params['weights'][1]:.1f}")
            
            st.caption("Recommendations")
            if params["chunk_overlap"] < 1000:
                st.warning("Consider increasing chunk overlap")
            if params["weights"][1] > 0.8:
                st.warning("High FAISS weight may reduce diversity")
        else:
            st.info("No diagnostics available yet")