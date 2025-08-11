# === PAGE CONFIG MUST BE FIRST ===
import streamlit as st
st.set_page_config(page_title="Retrieval Studio", layout="wide", page_icon="🔍")

# === IMPORTS ===
import os, json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import plotly.express as px

from strategy_optimizer import grid_search, random_search, bayesian_search
from faiss_gpu_entropy import CMSDenialAnalyzer
from strategy_profiler import (
    profile_objective, suggest_weights_from_variance,
    dynamic_objective, dynamic_objective_banded
)

# === CUSTOM CSS ===
def inject_design():
    st.markdown("""
    <style>
        :root {
            --primary: #0071E3;
            --primary-hover: #0062C4;
            --text-primary: #1D1D1F;
            --text-secondary: #86868B;
            --bg-primary: #F5F5F7;
            --card-bg: #FFFFFF;
            --border: #D2D2D7;
            --radius: 12px;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
        }

        html, body, .stApp {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
        }

        /* Sidebar container */
        section[data-testid="stSidebar"] .block-container{
            padding: 16px 12px !important;
        }
        section[data-testid="stSidebar"] .stMarkdown h1, 
        section[data-testid="stSidebar"] .stMarkdown h2, 
        section[data-testid="stSidebar"] .stMarkdown h3{
            margin: 0 0 8px 0;
        }
        section[data-testid="stSidebar"] .stContainer{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow-sm);
            padding: 16px;
            margin-bottom: 12px;
        }

        /* Cards in main area */
        .stContainer {
            background: var(--card-bg);
            border-radius: var(--radius);
            box-shadow: var(--shadow-sm);
            padding: 1.5rem;
            margin-bottom: 1rem;
        }

        .stButton>button {
            background: var(--primary) !important;
            color: white !important;
            border: none !important;
            border-radius: var(--radius) !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.2s ease !important;
        }
        .stButton>button:hover { background: var(--primary-hover) !important; transform: translateY(-1px); }

        .stSelectbox, .stTextInput, .stNumberInput, .stSlider, .stMultiSelect { border-radius: var(--radius) !important; }

        .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1rem;
            border-radius: var(--radius);
            transition: all 0.2s ease;
            border: 1px solid var(--border);
            background: var(--card-bg);
        }
        .stTabs [aria-selected="true"] {
            background: var(--primary) !important;
            color: white !important;
            border-color: var(--primary) !important;
        }

        .metric-card {
            background: var(--card-bg);
            border-radius: var(--radius);
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

inject_design()

# === CORE HELPERS ===
def plot_optimizer_results(df):
    """Enhanced visualization of optimizer results"""
    with st.expander("Strategy Performance", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(df, x="strategy_id", y="score", title="Score Distribution", text="score")
            fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_bar.update_layout(
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                xaxis_showgrid=False, yaxis_showgrid=False, height=380
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            if {"entropy", "coverage"}.issubset(df.columns):
                fig_scatter = px.scatter(
                    df, x="entropy", y="coverage",
                    color="score", size="score",
                    hover_data=["strategy_id"], color_continuous_scale="Viridis",
                    title="Entropy vs Coverage"
                )
                fig_scatter.update_layout(
                    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                    xaxis_showgrid=False, yaxis_showgrid=False, height=380
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

@st.cache_resource
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

# === MAIN APP ===
def main():
    # init session
    st.session_state.setdefault("log_lines", [])
    st.session_state.setdefault("retriever_initialized", False)
    st.session_state.setdefault("rebuild_index", False)
    st.session_state.setdefault("frozen_params", {
        "chunking_strategy": "Fixed-size",
        "chunk_size": 10000,
        "chunk_overlap": 2000,
        "faiss_k": 5,
        "bm25_k": 3,
        "faiss_fetch_k": 50,
        "weights": (0.4, 0.6),
        "exclude_tokens": ["page_0", "introduction", "scope", "purpose"]
    })

    # === SIDEBAR ===
    with st.sidebar:
        st.title("Configuration")
        st.session_state["app_mode"] = st.radio("Mode", ["Manual", "Optimizer"], horizontal=True)

        if st.session_state["app_mode"] == "Manual":
            render_manual_sidebar()
        else:
            # The return value must be captured & stored to session
            tp = render_optimizer_sidebar()
            st.session_state["target_profile"] = tp  # <- wire into session for use in content

    # === MAIN CONTENT ===
    st.title("Retrieval Studio")

    if st.session_state["app_mode"] == "Optimizer":
        render_optimizer_content()
    else:
        render_manual_content()

# --- Sidebar blocks ---
def render_manual_sidebar():
    with st.container():
        st.subheader("Chunking Settings")
        st.session_state["frozen_params"]["chunking_strategy"] = st.selectbox(
            "Strategy", ["Fixed-size", "Header-aware", "Semantic"]
        )
        st.session_state["frozen_params"]["chunk_size"] = st.number_input(
            "Chunk Size", 1000, 20000, 10000, 500
        )
        st.session_state["frozen_params"]["chunk_overlap"] = st.number_input(
            "Overlap", 0, 5000, 2000, 100
        )

    with st.container():
        st.subheader("Retriever Settings")
        st.session_state["frozen_params"]["faiss_k"] = st.number_input("FAISS Top-k", 1, 20, 5)
        st.session_state["frozen_params"]["bm25_k"] = st.number_input("BM25 Top-k", 1, 20, 3)
        st.session_state["frozen_params"]["faiss_fetch_k"] = st.number_input("FAISS Fetch_k", 5, 100, 50)
        faiss_weight = st.number_input("FAISS Weight", 0.0, 1.0, 0.6, 0.05)
        st.session_state["frozen_params"]["weights"] = (1.0 - faiss_weight, faiss_weight)

    with st.container():
        st.subheader("Exclusion Settings")
        exclude_tokens = [
            x.strip().lower()
            for x in st.text_area("Exclude patterns", "Page_0,Introduction,Scope,Purpose").split(",")
            if x.strip()
        ]
        st.session_state["frozen_params"]["exclude_tokens"] = exclude_tokens

    if st.button("Apply Configuration", type="primary", use_container_width=True):
        st.session_state["rebuild_index"] = True
        st.rerun()

def render_optimizer_sidebar():
    st.subheader("Target Profile")
    return {
        "query_entropy_range": st.slider("Entropy Range", 0.0, 1.0, (0.7, 0.9)),
        "max_chunk_frequency": st.slider("Max Frequency", 0.01, 0.5, 0.1),
        "gini_threshold": st.slider("Gini Threshold", 0.0, 1.0, 0.4),
        "required_code_coverage": st.slider("Coverage", 0.0, 1.0, 0.95),
    }

# --- Manual content ---
def render_manual_content():
    tab1, tab2, tab3 = st.tabs(["Entropy Analysis", "Token Frequencies", "Diagnostics"])

    # init or rebuild
    if (not st.session_state.get("retriever_initialized")) or st.session_state.get("rebuild_index"):
        with st.spinner("Initializing retriever..."):
            analyzer = load_analyzer(
                st.session_state["frozen_params"],
                force_rebuild=st.session_state.get("rebuild_index", False)
            )
            st.session_state.update({
                "retriever": analyzer.retrieval["retriever"],
                "analyzer": analyzer,
                "retriever_initialized": True,
                "rebuild_index": False
            })

    retriever = st.session_state.get("retriever")

    # === TAB 1: ENTROPY ===
    with tab1:
        st.subheader("Upload Claims")
        claims_file = st.file_uploader("Upload JSONL", type="jsonl", accept_multiple_files=False, label_visibility="collapsed")
        run = st.button("Generate Entropy Map")
        claims = []
        if claims_file:
            try:
                claims = [json.loads(line) for line in claims_file]
                st.success(f"Loaded {len(claims)} claims")
            except Exception as e:
                st.error(f"Failed to load claims: {e}")

        if run and claims and retriever:
            retrieval_log = defaultdict(int)
            token_data = []
            with st.spinner("Analyzing..."):
                for claim in claims:
                    try:
                        docs = retriever.invoke(claim["cpt_code"])
                        for doc in docs:
                            src = doc.metadata.get("source", "unknown")
                            page = doc.metadata.get("page", "0")
                            cid = doc.metadata.get("chunk_id") or f"{src}::Page_{page}"
                            retrieval_log[cid] += 1
                    except Exception as e:
                        st.warning(f"Query error: {e}")

                entropy_df = pd.DataFrame([{"chunk_id": k, "retrieval_count": v} for k, v in retrieval_log.items()]).sort_values(
                    "retrieval_count", ascending=False
                )
                st.session_state["entropy_df"] = entropy_df
                st.session_state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                analyzer = st.session_state.get("analyzer")
                if analyzer:
                    analyzer.compute_token_frequencies()
                    for ch in analyzer.chunks:
                        freqs = ch.metadata.get("token_frequencies", {})
                        for token, count in freqs.items():
                            token_data.append({
                                "source": ch.metadata.get("source", "unknown"),
                                "page": int(ch.metadata.get("page", 0)),
                                "token": token, "count": count
                            })
                    st.session_state["token_freq_df"] = pd.DataFrame(token_data)

        if "entropy_df" in st.session_state:
            st.caption(f"Last run: {st.session_state['last_run']}")
            df = st.session_state["entropy_df"]
            min_count = st.slider("Min Retrieval Count", 1, max(1, int(df["retrieval_count"].max())), 1)
            view = df[df["retrieval_count"] >= min_count]
            fig = px.bar(view, x="chunk_id", y="retrieval_count", title="Retrieval Frequency", text="retrieval_count")
            fig.update_traces(textposition="outside")
            fig.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", xaxis_showgrid=False, yaxis_showgrid=False, height=420)
            st.plotly_chart(fig, use_container_width=True)

    # === TAB 2: TOKEN FREQUENCIES ===
    with tab2:
        tf = st.session_state.get("token_freq_df")
        if tf is not None and not tf.empty:
            st.subheader("Token Frequency Analysis")
            token = st.selectbox("Token", sorted(tf["token"].unique()))
            subset = tf[tf["token"] == token]
            colA, colB = st.columns(2)
            with colA: st.metric("Total Occurrences", int(subset["count"].sum()))
            with colB: st.metric("Unique Documents", int(subset["source"].nunique()))
            pivot = subset.pivot_table(index="source", columns="page", values="count", fill_value=0)
            heat = px.imshow(pivot, color_continuous_scale="Blues", title=f"Frequency Map: '{token}'")
            heat.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", xaxis_showgrid=False, yaxis_showgrid=False, height=420)
            st.plotly_chart(heat, use_container_width=True)
        else:
            st.info("Run Entropy Analysis to populate token frequencies.")

    # === TAB 3: DIAGNOSTICS ===
    with tab3:
        st.subheader("Diagnostics")
        params = st.session_state["frozen_params"]
        st.write(f"**Chunk Size:** {params['chunk_size']}  |  **Overlap:** {params['chunk_overlap']}  |  **FAISS Weight:** {params['weights'][1]:.1f}")
        if "entropy_df" in st.session_state:
            df = st.session_state["entropy_df"]
            if not df.empty:
                total = df["retrieval_count"].sum()
                top5 = df.head(5)["retrieval_count"].sum()
                share = (top5/total)*100 if total else 0
                st.metric("Top 5 Chunk Share", f"{share:.1f}%")
                if share > 60:
                    st.warning("Top chunks dominate retrievals. Consider semantic splitting.")
                else:
                    st.success("Retrieval distribution looks balanced.")

# --- Optimizer content ---
def render_optimizer_content():
    st.header("Strategy Optimization")
    claims_file = st.file_uploader("Upload Synthetic Claims (JSONL)", type="jsonl", key="opt_claims")
    claims = []
    if claims_file:
        try:
            claims = [json.loads(line) for line in claims_file]
            st.success(f"✅ Loaded {len(claims)} claims")
        except Exception as e:
            st.error(f"❌ Failed to parse file: {e}")

    param_grid = {
        "chunk_size": [5000, 7500, 10000, 15000],
        "weights": ["0.4,0.6", "0.5,0.5", "0.6,0.4"],
        "chunk_overlap": [500, 1000, 2000, 3000],
        "faiss_k": list(range(3, 8)),
        "bm25_k": list(range(2, 6)),
        "fetch_k": [25, 50],
    }

    method = st.selectbox("Search Method", ["Grid Search", "Random Search", "Bayesian Optimization"])
    n_iter = st.slider("Random Iterations", 5, 50, 12) if method == "Random Search" else None
    n_calls = st.slider("Bayesian Calls", 5, 30, 15) if method == "Bayesian Optimization" else None

    tp = st.session_state.get("target_profile", {
        "query_entropy_range": (0.7, 0.9),
        "max_chunk_frequency": 0.1,
        "gini_threshold": 0.4,
        "required_code_coverage": 0.95
    })

    def evaluate_strategy(params, claims, target_profile, idx=None, total=None):
        # weights
        raw_w = params["weights"]
        weights = tuple(map(float, raw_w.split(","))) if isinstance(raw_w, str) else raw_w
        # env for chunking
        csize = int(params.get("chunk_size", 10000))
        cover = int(params.get("chunk_overlap", 2000))
        os.environ.update({
            "CHUNK_STRATEGY": "Fixed-size",
            "CHUNK_SIZE": str(csize),
            "CHUNK_OVERLAP": str(cover),
            "FORCE_REBUILD": "1",
            "QDRANT_COLLECTION": f"qdrant_c{csize}_o{cover}"
        })
        sid = f"faiss{params.get('faiss_k')}_bm25{params.get('bm25_k')}_fetch{params.get('fetch_k')}_w{params['weights']}_c{csize}_o{cover}"
        analyzer = CMSDenialAnalyzer(
            exclude_tokens=[],
            faiss_k=int(params.get("faiss_k", 5)),
            bm25_k=int(params.get("bm25_k", 3)),
            faiss_fetch_k=int(params.get("fetch_k", 50)),
            weights=weights
        )
        res = analyzer.evaluate_entropy_score(claims, target_profile)
        res.update({
            "strategy_id": sid, "chunk_size": csize, "chunk_overlap": cover,
            "faiss_k": int(params.get("faiss_k", 5)),
            "bm25_k": int(params.get("bm25_k", 3)),
            "fetch_k": int(params.get("fetch_k", 50)),
            "weights": params["weights"]
        })
        return res

    go = st.button("🚀 Run Optimizer", disabled=not bool(claims))
    if go:
        st.info("Profiling objective…")
        profile = profile_objective(param_grid, lambda p: evaluate_strategy(p, claims, tp), n=12)
        if profile.get("score_flat"):
            sw = suggest_weights_from_variance(profile["stats"])
            band = tp.get("query_entropy_range", (0.7, 0.9))
            def scorer(p): 
                r = evaluate_strategy(p, claims, tp)
                r["score"] = dynamic_objective_banded(r, sw, band)
                return r
        else:
            def scorer(p): return evaluate_strategy(p, claims, tp)

        if method == "Grid Search":
            hist = grid_search(param_grid, scorer)
        elif method == "Random Search":
            hist = random_search(param_grid, scorer, n_iter=n_iter)
        else:
            hist = bayesian_search(param_grid, scorer, n_calls=n_calls)

        if hist:
            df = pd.DataFrame(hist)
            best = max(hist, key=lambda x: x["score"])
            st.success(f"Best Score: {best['score']:.4f}")
            st.json(best, expanded=False)
            plot_optimizer_results(df)

            if st.button("📥 Apply Best & Rebuild Index"):
                st.session_state["frozen_params"] = {
                    "chunking_strategy": "Fixed-size",
                    "chunk_size": best.get("chunk_size", 10000),
                    "chunk_overlap": best.get("chunk_overlap", 2000),
                    "faiss_k": best["faiss_k"],
                    "bm25_k": best["bm25_k"],
                    "faiss_fetch_k": best["fetch_k"],
                    "weights": tuple(map(float, best["weights"].split(","))),
                    "exclude_tokens": st.session_state["frozen_params"].get("exclude_tokens", [])
                }
                st.session_state["rebuild_index"] = True
                st.experimental_rerun()

if __name__ == "__main__":
    main()
