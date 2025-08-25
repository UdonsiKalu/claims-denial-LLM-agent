# ===========================================
# Retrieval Studio — Atomic Redesign (Apple × Braun)
# ===========================================
import os, sys, json, numpy as np, pandas as pd
import streamlit as st
from datetime import datetime
from collections import defaultdict
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Local modules (unchanged behavior)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from strategy_optimizer import grid_search, random_search, bayesian_search
from strategy_profiler import (
    profile_objective, suggest_weights_from_variance,
    dynamic_objective, dynamic_objective_banded,
)
from faiss_gpu_entropy import CMSDenialAnalyzer
from retrieval_optimizer import optimize_retrieval  # if used elsewhere

# ---------- Design Tokens ----------
COLORS = {
    "primary": "#0071E3", "primary_hover": "#0062CC", "destructive": "#FF453A",
    "bg_primary": "#F5F5F7", "card": "#FFFFFF", "border": "#D2D2D7",
    "text_primary": "#1D1D1F", "text_secondary": "#86868B",
    "log_bg": "#0A0A0A", "log_text": "#30D158"
}
BASE_SPACING = 8  # 8px baseline grid

# ---------- Page Setup ----------
st.set_page_config(page_title="Retrieval Studio", layout="wide", initial_sidebar_state="expanded")

# Inject styles.css + minimal inline fixes
def inject_global_styles():
    try:
        with open("styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass
    # Tab spacing enhancement (spec requires)
    st.markdown("""
    <style>
      div[data-baseweb="tab-list"]{ gap: 8px !important; }
      div[data-baseweb="tab"]{ padding:8px 16px !important; border-radius:8px !important; transition: all .2s !important; }
    </style>
    """, unsafe_allow_html=True)

inject_global_styles()

# ---------- Helpers ----------
def ui_card(title: str, body_renderer=None, width: str|None=None):
    style = f'width:{width};' if width else ''
    st.markdown(f"""
    <div class="rs-card" style="{style}">
      <h3 style="margin-top:0;color:{COLORS['text_primary']};font-weight:600">{title}</h3>
    </div>""", unsafe_allow_html=True)
    # Immediately render into the last container via placeholder:
    return st.container()

def pill(label, value):
    st.markdown(f"""<span class="rs-pill"><strong>{label}</strong> {value}</span>""",
                unsafe_allow_html=True)

def plotly_gridless(fig):
    fig.update_layout(
        plot_bgcolor=COLORS['card'],
        paper_bgcolor=COLORS['card'],
        xaxis_showgrid=False, yaxis_showgrid=False,
        margin=dict(l=24, r=24, t=40, b=24)
    )
    return fig

# ---------- Session Defaults ----------
if "log_lines" not in st.session_state: st.session_state["log_lines"] = []
if "retriever_initialized" not in st.session_state: st.session_state["retriever_initialized"] = False
if "rebuild_index" not in st.session_state: st.session_state["rebuild_index"] = False
if "frozen_params" not in st.session_state:
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

# ---------- Cached Analyzer Loader ----------
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
        weights=params["weights"],
    )

# ---------- Terminal Logger ----------
def append_log(text: str):
    st.session_state["log_lines"].append(text)
    log_text = "\n".join(st.session_state["log_lines"])
    st.markdown(f'<div class="rs-terminal">{log_text}</div>', unsafe_allow_html=True)

# ---------- Header ----------
st.title("Retrieval Studio")
st.caption("Clinical precision × Apple aesthetics × Braun minimalism")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio("App Mode", ["Manual", "Optimizer"], horizontal=True, label_visibility="collapsed", key="app_mode")

    if mode == "Manual":
        with st.expander("⚙️ Chunking", expanded=True):
            ui_chunking_strategy = st.selectbox("Strategy", ["Fixed-size", "Header-aware", "Semantic"], key="chunk_strategy")
            ui_chunk_size = st.number_input("Chunk Size", 1000, 20000, 10000, 500, key="chunk_size", help="Characters per chunk")
            ui_chunk_overlap = st.number_input("Chunk Overlap", 0, 5000, 2000, 100, key="chunk_overlap")

        with st.expander("🔎 Retriever", expanded=True):
            ui_faiss_k = st.number_input("FAISS Top-k", 1, 20, 5, key="faiss_k")
            ui_bm25_k = st.number_input("BM25 Top-k", 1, 20, 3, key="bm25_k")
            ui_faiss_fetch_k = st.number_input("FAISS Fetch_k (MMR)", ui_faiss_k, 100, 50, key="faiss_fetch_k")
            ui_faiss_weight = st.number_input("Weight: FAISS", 0.0, 1.0, 0.6, 0.05, key="faiss_weight")
            ui_weights = (1.0 - ui_faiss_weight, ui_faiss_weight)

        with st.expander("🚫 Exclusions"):
            exclude_pages = st.text_area("Patterns to exclude (comma-separated)", "Page_0,Introduction,Scope,Purpose")
            exclude_tokens = [x.strip().lower() for x in exclude_pages.split(",") if x.strip()]
            st.session_state["exclude_tokens_runtime"] = exclude_tokens

        rechunk = st.button("🚀 Rechunk & Reindex", use_container_width=True)
        if rechunk:
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

    else:
        with st.expander("🎯 Target Profile", expanded=True):
            st.session_state["target_profile"] = {
                "query_entropy_range": st.slider("Query Entropy Range", 0.0, 1.0, (0.7, 0.9), step=0.01, key="entropy_range"),
                "max_chunk_frequency": st.slider("Max Chunk Frequency", 0.01, 0.5, 0.1, key="max_chunk_freq"),
                "gini_threshold": st.slider("Gini Coefficient", 0.0, 1.0, 0.4, key="gini_threshold"),
                "required_code_coverage": st.slider("Code Coverage Threshold", 0.0, 1.0, 0.95, key="coverage_threshold"),
            }

        st.markdown("### Optimization Method")
        search_method = st.selectbox("Select", ["Grid Search", "Random Search", "Bayesian Optimization"], key="search_method")
        if search_method == "Random Search":
            n_iter = st.slider("Random Iterations", 5, 50, 12, key="rand_n")
        elif search_method == "Bayesian Optimization":
            n_calls = st.slider("Bayesian Calls", 5, 30, 15, key="bayes_n")

# ---------- Tabs ----------
tabs = st.tabs(["📊 Entropy Analysis", "🔍 Token Frequencies", "⚙️ Diagnostics"])

# ---------- Analyzer Ready ----------
if (not st.session_state["retriever_initialized"]) or st.session_state["rebuild_index"]:
    with st.spinner("Initializing retriever..."):
        analyzer = load_analyzer(st.session_state["frozen_params"], force_rebuild=st.session_state["rebuild_index"])
        st.session_state["analyzer"] = analyzer
        st.session_state["retriever"] = analyzer.retrieval["retriever"]
        st.session_state["retriever_initialized"] = True
        st.session_state["rebuild_index"] = False
        st.success("Retriever is ready.")

retriever = st.session_state.get("retriever")

# ---------- Tab 1: Entropy Analysis ----------
with tabs[0]:
    left, right = st.columns([1, 2], gap="medium")

    with left:
        card = ui_card("Upload Claims")
        with card:
            claims_file = st.file_uploader(
                "Upload Claims", type="jsonl",
                label_visibility="collapsed", accept_multiple_files=False
            )

        run_col, _ = st.columns([1,1])
        with run_col:
            run_entropy = st.button("Generate Entropy Map", key="generate_entropy")

    with right:
        meta = ui_card("Run Details")
        with meta:
            fp = st.session_state["frozen_params"]
            pill("Strategy", fp["chunking_strategy"])
            pill("Chunk", f"{fp['chunk_size']} (+{fp['chunk_overlap']})")
            pill("Top-k", f"FAISS {fp['faiss_k']} / BM25 {fp['bm25_k']}")
            pill("Weights", f"{fp['weights'][0]:.1f},{fp['weights'][1]:.1f}")

    if claims_file:
        try:
            claims = [json.loads(line) for line in claims_file]
            st.success(f"Loaded {len(claims)} claims")
        except Exception as e:
            st.error(f"Failed to load claims: {e}")
            claims = []
    else:
        claims = []

    if run_entropy and claims and retriever:
        retrieval_log = defaultdict(int)
        token_data = []
        with st.spinner("Running retrieval and frequency analysis..."):
            # Build retrieval frequency & token frequencies using your analyzer
            for claim in claims:
                try:
                    docs = retriever.invoke(claim["cpt_code"])
                    for doc in docs:
                        source = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "0")
                        chunk_id = doc.metadata.get("chunk_id") or f"{source}::Page_{page}"
                        retrieval_log[chunk_id] += 1
                except Exception as e:
                    append_log(f"Error on {claim.get('cpt_code')}: {e}")

            entropy_df = pd.DataFrame([{"chunk_id": k, "retrieval_count": v} for k, v in retrieval_log.items()])
            entropy_df.sort_values(by="retrieval_count", ascending=False, inplace=True)
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

    # Results
    if "entropy_df" in st.session_state:
        st.markdown(f"_Last run: {st.session_state['last_run']}_")

        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            filter_val = st.slider("📉 Min Retrieval Count", 1, 100, 1, key="retrieval_filter")
            df = st.session_state["entropy_df"]
            filtered_df = df[df["retrieval_count"] >= filter_val]

            runtime_excludes = st.session_state.get("exclude_tokens_runtime", [])
            if runtime_excludes:
                filtered_df = filtered_df[
                    ~filtered_df["chunk_id"].str.lower().apply(
                        lambda cid: any(tok in cid for tok in runtime_excludes)
                    )
                ]

            fig_bar = px.bar(filtered_df, x="chunk_id", y="retrieval_count", title="Retrieval Frequency")
            plotly_gridless(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            # Query vs Chunk matrix (gridless)
            if claims and retriever:
                query_chunk_matrix = defaultdict(lambda: defaultdict(int))
                query_ids = []; chunk_ids = set()
                for i, claim in enumerate(claims):
                    qid = f"Query_{i}"; query_ids.append(qid)
                    try:
                        docs = retriever.invoke(claim["cpt_code"])
                        for doc in docs:
                            cid = doc.metadata.get("chunk_id") or f"{doc.metadata.get('source','unknown')}::Page_{doc.metadata.get('page','0')}"
                            if any(tok in cid.lower() for tok in st.session_state.get("exclude_tokens_runtime", [])):
                                continue
                            query_chunk_matrix[qid][cid] += 1
                            chunk_ids.add(cid)
                    except Exception as e:
                        append_log(f"Matrix build error on {claim.get('cpt_code')}: {e}")

                chunk_ids = sorted(chunk_ids)
                matrix_df = pd.DataFrame(0, index=query_ids, columns=chunk_ids)
                for q in query_ids:
                    for c in query_chunk_matrix[q]:
                        matrix_df.at[q, c] = query_chunk_matrix[q][c]

                matrix_display_df = matrix_df.copy()
                max_val = matrix_display_df.values.max()
                if max_val > 10:
                    matrix_display_df = matrix_display_df.clip(upper=10)

                fig = px.imshow(matrix_display_df, labels=dict(x="", y="Queries"), color_continuous_scale="YlGnBu", aspect="auto", height=420)
                plotly_gridless(fig).update_layout(coloraxis_showscale=False)
                fig.update_xaxes(showticklabels=False)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📄 View Raw Matrix Data"):
                    st.dataframe(matrix_df, use_container_width=True)
                    csv_data = matrix_df.reset_index().rename(columns={"index": "Query"}).to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Matrix CSV", csv_data, file_name="query_chunk_matrix.csv", use_container_width=True)

# ---------- Tab 2: Token Frequencies ----------
with tabs[1]:
    if "token_freq_df" in st.session_state and not st.session_state["token_freq_df"].empty:
        tf = st.session_state["token_freq_df"]

        left, right = st.columns([1, 3], gap="large")
        with left:
            token_to_view = st.selectbox("Select Token", sorted(tf["token"].unique()))
            subset = tf[tf["token"] == token_to_view]
            st.metric("Total Occurrences", int(subset["count"].sum()))
            st.metric("Unique Documents", int(subset["source"].nunique()))
            top_sources = subset.groupby("source")["count"].sum().nlargest(5)
            st.write("**Top Sources**")
            st.dataframe(top_sources.reset_index(), hide_index=True, use_container_width=True)

        with right:
            st.markdown(f"### Frequency Heatmap for `{token_to_view}`")
            pivot_freq = subset.pivot_table(index="source", columns="page", values="count", fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_freq, annot=False, cmap="Blues", ax=ax)
            ax.set_title(f"Frequency Map: '{token_to_view}'"); ax.set_xlabel("Page"); ax.set_ylabel("Source")
            st.pyplot(fig)
    else:
        st.info("Load a claims file and run Entropy Analysis to view token frequencies.")

# ---------- Tab 3: Diagnostics ----------
with tabs[2]:
    params = st.session_state["frozen_params"]
    grid = st.columns(2, gap="large")

    with grid[0]:
        card = ui_card("Retrieval Distribution")
        with card:
            if "entropy_df" in st.session_state:
                df = st.session_state["entropy_df"]
                top_chunks = df.head(5)
                if not top_chunks.empty:
                    total = df["retrieval_count"].sum(); top_total = top_chunks["retrieval_count"].sum()
                    percent = (top_total / total) * 100 if total else 0
                    st.metric("Top 5 Chunk Share", f"{percent:.1f}%")
                    st.progress(min(1.0, percent/100))
                    if percent > 60: st.warning("Top chunks dominate retrievals. Consider semantic splitting.")
                    else: st.success("Retrieval distribution is balanced.")
            else:
                st.info("Run Entropy Analysis to compute distribution stats.")

    with grid[1]:
        card = ui_card("Configuration Check")
        with card:
            st.write(f"**Chunk Size:** {params['chunk_size']}")
            st.write(f"**Chunk Overlap:** {params['chunk_overlap']}")
            st.write(f"**FAISS Weight:** {params['weights'][1]:.1f}")
            if params["chunk_overlap"] < 1000: st.warning("Low chunk overlap may lose boundary context.")
            if params["weights"][1] > 0.8: st.warning("High FAISS weight may reduce diversity.")

    st.markdown("### Recommendations")
    recs = []
    if "entropy_df" in st.session_state:
        df = st.session_state["entropy_df"]; top_chunks = df.head(5)
        if not top_chunks.empty:
            total = df["retrieval_count"].sum(); top_total = top_chunks["retrieval_count"].sum()
            percent = (top_total / total) * 100 if total else 0
            if percent > 60: recs.append(f"⚠️ Top 5 chunks account for {percent:.1f}% of all retrievals. Try semantic splitting.")
            if any("unknown" in cid for cid in top_chunks["chunk_id"]): recs.append("📌 Missing metadata: add `source` and `page` fields.")
            if params["chunk_overlap"] < 1000: recs.append("🔁 Chunk overlap is low. Consider increasing to preserve boundary context.")
    if recs: [st.warning(r) for r in recs]
    else: st.info("✅ Retrieval distribution is healthy.")

# =======================
# Optimizer Mode Section
# =======================
if st.session_state["app_mode"] == "Optimizer":
    st.markdown("---")
    st.header("🤖 Retrieval Strategy Optimizer")

    claims_file_opt = st.file_uploader("Upload Synthetic Claims (JSONL)", type="jsonl", key="opt_claims")
    claims_opt = []
    if claims_file_opt:
        try:
            claims_opt = [json.loads(line) for line in claims_file_opt]
            st.success(f"✅ Loaded {len(claims_opt)} claims")
        except Exception as e:
            st.error(f"❌ Failed to parse file: {e}")

    target_profile = st.session_state["target_profile"]
    search_method = st.session_state["search_method"]

    # Required param grid (weights must be strings)
    param_grid = {
        "chunk_size": [5000, 7500, 10000, 15000],
        "weights": ["0.4,0.6", "0.5,0.5", "0.6,0.4"],  # keep spec exact
        # (You can keep your extra knobs; they won't break spec)
        "chunk_overlap": [500, 1000, 2000, 3000],
        "faiss_k": list(range(3, 8)),
        "bm25_k": list(range(2, 6)),
        "fetch_k": [25, 50],
    }

    # Pills visualization (compact)
    pill_row = st.container()
    with pill_row:
        st.markdown("**Parameter Grid**")
        for k, v in param_grid.items():
            pill(k, f"{len(v)} options" if isinstance(v, list) else v)

    def evaluate_strategy(params, claims, target_profile, idx=None, total=None):
        # Basic progress note
        if idx is not None and total is not None:
            append_log(f"⚙️ Evaluating strategy {idx}/{total}: {params}")

        # Weights parsing
        raw_weights = params["weights"]
        weights = tuple(map(float, raw_weights.split(","))) if isinstance(raw_weights, str) else raw_weights

        # Chunking env
        chunk_size = int(params.get("chunk_size", 10000))
        chunk_overlap = int(params.get("chunk_overlap", 2000))
        os.environ["CHUNK_STRATEGY"] = "Fixed-size"
        os.environ["CHUNK_SIZE"] = str(chunk_size)
        os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
        os.environ["FORCE_REBUILD"] = "1"
        os.environ["QDRANT_COLLECTION"] = f"qdrant_c{chunk_size}_o{chunk_overlap}"

        # Strategy id
        strategy_id = f"faiss{params.get('faiss_k')}_bm25{params.get('bm25_k')}_fetch{params.get('fetch_k')}_w{params['weights']}_c{chunk_size}_o{chunk_overlap}"
        params["strategy_id"] = strategy_id

        analyzer = CMSDenialAnalyzer(
            exclude_tokens=[],
            faiss_k=int(params.get("faiss_k", 5)),
            bm25_k=int(params.get("bm25_k", 3)),
            faiss_fetch_k=int(params.get("fetch_k", 50)),
            weights=weights
        )

        append_log(f"🔎 Running queries for {len(claims)} claims...")
        result = analyzer.evaluate_entropy_score(
            claims, target_profile, log_fn=append_log, idx=idx, total=total
        )
        # Preserve id & knobs
        result["strategy_id"] = strategy_id
        result["chunk_size"] = chunk_size
        result["chunk_overlap"] = chunk_overlap
        result["faiss_k"] = int(params.get("faiss_k", 5))
        result["bm25_k"] = int(params.get("bm25_k", 3))
        result["fetch_k"] = int(params.get("fetch_k", 50))
        result["weights"] = params["weights"]

        # Extra quick log
        append_log(
            "📊 Metrics: "
            f"entropy={result.get('entropy'):.4f}, "
            f"max_freq={result.get('max_freq'):.4f}, "
            f"gini={result.get('gini'):.4f}, "
            f"coverage={result.get('coverage'):.4f}, "
            f"final_score={result.get('score'):.4f}"
        )
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

    # Run optimizer
    go_opt = st.button("🚀 Run Optimizer", disabled=not bool(claims_opt))
    if go_opt:
        st.session_state["log_lines"] = []  # clear
        append_log("🚀 Starting optimization run...\n")

        with st.spinner("Evaluating strategy combinations..."):
            # Quick objective profiling
            append_log("🧪 Profiling objective for variance...")
            profile = profile_objective(param_grid, lambda p: evaluate_strategy(p, claims_opt, target_profile), n=12)
            append_log(f"📊 Profiling stats:\n{json.dumps(profile['stats'], indent=2)}")

            if profile.get("score_flat"):
                append_log("⚠️ Objective looks flat. Switching to data-driven dynamic objective.")
                weights_suggested = suggest_weights_from_variance(profile["stats"])
                append_log(f"🧭 Suggested metric weights: {json.dumps(weights_suggested, indent=2)}")
                use_banded = True
                entropy_band = target_profile.get("query_entropy_range", (0.7, 0.9))

                def scorer_for_optimizer(p, i=None, t=None):
                    r = evaluate_strategy(p, claims_opt, target_profile, idx=i, total=t)
                    r["score"] = dynamic_objective_banded(r, weights_suggested, entropy_band) if use_banded \
                                 else dynamic_objective(r, weights_suggested)
                    return r
            else:
                append_log("✅ Objective has usable variance. Keeping base score.")
                def scorer_for_optimizer(p, i=None, t=None):
                    return evaluate_strategy(p, claims_opt, target_profile, idx=i, total=t)

            # Dispatch
            if search_method == "Grid Search":
                full_history = grid_search(param_grid, lambda p: scorer_for_optimizer(p), log_fn=append_log)
            elif search_method == "Random Search":
                full_history = random_search(param_grid, lambda p: scorer_for_optimizer(p), n_iter=st.session_state.get("rand_n", 12), log_fn=append_log)
            else:
                full_history = bayesian_search(param_grid, lambda p, i=None, t=None: scorer_for_optimizer(p, i, t),
                                               n_calls=st.session_state.get("bayes_n", 15), log_fn=append_log)

        # Results table + plots
        if full_history:
            df = pd.DataFrame(full_history)

            # Top strategy
            best = max(full_history, key=lambda x: x["score"])
            st.success(f"✅ Best Score: {best['score']:.4f}")
            top_card = ui_card("Top Strategy (Applied?)")
            with top_card:
                st.json(best, expanded=False)

            # Numbered Top 5 as cards
            st.markdown("#### Top 5 Strategies")
            top5 = sorted(full_history, key=lambda x: x['score'], reverse=True)[:5]
            for i, strategy in enumerate(top5, start=1):
                card = ui_card(f"Strategy #{i} — Score: {strategy['score']:.2f}")
                with card:
                    st.json(strategy, expanded=False)

            # Plots (gridless)
            st.subheader("📊 Strategy Scores")
            fig_bar = px.bar(df, x="strategy_id", y="score", text="score", title="Score per Strategy",
                             labels={"strategy_id": "Strategy ID", "score": "Score"})
            fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_bar.update_layout(yaxis_range=[0, 1], height=400)
            plotly_gridless(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True)

            # Entropy vs Coverage scatter
            if {"entropy", "coverage"}.issubset(df.columns):
                fig_sc = px.scatter(df, x="entropy", y="coverage", color="score",
                                    hover_data=["strategy_id"], title="Entropy vs Coverage",
                                    color_continuous_scale="Viridis")
                fig_sc.update_layout(height=400)
                plotly_gridless(fig_sc)
                st.plotly_chart(fig_sc, use_container_width=True)

            # Gini vs Score
            if {"gini", "score"}.issubset(df.columns):
                fig_g = px.scatter(df, x="gini", y="score", color="score",
                                   hover_data=["strategy_id"], title="Gini vs Score",
                                   color_continuous_scale="Plasma")
                fig_g.update_layout(height=400)
                plotly_gridless(fig_g)
                st.plotly_chart(fig_g, use_container_width=True)

            # Apply best button
            if st.button("📥 Apply Best Config and Rebuild Index"):
                apply_best_config_to_session(best)
                try:
                    load_analyzer.clear()  # clear cache
                except Exception:
                    pass
                with st.spinner("Rebuilding index with best configuration..."):
                    analyzer = load_analyzer(st.session_state["frozen_params"], force_rebuild=True)
                    st.session_state["retriever"] = analyzer.retrieval["retriever"]
                    st.session_state["analyzer"] = analyzer
                    st.success("✅ Index rebuilt with best configuration.")

        # Live log console
        st.markdown("### Live Optimizer Log")
        append_log("Ready.\n")

