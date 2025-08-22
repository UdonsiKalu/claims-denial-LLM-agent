import os
import json
import itertools
from datetime import datetime
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from faiss_gpu_entropy import CMSDenialAnalyzer

LOG_PATH = "logs/optimization_log.jsonl"
os.makedirs("logs", exist_ok=True)

def append_log(text):
    ctx = get_script_run_ctx()
    if ctx and "log_lines" in st.session_state and "log_box" in st.session_state:
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

def optimize_retrieval(claims_data, target_profile, log_fn=None):
    chunking_strategies = ["Fixed-size", "Header-aware"]
    chunk_sizes = [8000, 10000]
    overlaps = [1000, 2000]
    faiss_ks = [3, 5]
    bm25_ks = [2, 3]
    fetch_ks = [25, 50]
    weights = [(0.5, 0.5), (0.7, 0.3), (0.4, 0.6)]

    best_score = float("inf")
    best_config = None
    history = []
    strategy_id = 0

    progress_bar = st.progress(0.0, text="Starting optimization...")

    search_space = list(itertools.product(
        chunking_strategies, chunk_sizes, overlaps, faiss_ks, bm25_ks, fetch_ks, weights
    ))
    total_runs = len(search_space)
    completed_runs = 0

    for chunking, size, overlap, faiss_k, bm25_k, fetch_k, weight in search_space:
        strategy_id += 1
        completed_runs += 1

        append_log(f"\n⚙️ Strategy {strategy_id}/{total_runs}")
        append_log(f"    🔹 Chunking: {chunking}, Size: {size}, Overlap: {overlap}")
        append_log(f"    🔹 FAISS_k: {faiss_k}, BM25_k: {bm25_k}, Fetch_k: {fetch_k}, Weights: {weight}")

        os.environ["CHUNK_STRATEGY"] = chunking
        os.environ["CHUNK_SIZE"] = str(size)
        os.environ["CHUNK_OVERLAP"] = str(overlap)
        os.environ["QDRANT_COLLECTION"] = f"qdrant_{chunking}_c{size}_o{overlap}"
        os.environ["FORCE_REBUILD"] = "1"

        try:
            analyzer = CMSDenialAnalyzer(
                exclude_tokens=[],
                faiss_k=faiss_k,
                bm25_k=bm25_k,
                faiss_fetch_k=fetch_k,
                weights=weight
            )

            metrics = analyzer.run_entropy_analysis(claims_data, target_profile)
            score, penalties = analyzer.compare_with_target(metrics, target_profile)
            final_score = round(1.0 - score, 4)

            config = {
                "strategy_id": strategy_id,
                "chunking": chunking,
                "chunk_size": size,
                "chunk_overlap": overlap,
                "faiss_k": faiss_k,
                "bm25_k": bm25_k,
                "fetch_k": fetch_k,
                "weights": weight,
                "score": final_score,
                "metrics": metrics,
                "penalties": penalties,
                "timestamp": datetime.now().isoformat()
            }

            history.append(config)

            if score < best_score:
                best_score = score
                best_config = config

            append_log(f"✅ Score: {final_score}")
            append_log(f"    📊 Metrics: {json.dumps(metrics, indent=2)}")
            append_log(f"    📉 Penalties: {json.dumps(penalties, indent=2)}")

        except Exception as e:
            append_log(f"❌ Failed strategy #{strategy_id}: {e}")
            continue

        progress_bar.progress(completed_runs / total_runs, text=f"Completed {completed_runs}/{total_runs} strategies...")

    with open(LOG_PATH, "a") as f:
        for entry in history:
            f.write(json.dumps(entry) + "\n")

    append_log("🎯 Optimization complete!")
    return best_config, history

def apply_best_config_to_session(best_config):
    config = {
        "chunking_strategy": best_config["chunking"],
        "chunk_size": best_config["chunk_size"],
        "chunk_overlap": best_config["chunk_overlap"],
        "faiss_k": best_config["faiss_k"],
        "bm25_k": best_config["bm25_k"],
        "faiss_fetch_k": best_config["fetch_k"],
        "weights": tuple(best_config["weights"]),
        "exclude_tokens": []
    }

    st.session_state["frozen_params"] = config
    st.session_state["rebuild_index"] = True
    st.session_state["retriever_initialized"] = False

    ctx = get_script_run_ctx()
    if ctx and "log_lines" in st.session_state and "log_box" in st.session_state:
        append_log("\n🔧 Applying Best Strategy to Manual Mode:")
        for key, val in config.items():
            append_log(f"    {key}: {val}")
