# compare_both_strategies.py
# Forces FULL rebuild for BOTH baseline and final (separate Qdrant collections),
# runs the same claims, saves metrics, TSVs, individual entropy PNGs,
# query–chunk heatmaps, and a side-by-side comparison PNG.

import os, json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from faiss_gpu_entropy import CMSDenialAnalyzer  # your analyzer

# ---------- utils ----------
def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)

def load_jsonl(p: str) -> List[Dict[str, Any]]:
    with open(p, "r") as f:
        return [json.loads(l) for l in f]

def save_entropy_png(retrieval_log: dict, out_png: Path, title: str, top_n: int = 40):
    items = sorted(retrieval_log.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    labels, counts = zip(*items) if items else ([], [])
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(counts)), counts)
    plt.title(title)
    plt.xlabel("Top chunks")
    plt.ylabel("Retrieval count")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

def save_query_chunk_heatmap(query_chunk_matrix: np.ndarray, queries: List[str], chunks: List[str], out_png: Path, title: str):
    plt.figure(figsize=(12, 8))
    # reverse colorbar using _r
    sns.heatmap(
        query_chunk_matrix,
        cmap="viridis_r",  # reversed viridis
        cbar=True,
        xticklabels=False,
        yticklabels=False
    )
    plt.title(title)
    plt.xlabel("Chunks")
    plt.ylabel("Queries")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def side_by_side_png(left_png: Path, right_png: Path, out_png: Path):
    a = Image.open(left_png)
    b = Image.open(right_png)
    w = a.width + b.width
    h = max(a.height, b.height)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width, 0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)

# ---------- runner ----------
def run_strategy(tag: str, cfg: Dict[str, Any], claims_path: str, out_dir: Path):
    # Force FULL rebuild & unique collection for this strategy
    os.environ["CHUNK_STRATEGY"]    = "Fixed-size"
    os.environ["CHUNK_SIZE"]        = str(int(cfg["chunk_size"]))
    os.environ["CHUNK_OVERLAP"]     = str(int(cfg["chunk_overlap"]))
    os.environ["FORCE_REBUILD"]     = "1"
    os.environ["QDRANT_COLLECTION"] = f"cms_{tag}_c{cfg['chunk_size']}_o{cfg['chunk_overlap']}"

    analyzer = CMSDenialAnalyzer(
        exclude_tokens=[],
        faiss_k=int(cfg["faiss_k"]),
        bm25_k=int(cfg["bm25_k"]),
        faiss_fetch_k=int(cfg["fetch_k"]),
        weights=tuple(cfg["weights"]),
    )

    claims = load_jsonl(claims_path)

    # Metrics using analyzer’s built-ins
    target_profile = {
        "query_entropy_range": (0.7, 0.9),
        "max_chunk_frequency": 0.10,
        "gini_threshold": 0.40,
        "required_code_coverage": 0.95,
    }
    metrics = analyzer.run_entropy_analysis(claims, target_profile)

    # Retrieval log + query/chunk matrix
    retrieval_log = defaultdict(int)
    all_chunks = set()
    matrix_rows = []
    for claim in claims:
        try:
            docs = analyzer.retrieval["retriever"].invoke(claim["cpt_code"])
            chunk_ids = []
            for doc in docs:
                cid = doc.metadata.get("chunk_id") or f"{doc.metadata.get('source','unknown')}::Page_{doc.metadata.get('page',0)}"
                retrieval_log[cid] += 1
                chunk_ids.append(cid)
                all_chunks.add(cid)
            matrix_rows.append(chunk_ids)
        except Exception as e:
            print(f"[{tag}] retrieval failed for {claim.get('cpt_code','?')}: {e}")

    all_chunks = sorted(all_chunks)
    chunk_index = {cid: i for i, cid in enumerate(all_chunks)}
    query_chunk_matrix = np.zeros((len(claims), len(all_chunks)))
    for qi, chunk_ids in enumerate(matrix_rows):
        for cid in chunk_ids:
            query_chunk_matrix[qi, chunk_index[cid]] += 1

    # Save artifacts
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(run_dir / "metrics.json", "w"), indent=2)
    (run_dir / "retrieval_counts.tsv").write_text(
        "\n".join(f"{k}\t{v}" for k, v in sorted(retrieval_log.items(), key=lambda kv: kv[1], reverse=True))
    )

    # PNGs
    save_entropy_png(retrieval_log, run_dir / "entropy.png", f"{tag.title()} — Top Chunk Retrieval")
    save_query_chunk_heatmap(query_chunk_matrix, list(range(len(claims))), all_chunks, run_dir / "query_chunk_entropy.png", f"{tag.title()} — Query–Chunk Heatmap")

    return metrics

def main(baseline_json: str, final_json: str, claims_jsonl: str, out_root: str = ".retrieval_compare"):
    out = Path(out_root); out.mkdir(exist_ok=True)

    base = load_json(baseline_json)
    fin  = load_json(final_json)

    mb = run_strategy("baseline", base, claims_jsonl, out)
    mf = run_strategy("final",    fin,  claims_jsonl, out)

    # delta summary
    keys = set(mb) | set(mf)
    delta = {k: round(mf.get(k, 0) - mb.get(k, 0), 6) for k in keys}
    json.dump({"baseline": mb, "final": mf, "delta": delta}, open(out/"summary.json", "w"), indent=2)

    # side-by-side PNG (entropy maps only)
    side_by_side_png(out/"baseline/entropy.png", out/"final/entropy.png", out/"comparison_entropy.png")
    side_by_side_png(out/"baseline/query_chunk_entropy.png", out/"final/query_chunk_entropy.png", out/"comparison_query_chunk_entropy.png")

    print("✅ Wrote:", out/"summary.json")
    print("   -", out/"baseline/metrics.json")
    print("   -", out/"final/metrics.json")
    print("   -", out/"baseline/retrieval_counts.tsv")
    print("   -", out/"final/retrieval_counts.tsv")
    print("   -", out/"baseline/entropy.png")
    print("   -", out/"final/entropy.png")
    print("   -", out/"comparison_entropy.png")
    print("   -", out/"baseline/query_chunk_entropy.png")
    print("   -", out/"final/query_chunk_entropy.png")
    print("   -", out/"comparison_query_chunk_entropy.png")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python compare_both_strategies.py <baseline.json> <final.json> <claims.jsonl> [<out_root>]")
        sys.exit(1)
    baseline_json, final_json, claims_jsonl = sys.argv[1:4]
    out_root = sys.argv[4] if len(sys.argv) > 4 else ".retrieval_compare"
    main(baseline_json, final_json, claims_jsonl, out_root)
