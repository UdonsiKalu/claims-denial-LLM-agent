import os, json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List
from faiss_gpu_entropy import CMSDenialAnalyzer  # your analyzer

def load_json(p:str)->Dict[str,Any]: return json.load(open(p))
def load_jsonl(p:str)->List[Dict[str,Any]]: return [json.loads(l) for l in open(p)]

def run_strategy(tag:str, cfg:Dict[str,Any], claims_path:str, out_dir:Path):
    # Force FULL rebuild & unique collection for this strategy
    os.environ["CHUNK_STRATEGY"]   = "Fixed-size"
    os.environ["CHUNK_SIZE"]       = str(int(cfg["chunk_size"]))
    os.environ["CHUNK_OVERLAP"]    = str(int(cfg["chunk_overlap"]))
    os.environ["FORCE_REBUILD"]    = "1"
    os.environ["QDRANT_COLLECTION"]= f"cms_{tag}_c{cfg['chunk_size']}_o{cfg['chunk_overlap']}"

    analyzer = CMSDenialAnalyzer(
        exclude_tokens=[],
        faiss_k=int(cfg["faiss_k"]),
        bm25_k=int(cfg["bm25_k"]),
        faiss_fetch_k=int(cfg["fetch_k"]),
        weights=tuple(cfg["weights"])
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

    # Retrieval distribution (for sanity/inspection)
    retrieval_log = defaultdict(int)
    for claim in claims:
        docs = analyzer.retrieval["retriever"].invoke(claim["cpt_code"])
        for doc in docs:
            cid = doc.metadata.get("chunk_id") or f"{doc.metadata.get('source','unknown')}::Page_{doc.metadata.get('page',0)}"
            retrieval_log[cid] += 1

    # Save artifacts
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(run_dir/"metrics.json","w"), indent=2)
    (run_dir/"retrieval_counts.tsv").write_text(
        "\n".join(f"{k}\t{v}" for k,v in sorted(retrieval_log.items(), key=lambda kv: kv[1], reverse=True))
    )
    return metrics

def main(baseline_json:str, final_json:str, claims_jsonl:str, out_root:str=".retrieval_compare"):
    out = Path(out_root); out.mkdir(exist_ok=True)
    base = load_json(baseline_json)
    fin  = load_json(final_json)

    mb = run_strategy("baseline", base, claims_jsonl, out)
    mf = run_strategy("final",    fin,  claims_jsonl, out)

    keys = set(mb)|set(mf)
    delta = {k: round(mf.get(k,0)-mb.get(k,0), 6) for k in keys}
    json.dump({"baseline":mb, "final":mf, "delta":delta}, open(out/"summary.json","w"), indent=2)
    print("✅ Wrote:", out/"summary.json")
    print("   -", out/"baseline/metrics.json")
    print("   -", out/"final/metrics.json")
    print("   -", out/"baseline/retrieval_counts.tsv")
    print("   -", out/"final/retrieval_counts.tsv")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python compare_both_strategies.py <baseline.json> <final.json> <claims.jsonl> [<out_root>]")
        sys.exit(1)
    baseline_json, final_json, claims_jsonl = sys.argv[1:4]
    out_root = sys.argv[4] if len(sys.argv) > 4 else ".retrieval_compare"
    main(baseline_json, final_json, claims_jsonl, out_root)
