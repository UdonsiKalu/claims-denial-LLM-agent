# compare_baseline_vs_final.py
# Purpose:
#   - Force a FULL rebuild for the FINAL strategy (re-chunk, re-embed, re-index, set params).
#   - Reuse the BASELINE artifacts (no rebuild), just run retrieval to get comparable outputs.
#   - Produce before/after entropy images + metrics delta and a side-by-side comparison.

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List
from PIL import Image

# Your analyzer
from faiss_gpu_entropy import CMSDenialAnalyzer

# ----------------------- helpers -----------------------

def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)

def _dump_json(obj: Dict[str, Any], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    with open(p, "r") as f:
        return [json.loads(line) for line in f]

def _ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

# ------------------ light adapters ---------------------

class AnalyzerAdapter:
    """Minimal adapter to tolerate small method-name diffs."""
    def __init__(self, analyzer: CMSDenialAnalyzer):
        self.A = analyzer

    # chunk
    def build_chunks(self):
        for name in ("chunk_manuals", "build_chunks", "make_chunks", "chunk"):
            fn = getattr(self.A, name, None)
            if callable(fn): return fn()
        raise AttributeError("Need a chunking method, e.g., chunk_manuals().")

    def save_chunks(self, chunks_path: Path, chunks=None):
        fn = getattr(self.A, "save_chunks", None)
        if callable(fn):
            # prefer analyzer’s own save
            return fn(chunks, str(chunks_path)) if chunks is not None else fn(str(chunks_path))
        if chunks is None:
            raise RuntimeError("Analyzer lacks save_chunks() and no chunks returned to write.")
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_path, "w") as f:
            for row in chunks:
                f.write(json.dumps(row) + "\n")

    # embed
    def build_embeddings(self, chunks_path: Path, out_npy: Path):
        for name in ("embed_chunks", "build_embeddings"):
            fn = getattr(self.A, name, None)
            if callable(fn): return fn(str(chunks_path), str(out_npy))
        raise AttributeError("Need embed_chunks(chunks_path, out_npy) in analyzer.")

    # index
    def build_index(self, embeddings_path: Path, vector_cfg: Dict[str, Any], out_dir: Path):
        for name in ("build_index", "create_index"):
            fn = getattr(self.A, name, None)
            if callable(fn): return fn(str(embeddings_path), vector_cfg, str(out_dir))
        raise AttributeError("Need build_index(embeddings_path, vector_cfg, out_dir) in analyzer.")

    def update_index_params(self, index_dir: Path, params: Dict[str, Any]) -> bool:
        fn = getattr(self.A, "update_index_params", None)
        if callable(fn):
            fn(str(index_dir), params)
            return True
        return False

    # retriever
    def get_retriever(self, index_dir: Path, retriever_cfg: Dict[str, Any]):
        for name in ("get_retriever", "create_retriever", "build_retriever"):
            fn = getattr(self.A, name, None)
            if callable(fn): return fn(str(index_dir), retriever_cfg)
        raise AttributeError("Need get_retriever(index_dir, retriever_cfg) in analyzer.")


class RunnerAdapter:
    def __init__(self, analyzer: CMSDenialAnalyzer):
        self.A = analyzer

    def run_retrieval_set(self, retriever, claims: List[Dict[str, Any]]):
        fn = getattr(self.A, "run_retrieval_set", None)
        if callable(fn):
            out = fn(retriever, claims)
            if isinstance(out, tuple) and len(out) == 2:
                return out
            return out, {}
        fn = getattr(self.A, "retrieve_claims", None)
        if callable(fn):
            log = fn(retriever, claims)
            return log, {}
        raise AttributeError("Need run_retrieval_set(retriever, claims) or retrieve_claims().")

    def plot_entropy(self, retrieval_log: List[Dict[str, Any]], save_to: Path) -> Dict[str, Any]:
        fn = getattr(self.A, "plot_entropy_map", None)
        if callable(fn):
            try:
                metrics = fn(retrieval_log, save_to=str(save_to))
            except TypeError:
                metrics = fn(retrieval_log, str(save_to))
            return metrics if isinstance(metrics, dict) else {}
        fn = getattr(self.A, "make_entropy_plot", None)
        if callable(fn):
            fn(retrieval_log, str(save_to))
            return {}
        raise AttributeError("Need plot_entropy_map(log, save_to=...) in analyzer.")

# -------------------- core operations ------------------

def _paths(out_root: Path, sid: str):
    return {
        "chunks": out_root / "chunks" / sid / "chunks.jsonl",
        "embs":   out_root / "embeddings" / sid / "embeddings.npy",
        "index":  out_root / "index" / sid,
        "runs":   out_root / "runs",
        "plots":  out_root / "plots",
        "reports":out_root / "reports",
    }

def force_rebuild_strategy(strategy: Dict[str, Any], data_path: str, out_root: Path):
    """Always re-chunk, re-embed, re-index for THIS strategy; then return retriever."""
    sid = strategy["strategy_id"]
    P = _paths(out_root, sid)

    # Clean slate for this strategy
    if P["index"].exists(): shutil.rmtree(P["index"])
    if P["chunks"].parent.exists(): shutil.rmtree(P["chunks"].parent)
    if P["embs"].parent.exists(): shutil.rmtree(P["embs"].parent)
    _ensure_dirs(P["chunks"].parent, P["embs"].parent, P["index"])

    analyzer = CMSDenialAnalyzer(
        manual_path=data_path,
        chunking=strategy.get("chunking", {}),
        embedding_cfg=strategy.get("embedding", {}),
    )
    A = AnalyzerAdapter(analyzer)

    # 1) chunk
    chunks = A.build_chunks()
    A.save_chunks(P["chunks"], chunks)

    # 2) embed
    A.build_embeddings(P["chunks"], P["embs"])

    # 3) index
    vector_cfg = strategy.get("vectorstore", {})
    A.build_index(P["embs"], vector_cfg, P["index"])

    # 4) retriever (apply params)
    retriever = A.get_retriever(P["index"], strategy.get("retriever", {}))
    return retriever

def reuse_existing_strategy(strategy: Dict[str, Any], data_path: str, out_root: Path):
    """DO NOT rebuild; assume artifacts exist; just create a retriever with current params."""
    sid = strategy["strategy_id"]
    P = _paths(out_root, sid)

    analyzer = CMSDenialAnalyzer(
        manual_path=data_path,
        chunking=strategy.get("chunking", {}),
        embedding_cfg=strategy.get("embedding", {}),
    )
    A = AnalyzerAdapter(analyzer)

    if not (P["chunks"].exists() and P["embs"].exists() and P["index"].exists()):
        raise FileNotFoundError(
            f"Baseline artifacts missing for {sid}. Expected:\n{P['chunks']}\n{P['embs']}\n{P['index']}"
        )

    # Optional: hot swap search params if supported
    vector_cfg = strategy.get("vectorstore", {})
    if vector_cfg.get("params"):
        A.update_index_params(P["index"], vector_cfg["params"])

    retriever = A.get_retriever(P["index"], strategy.get("retriever", {}))
    return retriever

def run_and_plot(strategy: Dict[str, Any], retriever, claims_file: Path, data_path: str, out_root: Path, tag: str):
    sid = strategy["strategy_id"]
    P = _paths(out_root, sid)

    # dirs
    run_dir = P["runs"] / f"{sid}_{tag}"
    _ensure_dirs(run_dir, P["plots"], P["reports"])

    # analyzer for running/plotting
    analyzer = CMSDenialAnalyzer(
        manual_path=data_path,
        chunking=strategy.get("chunking", {}),
        embedding_cfg=strategy.get("embedding", {}),
    )
    R = RunnerAdapter(analyzer)

    claims = _load_jsonl(claims_file)

    retrieval_log, _maybe = R.run_retrieval_set(retriever, claims)
    log_path = run_dir / "retrieval_log.jsonl"
    with open(log_path, "w") as f:
        for row in retrieval_log:
            f.write(json.dumps(row) + "\n")

    entropy_png = run_dir / "entropy.png"
    metrics = R.plot_entropy(retrieval_log, entropy_png)
    _dump_json(metrics, run_dir / "metrics.json")

    return {
        "sid": sid,
        "tag": tag,
        "entropy_png": str(entropy_png),
        "metrics": metrics
    }

def compare_images_and_metrics(baseline: Dict[str, Any], final: Dict[str, Any], out_root: Path):
    # side-by-side image
    img_b = Image.open(baseline["entropy_png"])
    img_f = Image.open(final["entropy_png"])
    w = img_b.width + img_f.width
    h = max(img_b.height, img_f.height)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(img_b, (0, 0))
    canvas.paste(img_f, (img_b.width, 0))

    comp_path = out_root / "plots" / "comparison_baseline_final.png"
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(comp_path)

    # metrics delta
    mb, mf = baseline["metrics"] or {}, final["metrics"] or {}
    keys = set(mb) | set(mf)
    delta = {k: (mf.get(k, 0) - mb.get(k, 0)) for k in keys}

    summary = {
        "baseline": {"strategy_id": baseline["sid"], "tag": baseline["tag"], "metrics": mb, "entropy_png": baseline["entropy_png"]},
        "final":    {"strategy_id": final["sid"],    "tag": final["tag"],    "metrics": mf, "entropy_png": final["entropy_png"]},
        "delta": delta,
        "comparison_png": str(comp_path)
    }
    _dump_json(summary, out_root / "reports" / "summary_baseline_final.json")
    return str(comp_path)

# ------------------------ main -------------------------

def main(
    baseline_json: str,
    final_json: str,
    claims_jsonl: str,
    CMS_MANUAL_PATH: str,
    out_root: str = ".retrieval_studio",
):
    out = Path(out_root)
    base = _load_json(Path(baseline_json))
    fin  = _load_json(Path(final_json))

    # 1) Reuse baseline (NO rebuild)
    base_ret = reuse_existing_strategy(base, CMS_MANUAL_PATH, out)
    base_res = run_and_plot(base, base_ret, Path(claims_jsonl), CMS_MANUAL_PATH, out, tag="baseline")

    # 2) Force FULL rebuild for FINAL
    fin_ret = force_rebuild_strategy(fin, CMS_MANUAL_PATH, out)
    fin_res = run_and_plot(fin, fin_ret, Path(claims_jsonl), CMS_MANUAL_PATH, out, tag="final")

    # 3) Side-by-side + summary
    comp = compare_images_and_metrics(base_res, fin_res, out)
    print("[compare] Wrote:", comp)
    print("[compare] Wrote:", out / "reports" / "summary_baseline_final.json")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("Usage: python compare_baseline_vs_final.py <baseline.json> <final.json> <claims.jsonl> <CMS_MANUAL_PATH> [<out_root>]")
        sys.exit(1)
    baseline_json, final_json, claims_jsonl, cms_path = sys.argv[1:5]
    out_root = sys.argv[5] if len(sys.argv) > 5 else ".retrieval_studio"
    main(baseline_json, final_json, claims_jsonl, cms_path, out_root)
