import os
import json
from collections import Counter, defaultdict
from retriever_qdrant import create_or_load_vector_store, create_retrievers
from chunking import load_and_chunk_manuals
from llm_chain import setup_llm_chain
import numpy as np
from datetime import datetime, date

# Optional pandas awareness (safe if pandas isn't installed)
try:
    import pandas as pd  # noqa: F401
except Exception:  # pragma: no cover
    pd = None

def _json_default(o):
    """Serialize numpy/pandas/time objects and other non-JSON-native types."""
    # numpy scalars
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)

    # numpy arrays
    if isinstance(o, np.ndarray):
        return o.tolist()

    # pandas timestamps/objects
    if pd is not None:
        if isinstance(o, getattr(pd, "Timestamp", ())):
            return o.isoformat()
        # Fallbacks for Series/DataFrame if ever logged
        if isinstance(o, getattr(pd, "Series", ())):
            return o.to_dict()
        if isinstance(o, getattr(pd, "DataFrame", ())):
            return o.to_dict(orient="list")

    # datetimes
    if isinstance(o, (datetime, date)):
        return o.isoformat()

    # collections
    if isinstance(o, (set, tuple)):
        return list(o)

    # last resort
    return str(o)


class CMSDenialAnalyzer:
    def __init__(
        self,
        exclude_tokens=None,
        faiss_k=3,
        bm25_k=3,
        faiss_fetch_k=20, 
        weights=(0.4, 0.6),
        # ✅ new chunking args
        chunking_strategy="Fixed-size",
        chunk_size=None,
        chunk_overlap=None,
        header_levels=None,
        semantic_threshold=None,
        embeddings=None, 
        chunks=None,
    ):
        print("🔍 Initializing CMS Denial Analyzer...")

        # === Store retriever params
        self.faiss_k = faiss_k
        self.bm25_k = bm25_k
        self.faiss_fetch_k = faiss_fetch_k
        self.weights = weights
        self.exclude_tokens = [t.lower() for t in exclude_tokens] if exclude_tokens else []

        # === Store chunking params
        self.chunking_strategy = chunking_strategy
        self.chunk_size = int(chunk_size or os.environ.get("CHUNK_SIZE", 10000))
        self.chunk_overlap = int(chunk_overlap or os.environ.get("CHUNK_OVERLAP", 2000))
        self.header_levels = header_levels or os.environ.get("HEADER_LEVELS")
        self.semantic_threshold = (
            float(semantic_threshold) if semantic_threshold is not None
            else float(os.environ.get("SEMANTIC_THRESHOLD", 0.5))
        )

        # === Store embeddings (for semantic chunking / retrieval)
        self.embeddings = embeddings

        # === Use precomputed chunks if provided, else load fresh
        if chunks is not None:
            self.chunks = chunks
            print(f"✅ Using provided chunks ({len(self.chunks)} loaded)")
        else:
            # === Debug print
            print(f"=1️⃣  Loading and chunking CMS manuals "
                  f"(strategy={self.chunking_strategy}, "
                  f"size={self.chunk_size}, overlap={self.chunk_overlap}, "
                  f"headers={self.header_levels}, semantic_th={self.semantic_threshold})...")

            # ✅ Pass new args into chunker
            self.chunks = load_and_chunk_manuals(
                chunking_strategy=self.chunking_strategy,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                header_levels=self.header_levels,
                semantic_threshold=self.semantic_threshold,
                embeddings=self.embeddings  # <- must exist!
            )

            print(f"✅ Loaded {len(self.chunks)} chunks")


        # === Build vectorstore + retriever
        print("2️⃣  Building Qdrant vectorstore and retriever...")
        self.retrieval = create_or_load_vector_store(self.chunks)

        if not self.retrieval["retriever"]:
            self.retrieval["retriever"] = create_retrievers(
                self.retrieval["vectorstore"],
                self.chunks,
                faiss_k=faiss_k,
                bm25_k=bm25_k,
                faiss_fetch_k=faiss_fetch_k, 
                weights=weights
            )

        print("3️⃣  Setting up LLM chain...")
        self.chain = setup_llm_chain(self.retrieval["retriever"])
        print("🎯 Analyzer ready for claim evaluation.\n" + "=" * 50)


    def compute_token_frequencies(self):
        print("🔢 Computing token frequencies per chunk...")
        for chunk in self.chunks:
            tokens = chunk.page_content.lower().split()
            token_counts = dict(Counter(tokens))
            chunk.metadata["token_frequencies"] = token_counts
        print("✅ Token frequencies stored in metadata.")

    def _generate_query(self, claim):
        if claim["cpt_code"].startswith("99"):
            return f"Evaluation and Management coding rules for CPT {claim['cpt_code']}"
        elif any(m in claim.get("modifiers", []) for m in ["-59", "-25", "25", "59"]):
            return f"Modifier {claim['modifiers'][0]} documentation requirements"
        else:
            return f"Medicare coverage criteria for {claim['cpt_code']} with {claim['diagnosis']}"

    def _filter_docs(self, docs):
        if not self.exclude_tokens:
            return docs
        return [
            doc for doc in docs
            if not any(
                token in (doc.metadata.get("chunk_id", "") or "").lower()
                or token in (doc.page_content or "").lower()
                for token in self.exclude_tokens
            )
        ]

    def analyze_claim(self, claim_data):
        query = self._generate_query(claim_data)
        print(f"\n📄 Analyzing claim for CPT {claim_data['cpt_code']}...")

        # Consider using the generated query for retrieval; keeping cpt_code if intentional
        retrieved_docs = self.retrieval["retriever"].invoke(claim_data["cpt_code"])
        filtered_docs = self._filter_docs(retrieved_docs)
        filtered_docs = [
            doc for doc in filtered_docs
            if any(keyword in (doc.page_content or "").lower() for keyword in ["modifier", "billing", "e/m"])
        ]

        if not filtered_docs:
            raise ValueError("❌ No relevant documents retrieved after filtering.")

        print("\n=== Top Retrieved Context ===")
        for i, doc in enumerate(filtered_docs[:3]):
            print(f"\n--- Document {i + 1} ---")
            print((doc.page_content or "")[:300], "...\n")

        response = self.chain.invoke({
            "input": query,
            "context": filtered_docs,
            "cpt_code": claim_data["cpt_code"],
            "diagnosis": claim_data.get("diagnosis", ""),
            "modifiers": claim_data.get("modifiers", []),
            "payer": claim_data.get("payer", "Medicare")
        })

        raw_answer = response.get("answer")
        if not raw_answer:
            raise RuntimeError("LLM response missing 'answer' field or empty")

        try:
            cleaned_response = raw_answer.strip()
            if not cleaned_response.startswith("{"):
                cleaned_response = "{" + cleaned_response.split("{", 1)[-1]
            if not cleaned_response.endswith("}"):
                cleaned_response = cleaned_response.rsplit("}", 1)[0] + "}"
            parsed_answer = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print("⚠️ Error parsing LLM output:", e)
            print("📉 Raw LLM response:", raw_answer)
            raise RuntimeError("❌ Failed to parse LLM output as JSON")

        return parsed_answer

    def run_entropy_analysis(self, claims, target_profile):
        retrieval_log = defaultdict(int)
        seen_codes = set()

        for claim in claims:
            try:
                query = claim.get("cpt_code", "")
                if not query:
                    continue
                docs = self.retrieval["retriever"].invoke(query)
                seen_codes.add(query)
                for doc in docs:
                    chunk_id = doc.metadata.get("chunk_id") or f"{doc.metadata.get('source', 'unknown')}::Page_{doc.metadata.get('page', 0)}"
                    retrieval_log[chunk_id] += 1
            except Exception as e:
                print(f"❌ Retrieval failed for query: {query} — {e}")

        retrieval_counts = np.array(list(retrieval_log.values()))
        total_chunks = int(len(retrieval_counts))
        total_queries = int(len(seen_codes))

        if total_chunks == 0:
            return {
                "query_entropy": 0.0,
                "max_chunk_frequency": 1.0,
                "gini_coefficient": 1.0,
                "code_coverage": 0.0,
            }

        probabilities = retrieval_counts / max(float(retrieval_counts.sum()), 1.0)
        # Cast to native Python floats to be JSON-safe everywhere
        query_entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-10)))
        max_chunk_frequency = float(retrieval_counts.max() / max(float(retrieval_counts.sum()), 1.0))
        gini_coefficient = float(self._gini(retrieval_counts))
        code_coverage = float(len(seen_codes) / max(total_queries, 1))

        return {
            "query_entropy": query_entropy,
            "max_chunk_frequency": max_chunk_frequency,
            "gini_coefficient": gini_coefficient,
            "code_coverage": code_coverage,
        }

    def _gini(self, array):
        array = np.sort(np.array(array, dtype=float))
        n = len(array)
        if n == 0:
            return 0.0
        index = np.arange(1, n + 1, dtype=float)
        return float((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array) + 1e-10))

    def compare_with_target(self, metrics, target_profile):
        penalties = {}

        def penalty(actual, target, weight=1.0):
            return float(weight) * max(0.0, float(actual) - float(target))

        penalties["entropy_penalty"] = penalty(
            target_profile["query_entropy_range"][0] - metrics["query_entropy"],
            0,
            weight=2.0
        ) + penalty(
            metrics["query_entropy"] - target_profile["query_entropy_range"][1],
            0,
            weight=2.0
        )

        penalties["frequency_penalty"] = penalty(
            metrics["max_chunk_frequency"],
            target_profile["max_chunk_frequency"],
            weight=1.5
        )

        penalties["gini_penalty"] = penalty(
            metrics["gini_coefficient"],
            target_profile["gini_threshold"],
            weight=1.0
        )

        penalties["coverage_penalty"] = penalty(
            target_profile["required_code_coverage"] - metrics["code_coverage"],
            0,
            weight=2.5
        )

        total_score = float(sum(penalties.values()))
        return total_score, penalties

    def evaluate_entropy_score(self, claims, target_profile, log_fn=None, idx=None, total=None, params=None, search_method=None):
        metrics = self.run_entropy_analysis(claims, target_profile)
        score, penalties = self.compare_with_target(metrics, target_profile)
        final_score = round(float(1.0 - score), 4)

        result = {
            "strategy_id": f"faiss{self.faiss_k}_bm25{self.bm25_k}_fetch{self.fetch_k}_w{self.weights}",
            "faiss_k": int(self.faiss_k),
            "bm25_k": int(self.bm25_k),
            "fetch_k": int(self.fetch_k),
            "weights": list(self.weights),  # JSON-friendly
            "score": float(final_score),
            "entropy": float(metrics["query_entropy"]),
            "coverage": float(metrics["code_coverage"]),
            "gini": float(metrics["gini_coefficient"]),
            "max_freq": float(metrics["max_chunk_frequency"]),
            "penalties": {k: float(v) for k, v in penalties.items()},
        }

        # ---- Pretty JSON log (now JSON-safe) ----
        if log_fn and params is not None:
            log_data = {
                "strategy_number": int(idx) if isinstance(idx, (int, np.integer)) else (idx if idx is not None else "?"),
                "total_strategies": int(total) if isinstance(total, (int, np.integer)) else (total if total is not None else "?"),
                "chunking": {
                    "strategy": params.get("chunking_strategy"),
                    "size": params.get("chunk_size"),
                    "overlap": params.get("chunk_overlap"),
                    "header_levels": params.get("header_levels"),
                    "semantic_threshold": params.get("semantic_threshold"),
                },
                "retriever": {
                    "faiss_k": params.get("faiss_k"),
                    "bm25_k": params.get("bm25_k"),
                    "fetch_k": params.get("fetch_k"),
                    "weights": params.get("weights"),
                    "rerank_top_k": params.get("rerank_top_k"),
                    "rerank_weight": params.get("rerank_weight"),
                },
                "optimization": {
                    "search_method": search_method or "N/A",
                    "params": params,  # may contain numpy scalars; default= handles them
                },
                "scores": {
                    "entropy_score": float(final_score),
                    "metrics": {
                        "query_entropy": float(metrics.get("query_entropy", 0.0)),
                        "max_chunk_frequency": float(metrics.get("max_chunk_frequency", 0.0)),
                        "gini_coefficient": float(metrics.get("gini_coefficient", 0.0)),
                        "code_coverage": float(metrics.get("code_coverage", 0.0)),
                    },
                    "penalties": {
                        "entropy_penalty": float(penalties.get("entropy_penalty", 0.0)),
                        "frequency_penalty": float(penalties.get("frequency_penalty", 0.0)),
                        "gini_penalty": float(penalties.get("gini_penalty", 0.0)),
                        "coverage_penalty": float(penalties.get("coverage_penalty", 0.0)),
                    },
                    "final_strategy_score": float(final_score),
                },
            }

            formatted_log = json.dumps(log_data, indent=4, default=_json_default)
            log_fn(formatted_log)
        # ---- end pretty JSON log ----

        return result
