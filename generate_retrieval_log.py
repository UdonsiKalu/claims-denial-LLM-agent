import json
from cms_analyzer import CMSDenialAnalyzer  # <-- adjust to your filename/module
from tqdm import tqdm

def run_probes(analyzer, claims_file="synthetic_claims.jsonl", output_log="retrieval_log.jsonl"):
    with open(claims_file, "r") as f:
        claims = [json.loads(line) for line in f.readlines()]

    log = []

    for claim in tqdm(claims, desc="Running retrieval probes"):
        query = analyzer._generate_query(claim)
        results = analyzer.retrieval["retriever"].invoke(query)

        log.append({
            "query": query,
            "cpt_code": claim["cpt_code"],
            "diagnosis": claim["diagnosis"],
            "retrieved_chunks": [
                {
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "source": doc.metadata.get("source", "unknown"),
                    "tokens": len(doc.page_content.split())
                }
                for doc in results
            ]
        })

    with open(output_log, "w") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Retrieval log written to {output_log}")
