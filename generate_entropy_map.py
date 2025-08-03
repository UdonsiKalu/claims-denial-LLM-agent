import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from faiss_gpu import CMSDenialAnalyzer  # Correct module name

CLAIMS_FILE = "simple_synthetic_claims.jsonl"
OUTPUT_CSV = "real_entropy_map.csv"

# Step 1: Load analyzer
analyzer = CMSDenialAnalyzer()
retriever = analyzer.retrieval["retriever"]

# Step 2: Load synthetic claims
with open(CLAIMS_FILE, "r") as f:
    claims = [json.loads(line) for line in f]

# Step 3: Track retrieval frequencies
retrieval_log = defaultdict(int)

for claim in tqdm(claims, desc="Processing claims"):
    query = analyzer._generate_query(claim)
    docs = retriever.invoke(claim["cpt_code"])  # You can change to query if needed

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "0")
        chunk_id = f"{source}::Page_{page}"
        retrieval_log[chunk_id] += 1

# Step 4: Save entropy map
df = pd.DataFrame([
    {"chunk_id": k, "retrieval_count": v} for k, v in retrieval_log.items()
])
df.sort_values(by="retrieval_count", ascending=False, inplace=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Entropy map saved to {OUTPUT_CSV} with {len(df)} unique chunks.")
