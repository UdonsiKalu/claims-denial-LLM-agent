import json
import random
from tqdm import tqdm

# Output location
OUTPUT_FILE = "simple_synthetic_claims.jsonl"

# Sample codes
CPT_CODES = ["99213", "99214", "99203", "20610", "36415", "81001", "11721", "20552", "99395", "G0439"]
ICD_CODES = ["Z79.899", "E11.9", "I10", "M54.5", "R51", "J45.909", "N39.0", "F41.1", "K21.9", "M25.561"]
MODIFIERS = ["25", "59", "76", "91", "LT", "RT"]

# Number of claims
NUM_CLAIMS = 1000

# Generate and write claims with progress bar
with open(OUTPUT_FILE, "w") as f:
    for _ in tqdm(range(NUM_CLAIMS), desc="Generating claims"):
        claim = {
            "cpt_code": random.choice(CPT_CODES),
            "diagnosis": random.choice(ICD_CODES),
            "modifiers": random.sample(MODIFIERS, k=random.randint(0, 2)),
            "payer": "Medicare"
        }
        f.write(json.dumps(claim) + "\n")

print(f"\n✅ Saved {NUM_CLAIMS} claims to {OUTPUT_FILE}")
