from glob import glob
import json

FHIR_DIR = "/media/udonsi-kalu/New Volume/denials/denials/synthea/output/fhir"
OUTPUT_FILE = "extracted_snomed_codes.json"

snomed_codes = set()

for file in glob(f"{FHIR_DIR}/*.json"):
    try:
        with open(file, "r") as f:
            data = json.load(f)
            for entry in data.get("entry", []):
                res = entry.get("resource", {})
                if res.get("resourceType") == "Procedure":
                    codings = res.get("code", {}).get("coding", [])
                    for c in codings:
                        if "snomed" in c.get("system", "").lower() or "sct" in c.get("system", "").lower():
                            snomed_codes.add(c.get("code"))
    except:
        continue

# Convert to sorted list
snomed_list = sorted(list(snomed_codes))

print(f"✅ Found {len(snomed_list)} unique SNOMED codes.")
print(f"📁 Saving to: {OUTPUT_FILE}")

# Save to file
with open(OUTPUT_FILE, "w") as f:
    json.dump(snomed_list, f, indent=2)
