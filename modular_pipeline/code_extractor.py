import re
from typing import List, Dict
from langchain.schema import Document

# Patterns for common healthcare code types
CPT_PATTERN = r'\b\d{5}\b'
ICD10_PATTERN = r'\b[A-TV-Z][0-9][A-Z0-9]{2,6}\b'
MODIFIER_PATTERN = r'\b(?:LT|RT|25|59|91|TC|26)\b'  # Extend as needed

def extract_codes(text: str) -> Dict[str, List[str]]:
    """Extract CPT, ICD-10, and Modifier codes from a string."""
    return {
        "cpt": re.findall(CPT_PATTERN, text),
        "icd10": re.findall(ICD10_PATTERN, text),
        "modifiers": re.findall(MODIFIER_PATTERN, text)
    }

def extract_codes_from_chunks(chunks: List[Document]) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract codes from each chunk and return a mapping:
    {
        chunk_id: {
            "cpt": [...],
            "icd10": [...],
            "modifiers": [...]
        }
    }
    """
    results = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "0")
        chunk_id = f"{source}::Page_{page}"
        codes = extract_codes(chunk.page_content)
        results[chunk_id] = codes
    return results
