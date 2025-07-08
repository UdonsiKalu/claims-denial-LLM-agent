from langchain.agents import Tool
from faiss_gpu import CMSDenialAnalyzer  # Changed from CMSDenialAnalyzer to match your class name
import re
import ast

# Initialize analyzer
analyzer = CMSDenialAnalyzer()

def retrieve_cms_rules(query: str):
    """Retrieve relevant CMS policies using FAISS/BM25 retriever."""
    return analyzer.retrieval["retriever"].invoke(query)

def analyze_claim_from_text(claim_str: str):
    """Parse a string into claim fields and analyze the claim."""
    fields = {
        "cpt_code": "99214",  # Default values
        "diagnosis": "Z79.899",
        "modifiers": [],
        "payer": "Medicare"
    }

    try:
        # Extract CPT (matches "CPT 99214" or "cpt:99214")
        cpt_match = re.search(r"(?i)cpt[\s:]+(\d{4,5})", claim_str)
        if cpt_match:
            fields["cpt_code"] = cpt_match.group(1)

        # Extract Diagnosis (matches "DX E11.65" or "diagnosis: Z79.9")
        diag_match = re.search(r"(?i)(diagnosis|dx)[\s:]+([A-Z]\d{2}(?:\.\d{1,2})?)", claim_str)
        if diag_match:
            fields["diagnosis"] = diag_match.group(2)

        # Extract Modifiers (matches "modifiers: [-25, -59]" or "modifiers = ['25']")
        mod_match = re.search(r"(?i)modifiers\s*[:=]\s*(\[[^]]*\])", claim_str)
        if mod_match:
            fields["modifiers"] = ast.literal_eval(mod_match.group(1))

        # Extract Payer (matches "payer: Medicare" or "payer = BCBS")
        payer_match = re.search(r"(?i)payer\s*[:=]\s*(\w+)", claim_str)
        if payer_match:
            fields["payer"] = payer_match.group(1).capitalize()

    except Exception as e:
        return {"error": f"Failed to parse input: {e}", "raw": claim_str}

    return analyzer.analyze_claim(fields)

# Register tools for the agent
cms_tools = [
    Tool(
        name="RetrieveCMSPolicy",
        func=retrieve_cms_rules,
        description="Fetch CMS billing policies for CPT codes, diagnoses, or modifiers."
    ),
    Tool(
        name="AnalyzeClaim",
        func=analyze_claim_from_text,
        description="""Analyze a Medicare claim from text. Expected format:
        "CPT [code] with diagnosis [code] [modifiers: [list]] [payer: name]"
        Example: "Analyze CPT 99214 with diagnosis E11.65 modifiers: [-25] payer: Medicare"
        """
    )
]