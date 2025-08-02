import os
import json
from retriever import create_or_load_vector_store, create_retrievers
from chunking import load_and_chunk_manuals
from llm_chain import setup_llm_chain  # You need to implement this if not done already

class CMSDenialAnalyzer:
    def __init__(self, exclude_tokens=None):
        print("🔍 Initializing CMS Denial Analyzer...")

        # Token filters for ignoring generic or low-signal content
        self.exclude_tokens = [t.lower() for t in exclude_tokens] if exclude_tokens else []

        # Read chunk settings from environment or fallback defaults
        chunk_size = int(os.environ.get("CHUNK_SIZE", 10000))
        chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", 2000))

        print(f"1️⃣  Loading and chunking CMS manuals (size={chunk_size}, overlap={chunk_overlap})...")
        self.chunks = load_and_chunk_manuals(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"✅ Loaded {len(self.chunks)} chunks")

        print("2️⃣  Building vectorstore and retriever...")
        self.retrieval = create_or_load_vector_store(self.chunks)

        if not self.retrieval["retriever"]:
            self.retrieval["retriever"] = create_retrievers(self.retrieval["vectorstore"])

        print("3️⃣  Setting up LLM chain...")
        self.chain = setup_llm_chain(self.retrieval["retriever"])
        print("🎯 Analyzer ready for claim evaluation.\n" + "=" * 50)

    def _generate_query(self, claim):
        """
        Convert a claim into a natural language query prompt.
        """
        if claim["cpt_code"].startswith("99"):
            return f"Evaluation and Management coding rules for CPT {claim['cpt_code']}"
        elif any(m in claim.get("modifiers", []) for m in ["-59", "-25"]):
            return f"Modifier {claim['modifiers'][0]} documentation requirements"
        else:
            return f"Medicare coverage criteria for {claim['cpt_code']} with {claim['diagnosis']}"

    def _filter_docs(self, docs):
        """
        Remove documents matching exclusion filters.
        """
        if not self.exclude_tokens:
            return docs
        return [
            doc for doc in docs
            if not any(
                token in doc.metadata.get("chunk_id", "").lower()
                or token in doc.page_content.lower()
                for token in self.exclude_tokens
            )
        ]

    def analyze_claim(self, claim_data):
        """
        Retrieve context, filter, and run LLM analysis for a single claim.
        """
        query = self._generate_query(claim_data)
        print(f"\n📄 Analyzing claim for CPT {claim_data['cpt_code']}...")

        retrieved_docs = self.retrieval["retriever"].invoke(claim_data["cpt_code"])
        filtered_docs = self._filter_docs(retrieved_docs)

        # Domain-aware filter: only keep docs with relevant billing terms
        filtered_docs = [
            doc for doc in filtered_docs
            if any(keyword in doc.page_content.lower() for keyword in ["modifier", "billing", "e/m"])
        ]

        if not filtered_docs:
            raise ValueError("❌ No relevant documents retrieved after filtering.")

        # Print top documents for transparency
        print("\n=== Top Retrieved Context ===")
        for i, doc in enumerate(filtered_docs[:3]):
            print(f"\n--- Document {i + 1} ---")
            print(doc.page_content[:300], "...\n")

        # Run LLM chain with structured prompt
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
            print("🔢 Raw LLM response:", raw_answer)
            raise RuntimeError("❌ Failed to parse LLM output as JSON")

        return parsed_answer
