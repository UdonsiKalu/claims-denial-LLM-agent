## Policy-Guided Claim Denial Agent

This repository contains a LangChain-powered agent designed to evaluate Medicare and payer-specific claim denial scenarios using retrieval-augmented generation (RAG). The agent combines FAISS-GPU document retrieval, structured parsing, tool-based execution, and Ollama LLM reasoning to provide clear, policy-grounded denial analysis.

## Features

- Modular tool-calling agent with `AnalyzeClaim`, `RetrieveCMSPolicy`, and `UpdatePolicies`
- Regex-based parsing of CPT codes, diagnoses, modifiers, and payer instructions
- Guardrails-enforced reasoning with structured JSON outputs
- FAISS-GPU semantic retrieval using CMS policy documents (Chapters 1, 12, 23, 30)
- EnsembleRetriever blending BM25 and FAISS with configurable weights
- CUDA-accelerated LLM inference using Ollama and Llama3
- Streamlit-compatible frontend (optional), embeddable via reverse proxy

## Architecture Overview

User Input → Agent → Tool Selection → Claim Parser / Retriever ↓ GPU Claim Analyzer (RAG) ↓ Policy-Grounded Response

## File Overview

| File                  | Description                                      |
|-----------------------|--------------------------------------------------|
| tool.py               | LangChain Tool wrappers for claim analysis       |
| faiss_gpu.py          | GPU-optimized analyzer and retriever             |
| agent.py              | MedicareClaimAgent for agent execution           |
| manuals/              | CMS PDFs parsed into vector chunks               |
| app.py                | Streamlit frontend (optional)                    |

## Installation

git clone https://github.com/UdonsiKalu/policy-denial-LLM.git
cd policy-denial-LLM
pip install -r requirements.txt
