Policy-Guided Claim Denial Agent

This project delivers a claim denial assistant built using Streamlit and Retrieval-Augmented Generation (RAG). 
It interprets denial scenarios based on real-world payer policy content and provides guided insights for insurance analysts, benefits auditors, and healthcare teams.

Features
    Natural language interface for submitting denial scenarios
    RAG-enabled grounding from benefit and coverage policy documents
    Agent reasoning chain with transparent step-by-step justification
    Interactive UI for rapid claim triage and eligibility review
    Support for multiple policy sources and CMS-based adjudication cues

Architecture Overview
    Core framework: Streamlit
    Retrieval: FAISS semantic search or vector DB
    Language model: OpenAI/GPT or local LLM
    Backend: Python with modular agent wrappers
