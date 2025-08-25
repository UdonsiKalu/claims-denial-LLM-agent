# retriever_test.py

import os
from constants import CMS_MANUAL_PATH, FAISS_INDEX_PATH
from retriever import create_or_load_vector_store, create_retrievers
from ent_chunking import load_and_chunk_manuals

# === Set Environment Configs ===
os.environ["CHUNK_SIZE"] = "10000"
os.environ["CHUNK_OVERLAP"] = "2000"
os.environ["INDEX_PATH"] = FAISS_INDEX_PATH

# === Load & Chunk CMS Manuals ===
print("🔹 Loading and chunking manuals...")
chunks = load_and_chunk_manuals(chunk_size=10000, chunk_overlap=2000)
print(f"✅ Loaded {len(chunks)} chunks")

# === Create or Load FAISS Vector Store ===
print("🔹 Building vector store...")
vs = create_or_load_vector_store(chunks)

# === Create Hybrid Retriever ===
print("🔹 Creating retrievers...")

retriever = create_retrievers(
    vectorstore=vs["vectorstore"],
    chunks=chunks,
    faiss_k=5,
    bm25_k=3,
    faiss_fetch_k=50,
    weights=(0.3, 0.7)  # More weight to FAISS results
)

# === Test Queries ===
test_queries = [
    "CPT 99214",
    "modifier -25 documentation",
    "billing for chronic care management",
    "Evaluation and Management rules",
    "CPT 93000"
]

# === Execute Queries ===
for query in test_queries:
    print(f"\n🔍 Query: {query}")
    docs = retriever.invoke(query)
    if not docs:
        print("No documents retrieved.")
        continue
    for i, doc in enumerate(docs[:3]):
        print(f"\n--- Document {i+1} ---")
        print("📄 Metadata:", doc.metadata)
        print("📃 Preview:", doc.page_content[:300], "...\n")
