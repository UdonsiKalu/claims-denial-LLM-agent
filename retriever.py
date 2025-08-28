# retriever.py

import os
import faiss
from langchain_community.vectorstores.faiss import FAISS
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from constants import CMS_MANUAL_PATH, PRIORITY_CHAPTERS, FAISS_INDEX_PATH


def index_needs_update(chunks):
    """Check if FAISS index needs rebuilding based on document changes or force flag."""
    if os.environ.get("FORCE_REBUILD") == "1":
        print("⚠️ FORCE_REBUILD is enabled — reindexing...")
        return True

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        return True

    index_time = os.path.getmtime(os.path.join(FAISS_INDEX_PATH, "index.faiss"))

    for chap, title in PRIORITY_CHAPTERS.items():
        filename = f"{chap} - {title}.pdf"
        path = os.path.join(CMS_MANUAL_PATH, filename)
        if os.path.exists(path) and os.path.getmtime(path) > index_time:
            return True

    try:
        existing_index = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings=NomicEmbeddings(model="nomic-embed-text-v1"),
            allow_dangerous_deserialization=True
        )
        if abs(existing_index.index.ntotal - len(chunks)) > 5:
            return True
    except Exception as e:
        print("Index check failed:", e)
        return True

    return False


def create_or_load_vector_store(chunks):
    """Create or load a FAISS vector store using Nomic embeddings."""
    embedder = NomicEmbeddings(
        model="nomic-embed-text-v1",
        inference_mode="local",
        device="cuda" if faiss.get_num_gpus() > 0 else "cpu"
    )

    if not index_needs_update(chunks):
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}")
        try:
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings=embedder,
                allow_dangerous_deserialization=True
            )
            if faiss.get_num_gpus() > 0 and not isinstance(vectorstore.index, faiss.GpuIndex):
                print("Moving FAISS index to GPU...")
                vectorstore.index = faiss.index_cpu_to_all_gpus(vectorstore.index)
            print("FAISS index loaded successfully.")
            return {"vectorstore": vectorstore, "retriever": None}
        except Exception as e:
            print(f"Failed to load existing index: {e}. Rebuilding...")

    print("Creating new FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embedder)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("FAISS index created and saved.")
    return {"vectorstore": vectorstore, "retriever": None}


def create_retrievers(vectorstore, chunks):
    """Create an EnsembleRetriever combining BM25 and FAISS (MMR) search."""
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 2

    faiss_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 25}
    )

    ensemble = EnsembleRetriever(
        retrievers=[bm25, faiss_retriever],
        weights=[0.4, 0.6]
    )

    return ensemble
