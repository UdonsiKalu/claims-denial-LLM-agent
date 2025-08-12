import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_community.vectorstores import Qdrant
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from constants import CMS_MANUAL_PATH, PRIORITY_CHAPTERS

# Load Qdrant config from environment
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "cms_chunks")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))


def index_needs_update(chunks):
    if os.environ.get("FORCE_REBUILD") == "1":
        print("⚠️ FORCE_REBUILD is enabled — reindexing...")
        return True
    return False


def create_or_load_vector_store(chunks):
    embedder = NomicEmbeddings(model="nomic-embed-text-v1", inference_mode="local", device="cuda")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections or index_needs_update(chunks):
        print(f"🔄 Rebuilding Qdrant collection: {QDRANT_COLLECTION}")
        print(f"📄 Embedding {len(chunks)} chunks...")

        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

        Qdrant.from_documents(
            documents=chunks,
            embedding=embedder,
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION,
        )

        print("✅ Qdrant collection built.")
    else:
        print(f"📦 Qdrant collection already exists: {QDRANT_COLLECTION}")

    vectorstore = Qdrant(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embeddings=embedder,
    )

    return {"vectorstore": vectorstore, "retriever": None}



def create_retrievers(
    vectorstore,
    chunks,
    faiss_k=3,
    bm25_k=2,
    faiss_fetch_k=25,
    weights=(0.4, 0.6)
):
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = bm25_k

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": faiss_k, "fetch_k": faiss_fetch_k},
    )

    return EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=weights,
    )
