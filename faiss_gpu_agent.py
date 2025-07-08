import os
import sys
import faiss
import json
import asyncio
from typing import Dict, List, Union
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents import tool
from langchain_core.messages import AIMessage, HumanMessage

# =================================
# GPU CONFIGURATION
# =================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Lock to GPU 0
faiss.standard_gpu_mode()  # Global GPU mode

print(f"FAISS GPU Available: {faiss.get_num_gpus()} devices")
print(f"FAISS Version: {faiss.__version__}")

# =================================
# CONSTANTS
# =================================
CMS_MANUAL_PATH = "/media/udonsi-kalu/New Volume/denials/denials/manuals/"
FAISS_INDEX_PATH = "cms_manuals_faiss_index_agent"  # GPU-only index
PRIORITY_CHAPTERS = {
    "Chapter 1": "General Billing Requirements",
    "Chapter 12": "Physicians Nonphysician Practitioners",
    "Chapter 23": "Fee Schedule Administration",
    "Chapter 30": "Financial Liability Protections"
}

# =================================
# DOCUMENT PROCESSING (UNCHANGED)
# =================================
class DocumentProcessor:
    @staticmethod
    def load_and_chunk_manuals():
        if not os.path.exists(CMS_MANUAL_PATH):
            raise FileNotFoundError(f"CMS manuals directory not found at {CMS_MANUAL_PATH}")

        loaders = []
        for chap, title in PRIORITY_CHAPTERS.items():
            filename = f"{chap} - {title}.pdf"
            path = os.path.join(CMS_MANUAL_PATH, filename)
            if os.path.exists(path):
                loaders.append(PyPDFLoader(path))

        if not loaders:
            raise ValueError("No CMS manual PDFs found")

        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
            ("#", "Chapter"), ("##", "Section"), ("###", "Subsection")
        ])
        content_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=2000,
            separators=["\n\n", "\n", r"(?<=\. )", " "]
        )

        chunks = []
        for loader in loaders:
            try:
                docs = loader.load()
                for doc in docs:
                    headers = header_splitter.split_text(doc.page_content)
                    chunks.extend(content_splitter.split_documents(headers))
            except Exception as e:
                print(f"Error loading {loader.file_path}: {e}")

        return chunks

# =================================
# PURE GPU VECTOR STORE
# =================================
class VectorStoreManager:
    @staticmethod
    def create_or_load_vector_store(chunks):
        embedder = NomicEmbeddings(
            model="nomic-embed-text-v1",
            inference_mode="local",
            device="cuda"
        )

        if not os.path.exists(FAISS_INDEX_PATH):
            print("Creating pure GPU index...")
            # Direct GPU index construction
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(
                res,
                chunks[0].embedding.shape[1],
                faiss.METRIC_L2
            )
            vectorstore = FAISS.from_documents(
                chunks,
                embedder,
                index=index
            )
            # Special GPU-optimized save
            cpu_index = faiss.index_gpu_to_cpu(vectorstore.index)  # Temporary conversion for saving
            vectorstore.index = cpu_index
            vectorstore.save_local(FAISS_INDEX_PATH)
            # Restore GPU index
            vectorstore.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            return vectorstore

        print("Loading GPU index...")
        try:
            res = faiss.StandardGpuResources()
            # Load directly to GPU
            index = faiss.read_index(os.path.join(FAISS_INDEX_PATH, "index.faiss"))
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings=embedder,
                allow_dangerous_deserialization=True
            )
            vectorstore.index = gpu_index
            return vectorstore
        except Exception as e:
            print(f"GPU index load failed: {e}")
            print("Rebuilding GPU index...")
            import shutil
            shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)
            return VectorStoreManager.create_or_load_vector_store(chunks)

# =================================
# CLAIM ANALYZER (GPU-OPTIMIZED)
# =================================
class ClaimAnalyzer:
    def __init__(self):
        self.gpu_enabled = faiss.get_num_gpus() > 0
        if not self.gpu_enabled:
            raise RuntimeError("GPU acceleration required but not available")

        print("Initializing GPU-accelerated analyzer...")
        self.chunks = DocumentProcessor.load_and_chunk_manuals()
        self.vectorstore = VectorStoreManager.create_or_load_vector_store(self.chunks)
        self.retriever = self._create_retrievers()
        self.chain = self._setup_llm_chain()

    def _create_retrievers(self):
        bm25 = BM25Retriever.from_documents(self.chunks)
        bm25.k = 2
        faiss_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 25}
        )
        return EnsembleRetriever(
            retrievers=[bm25, faiss_retriever],
            weights=[0.4, 0.6]
        )

    def _setup_llm_chain(self):
        prompt = ChatPromptTemplate.from_template("""
        [Your existing prompt template here]
        """)

        llm = OllamaLLM(
            model="llama3:8b-instruct-q4_0",
            temperature=0,
            num_ctx=4096,
            num_thread=4,
            top_k=30,
            device="cuda"  # Force GPU inference
        )

        document_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(self.retriever, document_chain)

    def analyze_claim(self, claim_data: Dict) -> Dict:
        response = self.chain.invoke({
            "input": self._generate_query(claim_data),
            "context": self._filter_docs(claim_data),
            **claim_data
        })
        return self._parse_response(response)

    # [Include all your original helper methods:
    # _filter_docs, _parse_response, _generate_query]

# =================================
# FULLY GPU-OPTIMIZED AGENT
# =================================
class MedicareClaimAgent:
    def __init__(self):
        print("Initializing pure GPU agent...")
        self.analyzer = ClaimAnalyzer()
        self.tools = self._setup_tools()
        self.llm = OllamaLLM(
            model="llama3:8b-instruct-q4_0",
            device="cuda",
            temperature=0,
            num_ctx=4096
        )
        self.agent = self._create_agent()
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def _setup_tools(self):
        @tool
        def analyze_claim(claim_data: Dict) -> Dict:
            """Analyze claim using GPU-accelerated RAG"""
            return self.analyzer.analyze_claim(claim_data)

        @tool
        def update_policies() -> str:
            """Refresh GPU-optimized knowledge base"""
            chunks = DocumentProcessor.load_and_chunk_manuals()
            VectorStoreManager.create_or_load_vector_store(chunks)
            return "GPU index updated"

        return [analyze_claim, update_policies]

    def _create_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a GPU-accelerated Medicare claims agent. Rules:
1. Always use tools
2. Maintain maximum GPU utilization
3. Respond with valid JSON"""),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")  # Critical for agent operation
        ])
        return create_tool_calling_agent(self.llm, self.tools, prompt)

    def run(self, input_text: str) -> Dict:
        """Execute with GPU acceleration"""
        try:
            return self.executor.invoke({
                "input": input_text,
                "agent_scratchpad": []  # Required field
            })
        except Exception as e:
            print(f"GPU agent error: {e}")
            raise

# =================================
# MAIN EXECUTION
# =================================
if __name__ == "__main__":
    try:
        # Verify GPU environment
        print("\n=== GPU Verification ===")
        print(f"FAISS GPUs: {faiss.get_num_gpus()}")
        print(f"CUDA Devices: {os.popen('nvidia-smi -L').read().strip()}")
        print(f"CUDA Version: {os.popen('nvcc --version').read().strip()}")

        # Initialize
        agent = MedicareClaimAgent()

        # Test claim
        test_claim = {
            "cpt_code": "99214",
            "diagnosis": "Z79.899",
            "modifiers": [],
            "payer": "Medicare"
        }

        result = agent.run(
            f"Analyze CPT {test_claim['cpt_code']} with diagnosis {test_claim['diagnosis']}"
        )
        print("\nGPU-accelerated Results:")
        print(json.dumps(result["output"], indent=2))

    except Exception as e:
        print(f"\nGPU Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify GPU memory: nvidia-smi")
        print("2. Check FAISS-GPU installation:")
        print("   conda install -c pytorch faiss-gpu cudatoolkit=12.1")
        print("3. To reset: rm -rf cms_manuals_faiss_index_gpu/")