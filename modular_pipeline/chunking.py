# === chunking.py ===
import os
import json
import pickle
import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document

from constants import CMS_MANUAL_PATH, PRIORITY_CHAPTERS

# -------------------------------
# Simple on-disk cache for chunks
# -------------------------------
_CHUNK_CACHE_DIR = Path(".cache/chunks")
_CHUNK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(key: str) -> Path:
    return _CHUNK_CACHE_DIR / f"{key}.pkl"


def _cache_exists(key: str) -> bool:
    return _cache_path(key).exists()


def _cache_load(key: str):
    with _cache_path(key).open("rb") as f:
        return pickle.load(f)


def _cache_save(key: str, value) -> None:
    with _cache_path(key).open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)


def _corpus_fingerprint(paths: list[str]) -> str:
    """Fingerprint PDFs (mtime+size) so cache invalidates on change."""
    items = []
    for p in paths:
        try:
            st = os.stat(p)
            items.append({"p": os.path.basename(p), "mt": int(st.st_mtime), "sz": st.st_size})
        except OSError:
            items.append({"p": os.path.basename(p), "mt": -1, "sz": -1})
    blob = json.dumps(items, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


# -------------------------------
# Enforce hard max chunk size
# -------------------------------
def _enforce_chunk_size(chunks, max_size, overlap=0):
    """Force-split any oversized chunks into <= max_size chars."""
    if not chunks or not max_size:
        return chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""],  # progressively break smaller
    )

    final = []
    for c in chunks:
        if len(c.page_content) > max_size:
            sub_docs = splitter.split_documents([c])
            for sd in sub_docs:
                final.append(Document(page_content=sd.page_content, metadata=c.metadata.copy()))
        else:
            final.append(c)
    return final


# -------------------------------
# Main loader & chunker
# -------------------------------
def load_and_chunk_manuals(
    chunking_strategy="Fixed-size",
    chunk_size=None,
    chunk_overlap=None,
    header_levels=None,
    semantic_threshold=0.5,
    embeddings=None,
    include_headers=True,
    adaptive_overlap=False,
    min_tokens=None,
    max_tokens=None,
    min_chunk_size=0,
    page_group=None,
    input_files=None,       # NEW: override default CMS manuals
    enforce_max_size=True,  # NEW: toggle oversize enforcement
):
    """
    Load CMS manuals and split into chunks using chosen strategy.
    Supports: Fixed-size, Recursive, Header-aware, Semantic, By-page.
    Extra knobs: include_headers, adaptive_overlap, min/max tokens, min_chunk_size, page_group.
    """

    chunk_size = int(os.environ.get("CHUNK_SIZE", chunk_size or 10000))
    base_overlap = int(os.environ.get("CHUNK_OVERLAP", chunk_overlap or 2000))
    force_rebuild = os.environ.get("FORCE_REBUILD") == "1"

    # --- load PDFs ---
    loaders, pdf_paths = [], []

    if input_files:  # use provided files
        for path in input_files:
            if os.path.exists(path):
                loaders.append(("Custom", path))
                pdf_paths.append(path)
            else:
                print(f"⚠️ Warning: {path} not found")
    else:  # fallback to PRIORITY_CHAPTERS
        for chap, title in PRIORITY_CHAPTERS.items():
            filename = f"{chap} - {title}.pdf"
            path = os.path.join(CMS_MANUAL_PATH, filename)
            if os.path.exists(path):
                loaders.append((chap, path))
                pdf_paths.append(path)
            else:
                print(f"⚠️ Warning: {filename} not found")

    if not loaders:
        raise ValueError("❌ No PDF files found")

    # --- cache key with ALL knobs ---
    fp = _corpus_fingerprint(pdf_paths)
    cache_key = (
        f"{chunking_strategy}_c{chunk_size}_o{base_overlap}_h{header_levels}"
        f"_s{semantic_threshold}_ih{include_headers}_ao{adaptive_overlap}"
        f"_tok{min_tokens}-{max_tokens}_pg{page_group}_m{min_chunk_size}_em{enforce_max_size}_{fp}"
    )

    if not force_rebuild and _cache_exists(cache_key):
        print(f"🗃️ Using cached chunks: {cache_key}")
        return _cache_load(cache_key)

    # --- load documents ---
    all_docs = []
    for chapter, path in loaders:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = os.path.basename(path)
            d.metadata["chapter"] = chapter
        all_docs.extend(docs)

    print(f"📘 Loaded {len(all_docs)} docs")

    # --- helper: adaptive overlap calculation ---
    def _get_overlap(text: str) -> int:
        if not adaptive_overlap:
            return base_overlap
        length = len(text)
        if length >= chunk_size:
            return base_overlap
        return max(0, int(base_overlap * (length / chunk_size)))

    # --- chunk by strategy ---
    strategy = chunking_strategy.lower()
    chunks = []

    if strategy.startswith("fixed"):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=base_overlap,
        )
        chunks = splitter.split_documents(all_docs)

    elif strategy.startswith("recursive"):
        chunks = []
        for d in all_docs:
            ov = _get_overlap(d.page_content)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=ov,
                separators=["\n\n", "\n", r"(?<=\. )", " "],
            )
            chunks.extend(splitter.split_documents([d]))

    elif strategy.startswith("header"):
        levels = header_levels or 3
        headers_to_split_on = [(f"h{i}", f"H{i}") for i in range(1, levels + 1)]
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        for d in all_docs:
            header_docs = header_splitter.split_text(d.page_content)
            for c in header_docs:
                ov = _get_overlap(c.page_content)
                inner_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=ov,
                    separators=["\n\n", "\n", ".", " ", ""],
                )
                inner_chunks = inner_splitter.split_text(c.page_content)
                for ic in inner_chunks:
                    text = ic if include_headers else ic.replace(c.metadata.get("header", ""), "")
                    chunks.append(Document(page_content=text, metadata={**d.metadata, **c.metadata}))

    elif strategy.startswith("semantic"):
        if embeddings is None:
            raise ValueError("SemanticChunker requires embeddings")

        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="interquartile",
            breakpoint_threshold_amount=float(semantic_threshold or 0.5),
        )
        chunks = splitter.split_documents(all_docs)

        if adaptive_overlap:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            adjusted_chunks = []
            prev_vec = None
            window = 200  # chars from end/start to compare

            for i, c in enumerate(chunks):
                text = c.page_content

                if prev_vec is not None:
                    curr_vec = embeddings.embed_query(text[:window])
                    sim = cosine_similarity([prev_vec], [curr_vec])[0][0]

                    if sim > 0.75:
                        overlap_size = int(len(text) * 0.25)
                    elif sim > 0.5:
                        overlap_size = int(len(text) * 0.15)
                    else:
                        overlap_size = int(len(text) * 0.05)

                    if overlap_size > 0 and adjusted_chunks:
                        adjusted_chunks[-1].page_content += "\n" + text[:overlap_size]

                    adjusted_chunks.append(
                        Document(page_content=text[overlap_size:], metadata=c.metadata)
                    )
                else:
                    adjusted_chunks.append(c)

                prev_vec = embeddings.embed_query(text[-window:])

            chunks = adjusted_chunks

    elif strategy.startswith("by-page"):
        chunks = all_docs

    else:
        print(f"⚠️ Unknown strategy '{chunking_strategy}', defaulting to fixed-size")
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=base_overlap)
        chunks = splitter.split_documents(all_docs)

    # --- enforce hard max chunk size (only if enabled) ---
# --- enforce hard max chunk size (only if enabled) ---
    if enforce_max_size:
        # normalize: dict → Document, str → Document
        norm_chunks = []
        for c in chunks:
            if isinstance(c, Document):
                norm_chunks.append(c)
            elif isinstance(c, dict):
                norm_chunks.append(Document(page_content=c.get("content", ""), metadata=c.get("metadata", {})))
            else:  # plain string
                norm_chunks.append(Document(page_content=str(c), metadata={}))
        chunks = _enforce_chunk_size(norm_chunks, chunk_size, base_overlap)

    # --- normalize, filter, enforce min_chunk_size ---
    final_chunks = []
    for i, c in enumerate(chunks):
        if min_chunk_size and len(c.page_content) < min_chunk_size:
            continue
        if min_tokens and len(c.page_content.split()) < min_tokens:
            continue
        if max_tokens and len(c.page_content.split()) > max_tokens:
            continue

        meta = dict(c.metadata)
        pg = meta.get("page", i)
        try:
            meta["page"] = int(pg)
        except Exception:
            meta["page"] = pg
        meta["chunk_id"] = f"{meta.get('source','unknown')}::Page_{meta['page']}::Chunk_{i}"

        final_chunks.append(Document(page_content=c.page_content, metadata=meta))

    print(
        f"✅ Created {len(final_chunks)} chunks with metadata "
        f"(strategy={chunking_strategy}, enforce_max_size={enforce_max_size})"
    )
