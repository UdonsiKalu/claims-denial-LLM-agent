import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from constants import CMS_MANUAL_PATH, PRIORITY_CHAPTERS

def load_and_chunk_manuals(chunk_size=10000, chunk_overlap=2000):
    print(f"> Chunking with size={chunk_size} and overlap={chunk_overlap}")

    loaders = []
    for chap, title in PRIORITY_CHAPTERS.items():
        filename = f"{chap} - {title}.pdf"
        path = os.path.join(CMS_MANUAL_PATH, filename)
        if os.path.exists(path):
            loaders.append((chap, path))
        else:
            print(f"Warning: {filename} not found")

    if not loaders:
        raise ValueError("No CMS manual PDFs found")

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "Chapter"), ("##", "Section"), ("###", "Subsection"), ("####", "PolicyNumber")
    ])
    content_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", r"(?<=\. )", " "]
    )

    all_chunks = []
    for chapter, path in loaders:
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            for doc in docs:
                headers = header_splitter.split_text(doc.page_content)
                split_chunks = content_splitter.split_documents(headers)
                for chunk in split_chunks:
                    chunk.metadata["source"] = os.path.basename(path)
                    chunk.metadata["chapter"] = chapter
                    chunk.metadata["page"] = doc.metadata.get("page", "0")
                all_chunks.extend(split_chunks)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    print(f"✅ Created {len(all_chunks)} chunks with metadata.")
    return all_chunks
