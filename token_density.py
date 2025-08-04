import os
import csv
from transformers import AutoTokenizer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

CMS_MANUAL_PATH = "/media/udonsi-kalu/New Volume/denials/denials/manuals/"
PRIORITY_CHAPTERS = {
    "Chapter 1": "General Billing Requirements",
    "Chapter 12": "Physicians Nonphysician Practitioners",
    "Chapter 23": "Fee Schedule Administration",
    "Chapter 30": "Financial Liability Protections"
}

def load_and_chunk_manuals():
    loaders = []
    for chap, title in PRIORITY_CHAPTERS.items():
        filename = f"{chap} - {title}.pdf"
        path = os.path.join(CMS_MANUAL_PATH, filename)
        if os.path.exists(path):
            loaders.append(PyPDFLoader(path))
        else:
            print(f"Warning: {filename} not found")

    if not loaders:
        raise ValueError("No CMS manual PDFs found")

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "Chapter"), ("##", "Section"), ("###", "Subsection"), ("####", "PolicyNumber")
    ])
    content_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=2000,
        separators=["\n\n", "\n", r"(?<=\. )", " "]
    )

    chunks = []
    for loader in loaders:
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = loader.file_path
            doc.metadata["page"] = doc.metadata.get("page", -1)
            headers = header_splitter.split_text(doc.page_content)

            for header_doc in headers:
                # Determine heading depth from header style (count '#' at start)
                header_text = header_doc.metadata.get("header_text", "")
                if header_text.startswith("####"):
                    depth = 4
                elif header_text.startswith("###"):
                    depth = 3
                elif header_text.startswith("##"):
                    depth = 2
                elif header_text.startswith("#"):
                    depth = 1
                else:
                    depth = 0
                header_doc.metadata["heading_depth"] = depth

                # Split further into smaller chunks preserving heading_depth metadata
                sub_chunks = content_splitter.split_documents([header_doc])
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata["heading_depth"] = depth
                chunks.extend(sub_chunks)

    print(f"Loaded and chunked {len(chunks)} chunks from CMS manuals.")
    return chunks

# Initialize tokenizer (use a lightweight tokenizer like bert-base-uncased or similar)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def compute_token_density(chunks):
    densities = []
    for chunk in chunks:
        tokens = tokenizer.tokenize(chunk.page_content)
        densities.append(len(tokens))
    return densities

def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def compute_neighbor_overlap(chunks):
    overlaps = []
    for i in range(len(chunks) - 1):
        tokens_a = set(tokenizer.tokenize(chunks[i].page_content))
        tokens_b = set(tokenizer.tokenize(chunks[i + 1].page_content))
        overlap_score = jaccard_similarity(tokens_a, tokens_b)
        overlaps.append(overlap_score)
    overlaps.append(0.0)  # last chunk has no neighbor ahead
    return overlaps

def save_maps_to_csv(chunks, token_density, heading_depth, neighbor_overlap, output_path="passive_maps.csv"):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_index", "token_density", "heading_depth", "neighbor_overlap", "source_file", "page"])
        for i, chunk in enumerate(chunks):
            writer.writerow([
                i,
                token_density[i],
                heading_depth[i],
                neighbor_overlap[i],
                chunk.metadata.get("source_file", "unknown"),
                chunk.metadata.get("page", -1)
            ])
    print(f"Saved passive maps to {output_path}")

if __name__ == "__main__":
    chunks = load_and_chunk_manuals()
    token_density = compute_token_density(chunks)
    heading_depth = [chunk.metadata.get("heading_depth", 0) for chunk in chunks]
    neighbor_overlap = compute_neighbor_overlap(chunks)

    save_maps_to_csv(chunks, token_density, heading_depth, neighbor_overlap)
