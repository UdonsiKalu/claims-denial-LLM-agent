import re
import fitz  # PyMuPDF
import streamlit as st

# --- Utility: extract text from PDF ---
def extract_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)

# --- Regex-based chunker with overlap ---
def chunk_with_regex(text, regex, max_len=1000, overlap=200):
    lines = text.split("\n")
    chunks, current = [], []

    for line in lines:
        if re.match(regex, line.strip()):
            if current:
                chunk = "\n".join(current)
                chunks.append(chunk)
                # 🔹 keep tail for overlap
                if overlap > 0:
                    overlap_text = chunk[-overlap:]
                    current = [overlap_text]
                else:
                    current = []
        current.append(line)

        # max length safety cutoff
        if len("\n".join(current)) > max_len:
            chunk = "\n".join(current)
            chunks.append(chunk)
            if overlap > 0:
                overlap_text = chunk[-overlap:]
                current = [overlap_text]
            else:
                current = []

    if current:
        chunks.append("\n".join(current))

    return chunks

# --- Streamlit UI ---
st.title("📑 Regex Chunker with Overlap")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    text = extract_text(uploaded)
    preview_lines = text.split("\n")[:200]

    st.subheader("Document Preview (first 200 lines)")
    st.text("\n".join(preview_lines))

    # Regex input
    regex = st.text_input("Enter regex for headers", r"^Chapter\s+\d+")
    
    # Chunking parameters
    max_len = st.slider("Max chunk length (chars)", 500, 5000, 1000, step=100)
    overlap = st.slider("Overlap size (chars)", 0, 500, 200, step=50)

    if st.button("Run Chunking"):
        chunks = chunk_with_regex(text, regex, max_len=max_len, overlap=overlap)

        st.success(f"Generated {len(chunks)} chunks with overlap={overlap}")
        st.write({
            "Total chunks": len(chunks),
            "Average length": sum(len(c) for c in chunks) / len(chunks),
            "Min length": min(len(c) for c in chunks),
            "Max length": max(len(c) for c in chunks),
        })

        # Preview a few chunks
        for i, chunk in enumerate(chunks[:5]):
            st.markdown(f"**Chunk {i+1}** ({len(chunk)} chars)")
            st.text(chunk[:500] + "...")
