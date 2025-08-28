import re
import fitz  # PyMuPDF
import streamlit as st

# --- Extract text from PDF ---
def extract_text(file_path):
    text = []
    doc = fitz.open(file_path)
    for page in doc:
        text.append(page.get_text("text"))
    return "\n".join(text)

# --- Chunking helpers ---
def regex_chunk(text, pattern, overlap=0):
    lines = text.split("\n")
    chunks, current = [], []

    for line in lines:
        if re.match(pattern, line.strip()):
            if current:
                chunks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current))

    # add overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk = "\n".join(chunks[i-1].splitlines()[-overlap:] + chunk.splitlines())
            overlapped.append(chunk)
        chunks = overlapped

    return chunks

def split_paragraphs(text, overlap=0):
    paras = text.split("\n\n")
    if overlap > 0 and len(paras) > 1:
        overlapped = []
        for i, para in enumerate(paras):
            if i > 0:
                para = paras[i-1][-overlap:] + para
            overlapped.append(para)
        paras = overlapped
    return paras

def split_pages(text):
    return text.split("\f")  # form feed = page break

# --- Auto-generate regex (basic heuristics) ---
AUTO_PATTERNS = {
    "Chapters": r"^Chapter\s+\d+",
    "Sections": r"^Section\s+\d+",
    "Numbered Headings": r"^\d+(\.\d+)+",
    "ALL CAPS Headings": r"^[A-Z][A-Z ]{3,}$",
    "Articles": r"^Article\s+\d+"
}

def auto_detect_chunk(text, overlap=0):
    lines = text.split("\n")
    best_name, best_pattern, best_hits = None, None, 0
    for name, pattern in AUTO_PATTERNS.items():
        hits = sum(1 for l in lines if re.match(pattern, l.strip()))
        if hits > best_hits:
            best_hits, best_pattern, best_name = hits, pattern, name
    if not best_pattern:
        return [text], None
    return regex_chunk(text, best_pattern, overlap=overlap), best_name

# --- Streamlit UI ---
st.title("📑 Flexible Chunking GUI")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
mode = st.radio("Choose split mode:", ["Chapters (Auto)", "Headers (Regex)", "Paragraphs", "Pages"])

overlap = st.slider("Overlap lines (for regex/paragraph):", 0, 5, 0)

if uploaded:
    text = extract_text(uploaded)

    if mode == "Chapters (Auto)":
        chunks, pattern_used = auto_detect_chunk(text, overlap=overlap)
        st.info(f"Auto-detected: {pattern_used}")

    elif mode == "Headers (Regex)":
        regex_pattern = st.text_input("Enter regex for headers:", r"^Chapter\s+\d+")
        chunks = regex_chunk(text, regex_pattern, overlap=overlap)

    elif mode == "Paragraphs":
        chunks = split_paragraphs(text, overlap=overlap)

    else:  # Pages
        chunks = split_pages(text)

    # Stats
    lengths = [len(c) for c in chunks]
    st.success(f"Produced {len(chunks)} chunks with {mode}")
    st.write(f"Average length: {sum(lengths)//len(lengths)} chars")
    st.write(f"Min length: {min(lengths)} chars")
    st.write(f"Max length: {max(lengths)} chars")

    # Optional preview
    with st.expander("Preview first 3 chunks"):
        for i, chunk in enumerate(chunks[:3]):
            st.text_area(f"Chunk {i+1}", chunk[:2000], height=200)
