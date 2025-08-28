import re
import streamlit as st
import fitz  # PyMuPDF
import json

try:
    from regexgen import regexgen
except ImportError:
    regexgen = None

# --- Extract text from PDF ---
def extract_text(uploaded_file):
    text = []
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text.append(page.get_text("text"))
    return "\n".join(text)

# --- Regex generator ---
def generate_regex(samples):
    if regexgen:  # use regexgen if installed
        return regexgen(samples)
    # fallback heuristic
    parts = []
    for s in samples:
        s = re.escape(s.strip())
        s = re.sub(r"\\d+", r"\\d+", s)  # normalize numbers
        s = re.sub(r"[A-Z]{2,}", r"[A-Z ]+", s)  # normalize ALLCAPS
        parts.append(s)
    return "|".join(parts)

# --- Chunker ---
def chunk_with_regex(text, regex, max_len=2000, overlap=0):
    lines = text.split("\n")
    chunks, current = [], []
    for line in lines:
        if re.match(regex, line.strip()):
            if current:
                chunks.append("\n".join(current))
                if overlap > 0 and len(current) > overlap:
                    current = current[-overlap:]  # add overlap
                else:
                    current = []
        current.append(line)
        if len("\n".join(current)) > max_len:
            chunks.append("\n".join(current))
            current = []
    if current:
        chunks.append("\n".join(current))
    return chunks

# --- Streamlit App ---
st.title("📑 Interactive Regex-Based Chunker")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
max_len = st.slider("Max chunk length (chars)", 500, 5000, 2000, step=500)
overlap = st.slider("Overlap lines", 0, 10, 0)

if uploaded:
    text = extract_text(uploaded)
    preview_lines = text.split("\n")[:200]

    # Step 1: Choose what you’re chunking by
    mode = st.radio(
        "Select what you want to chunk by:",
        ["Headers", "Paragraphs", "Chapters", "Custom"]
    )

    st.subheader("Document Preview (first 200 lines)")
    selected_lines = st.multiselect(
        f"👉 Select multiple {mode.lower()} from the preview below",
        preview_lines
    )

    if selected_lines:
        regex = generate_regex(selected_lines)
        st.write("### Generated Regex Pattern")
        st.code(regex)

        if st.button("Run Chunking"):
            chunks = chunk_with_regex(text, regex, max_len=max_len, overlap=overlap)
            st.success(f"✅ Generated {len(chunks)} chunks!")

            # Stats
            lengths = [len(c) for c in chunks]
            st.write("### Stats")
            st.write(f"- Avg length: {sum(lengths)//len(lengths)} chars")
            st.write(f"- Min length: {min(lengths)} chars")
            st.write(f"- Max length: {max(lengths)} chars")

            # Preview
            st.write("### 🔎 First 5 Chunks")
            for i, c in enumerate(chunks[:5]):
                st.write(f"**Chunk {i+1}** ({len(c)} chars)")
                st.text(c[:500] + "...")

            # Download regex
            regex_file = json.dumps({"regex": regex}, indent=2)
            st.download_button(
                "⬇️ Download Regex Pattern",
                regex_file,
                file_name="regex_pattern.json",
                mime="application/json"
            )
