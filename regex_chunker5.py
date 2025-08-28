import re
import fitz
import streamlit as st

def extract_text(file):
    text = []
    if hasattr(file, "read"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
    else:
        doc = fitz.open(file)
    for page in doc:
        text.append(page.get_text("text"))
    return "\n".join(text)

def chunk_by_regex(text, pattern, overlap=0):
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if overlap > 0 and i > 0:
            prev_start = matches[i - 1].start()
            overlap_text = text[max(prev_start, start - overlap):start]
            chunk = overlap_text + "\n" + chunk
        chunks.append(chunk)
    return chunks

st.title("📄 Interactive Chunk Picker")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
mode = st.radio("Choose mode", ["Paragraphs", "Headers/Chapters", "Custom Selection"])

overlap = st.slider("Overlap characters", 0, 200, 0)

if uploaded:
    text = extract_text(uploaded)
    lines = text.split("\n")
    
    if mode == "Paragraphs":
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]

    elif mode == "Headers/Chapters":
        # Let user pick a line as a "sample delimiter"
        st.subheader("Pick a line that looks like a header/chapter")
        preview_lines = lines[:200]  # show first 200 lines
        picked_line = st.selectbox("Select a sample line", preview_lines)
        
        if picked_line:
            # Auto-generate a regex based on the style of the picked line
            if picked_line.lower().startswith("chapter"):
                regex = r"^Chapter\s+\d+.*$"
            elif re.match(r"^\d+(\.\d+)+", picked_line):
                regex = r"^\d+(\.\d+)*\s.+$"
            else:
                regex = re.escape(picked_line[:10])  # fallback: literal match
            
            st.write(f"Using regex: `{regex}`")
            chunks = chunk_by_regex(text, regex, overlap)

    elif mode == "Custom Selection":
        # User types their own regex
        user_regex = st.text_input("Enter your regex", r"^\d+(\.\d+)*\s.+$")
        chunks = chunk_by_regex(text, user_regex, overlap) if user_regex else []

    # --- Stats ---
    if "chunks" in locals() and chunks:
        st.subheader("📊 Chunking Stats")
        lengths = [len(c) for c in chunks]
        st.write(f"Total chunks: {len(chunks)}")
        st.write(f"Average length: {sum(lengths)/len(lengths):.1f} chars")
        st.write(f"Min length: {min(lengths)} chars")
        st.write(f"Max length: {max(lengths)} chars")

        st.subheader("🔎 Preview First 5 Chunks")
        for i, chunk in enumerate(chunks[:5]):
            st.markdown(f"**Chunk {i+1}** ({len(chunk)} chars)")
            st.text(chunk[:500])
