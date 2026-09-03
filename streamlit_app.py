import hashlib
import os

import streamlit as st
import pymupdf as fitz  # `import fitz` is deprecated by PyMuPDF as of 1.24+
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq, GroqError
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = "./my_db"
MAX_FILE_SIZE_MB = 50
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ---- Page config ----
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄"
)

st.title("📄 Chat with any PDF")
st.caption("Upload a PDF and ask anything about it!")

# ---- Helper functions ----
def load_pdf_pages(uploaded_file):
    """Returns a list of (page_number, page_text), 1-indexed, instead of
    one flat string - this is what lets us cite a page number back to the
    user later."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return [(i + 1, page.get_text()) for i, page in enumerate(doc)]

def chunk_pages(pages, chunk_size=500, overlap=50):
    """Chunk each page independently (rather than the whole document as one
    string) so every chunk can be tagged with the single page it came from."""
    chunks, page_numbers = [], []
    for page_number, text in pages:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
                page_numbers.append(page_number)
            start = end - overlap
    return chunks, page_numbers

def file_hash(uploaded_file):
    """Stable id for a file's contents, used as the Chroma collection name
    so re-uploading the same PDF reuses its index instead of crashing on
    a duplicate collection name."""
    uploaded_file.seek(0)
    digest = hashlib.sha256(uploaded_file.read()).hexdigest()[:16]
    uploaded_file.seek(0)
    return f"pdf_{digest}"

@st.cache_resource(show_spinner=False)
def get_embedder():
    # Cached across reruns/sessions so the model loads once per process,
    # not once per PDF upload.
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    return chromadb.PersistentClient(path=VECTOR_STORE_PATH)

def build_vector_store(chunks, page_numbers, collection_name):
    embedder = get_embedder()
    client = get_chroma_client()

    collection = client.get_or_create_collection(collection_name)

    if collection.count() > 0:
        # Already indexed this exact file in a previous run.
        return embedder, collection, collection.count()

    embeddings = embedder.encode(chunks, show_progress_bar=False)
    collection.add(
        documents=chunks,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[{"page": p} for p in page_numbers],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return embedder, collection, len(chunks)

def retrieve(question, embedder, collection, n_results=5):
    question_embedding = embedder.encode([question])
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=n_results
    )
    documents = results["documents"][0]
    pages = [m["page"] for m in results["metadatas"][0]]
    return documents, pages

def generate(question, context):
    client = Groq()
    prompt = f"""You are a helpful assistant that answers questions about an uploaded PDF document.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}
"""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
    except GroqError as e:
        raise RuntimeError(f"The Groq API request failed: {e}") from e

    return response.choices[0].message.content

# ---- Sidebar for PDF upload ----
with st.sidebar:
    st.header("Upload your PDF")

    if not os.getenv("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY is not set. Add it to a .env file in the project "
            "root (see README) before uploading a PDF."
        )
        st.stop()

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File is {size_mb:.1f} MB. Max supported size is {MAX_FILE_SIZE_MB} MB.")
            st.stop()

        collection_name = file_hash(uploaded_file)
        if "collection_name" not in st.session_state or st.session_state.collection_name != collection_name:
            with st.spinner("Reading and indexing PDF..."):
                try:
                    pages = load_pdf_pages(uploaded_file)
                except Exception:
                    st.error("Couldn't read this file. Is it a valid, non-encrypted PDF?")
                    st.stop()

                chunks, page_numbers = chunk_pages(pages)

                if not chunks:
                    st.error(
                        "No extractable text found in this PDF. It may be a "
                        "scanned/image-only document - OCR isn't supported yet."
                    )
                    st.stop()

                try:
                    embedder, collection, chunk_count = build_vector_store(
                        chunks, page_numbers, collection_name
                    )
                except Exception as e:
                    st.error(f"Failed to build the vector index: {e}")
                    st.stop()

                st.session_state.embedder = embedder
                st.session_state.collection = collection
                st.session_state.collection_name = collection_name
                st.session_state.filename = uploaded_file.name
                st.session_state.messages = []
            st.success(f"Indexed {chunk_count} chunks across {len(pages)} pages!")

# ---- Chat interface ----
if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection" not in st.session_state:
    st.info("👈 Upload a PDF from the sidebar to get started!")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                st.caption(f"📄 Sources: page {message['sources']}")

    # Chat input
    question = st.chat_input("Ask anything about your PDF...")

    if question:
        # Show user message
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Generate and show answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chunks_retrieved, pages_retrieved = retrieve(
                        question, st.session_state.embedder, st.session_state.collection
                    )
                    context = "\n\n".join(chunks_retrieved)
                    answer = generate(question, context)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()
            st.write(answer)

            sources = ", ".join(str(p) for p in sorted(set(pages_retrieved)))
            st.caption(f"📄 Sources: page {sources}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })