import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ---- Page config ----
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄"
)

st.title("📄 Chat with any PDF")
st.caption("Upload a PDF and ask anything about it!")

# ---- Helper functions ----
def load_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

def build_vector_store(chunks):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()
    collection = client.create_collection("pdf_collection")
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    collection.add(
        documents=chunks,
        embeddings=[e.tolist() for e in embeddings],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return embedder, collection

def retrieve(question, embedder, collection, n_results=5):
    question_embedding = embedder.encode([question])
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=n_results
    )
    return results["documents"][0]

def generate(question, context):
    client = Groq()
    prompt = f"""You are a helpful assistant that answers questions about an uploaded PDF document.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ---- Sidebar for PDF upload ----
with st.sidebar:
    st.header("Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        if "collection" not in st.session_state or st.session_state.filename != uploaded_file.name:
            with st.spinner("Reading and indexing PDF..."):
                raw_text = load_pdf(uploaded_file)
                chunks = chunk_text(raw_text)
                embedder, collection = build_vector_store(chunks)
                st.session_state.embedder = embedder
                st.session_state.collection = collection
                st.session_state.filename = uploaded_file.name
                st.session_state.messages = []
            st.success(f"Indexed {len(chunks)} chunks!")

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
                chunks_retrieved = retrieve(question, st.session_state.embedder, st.session_state.collection)
                context = "\n\n".join(chunks_retrieved)
                answer = generate(question, context)
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})