import fitz  # pymupdf
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ---- STEP 1: Load and extract text from PDF ----
def load_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

print("Loading PDF...")
raw_text = load_pdf("attention.pdf")
print(f"Extracted {len(raw_text)} characters from PDF")
print("\nFirst 500 characters preview:")
print(raw_text[:500])



# ---- STEP 2: Chunk the text ----
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # skip empty chunks
            chunks.append(chunk)
        start = end - overlap  # move forward but keep some overlap
    return chunks

print("\nChunking text...")
chunks = chunk_text(raw_text)
print(f"Total chunks created: {len(chunks)}")
print("\nSample chunk:")
print(chunks[5])


# ---- STEP 3: Embed and store in ChromaDB ----
def build_vector_store(chunks):
    print("\nLoading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="./my_db")

    # If collection already exists, just return it
    existing = [c.name for c in client.list_collections()]
    if "attention_paper" in existing:
        print("Found existing vector database, skipping indexing!")
        collection = client.get_collection("attention_paper")
        return embedder, collection

    print("Embedding chunks... this may take a moment...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)

    collection = client.create_collection("attention_paper")
    collection.add(
        documents=chunks,
        embeddings=[e.tolist() for e in embeddings],
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    print(f"Stored {collection.count()} chunks in vector database!")
    return embedder, collection

embedder, collection = build_vector_store(chunks)

# ---- STEP 4: Retrieve and generate ----
def retrieve(question, embedder, collection, n_results=5):
    question_embedding = embedder.encode([question])
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=n_results
    )
    return results["documents"][0]

def generate(question, context, client):
    prompt = f"""You are a helpful assistant that answers questions about the paper "Attention is All You Need".
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}
"""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ---- STEP 5: Chat loop ----
print("\nReady! Ask me anything about the Attention is All You Need paper.")
print("Type 'quit' to exit.\n")

groq_client = Groq()

while True:
    question = input("You: ")

    if question.lower() == "quit":
        break

    chunks_retrieved = retrieve(question, embedder, collection)

    # temporary debug line
    print("\n--- Retrieved Chunks ---")
    for i, chunk in enumerate(chunks_retrieved):
        print(f"\nChunk {i+1}:\n{chunk}")
    print("--- End of Chunks ---\n")

    context = "\n\n".join(chunks_retrieved)
    answer = generate(question, context, groq_client)

    print(f"\nAssistant: {answer}\n")


    

