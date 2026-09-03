# 📄 Chat with any PDF — RAG Application

A Retrieval-Augmented Generation (RAG) application that lets you upload a PDF and ask questions about it in natural language, with answers grounded in the document and cited back to a page number.

Built from scratch without a framework like LangChain or LlamaIndex, to show — and be able to explain — how each stage of a RAG pipeline actually works, rather than calling a single black-box abstraction.

## How it Works

```
 PDF upload
     │
     ▼
 Per-page text extraction (PyMuPDF)
     │
     ▼
 Chunking (500 chars, 50 overlap) — chunked per page so each
 chunk keeps a single, correct page number
     │
     ▼
 Embedding (sentence-transformers, all-MiniLM-L6-v2)
     │
     ▼
 Vector store (ChromaDB, persisted to disk, keyed by a hash
 of the file so re-uploads reuse the existing index)
     │
     ▼
 Query  ──▶  embed query ──▶ top-k retrieval (dot product) ──▶ context
     │
     ▼
 LLM generation (Llama 3.3 70B via Groq), grounded strictly in
 the retrieved context, answer shown with its source page(s)
```

1. **Upload** a PDF via the sidebar
2. Text is extracted **per page** (not as one flat blob), so page numbers survive into the index
3. Each page's text is **chunked** into 500-character pieces with 50-character overlap
4. Each chunk is **embedded** and **stored** in a persistent ChromaDB collection, tagged with its source page
5. A question is embedded and the most similar chunks are **retrieved**, along with which page(s) they came from
6. Retrieved chunks are passed as **context** to the LLM, which generates a grounded answer — shown with a **"Sources: page X, Y"** caption underneath

## Tech Stack

- **Frontend** — Streamlit
- **PDF parsing** — PyMuPDF
- **Embeddings** — sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector database** — ChromaDB (persistent, on-disk)
- **LLM** — Llama 3.3 70B via the Groq API

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/SaiKishoreReddy07/chat-with-pdf.git
cd chat-with-pdf
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 5. Run the app
```bash
streamlit run streamlit_app.py
```

A sample PDF ("Attention Is All You Need") is included in `examples/` if you want something to test with immediately — the CLI version (`app.py`) uses it by default.

## Project Structure

```
.
├── streamlit_app.py     # Streamlit UI and full RAG pipeline
├── app.py                # CLI version of the RAG pipeline (uses examples/attention.pdf)
├── examples/
│   └── attention.pdf     # Sample PDF for quick testing
├── requirements.txt       # Pinned Python dependencies
└── my_db/                 # Persistent vector store (gitignored, created on first run)
```

## Key Concepts Demonstrated

- Page-aware chunking strategies for document processing
- Semantic search using vector embeddings
- Dot-product similarity for nearest-neighbor retrieval
- Persistent vector storage with content-addressed collections (re-uploading the same file reuses its index instead of re-embedding or crashing)
- Source attribution / citation from retrieved chunks back to a page number
- Context window management and prompt engineering for grounded, hallucination-resistant responses
- Defensive error handling around file parsing, missing credentials, and external API failures

## Limitations and Future Improvements

- Naive fixed-length chunking (character count, not sentence/token-boundary aware) — a chunk can still split mid-sentence within a page
- Currently single-vector dense retrieval only — no hybrid search (BM25 + dense) or cross-encoder re-ranking yet
- Embedding model is lightweight (`all-MiniLM-L6-v2`); a larger model would likely improve retrieval accuracy at the cost of latency
- No query rewriting — rephrasing a question sometimes gets a better answer than the original phrasing
- No automated retrieval/answer-quality evaluation harness yet (no labeled eval set or tracked accuracy metric)
- Single PDF per session — no cross-document search yet
- No OCR — scanned/image-only PDFs are detected and rejected with a clear error rather than silently indexed as empty

