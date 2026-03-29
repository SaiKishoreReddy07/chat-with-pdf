# 📄 Chat with any PDF — RAG Application

A Retrieval Augmented Generation (RAG) application that lets you upload any PDF and ask questions about it in natural language.

Built from scratch without frameworks like LangChain to demonstrate a clear understanding of how RAG pipelines work under the hood.

## How it Works

1. **Upload** any PDF via the sidebar
2. The app **chunks** the document into 500 character pieces
3. Each chunk is **embedded** into a vector using sentence-transformers
4. Chunks are **stored** in a ChromaDB vector database
5. When you ask a question, it gets **embedded** and the most similar chunks are retrieved
6. Retrieved chunks are passed as **context** to an LLM which generates a grounded answer

## Tech Stack

- **Frontend** — Streamlit
- **Embeddings** — sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database** — ChromaDB
- **LLM** — LLaMA 3.3 70B via Groq API

## Getting Started

### 1. Clone the repository
git clone https://github.com/yourusername/chat-with-pdf.git
cd chat-with-pdf

### 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Set up your API key
Create a .env file in the root folder:
GROQ_API_KEY=your_groq_api_key_here

Get a free Groq API key at console.groq.com

### 5. Run the app
streamlit run streamlit_app.py

## Project Structure

- streamlit_app.py — Streamlit UI and RAG pipeline
- app.py — CLI version of the RAG pipeline
- requirements.txt — Python dependencies

## Key Concepts Demonstrated

- Chunking strategies for document processing
- Semantic search using vector embeddings
- Dot product similarity for nearest neighbor retrieval
- Context window management
- Prompt engineering for grounded responses
- Persistent vector storage with ChromaDB

## Limitations and Future Improvements

- Currently uses naive RAG — could be improved with hybrid search and re-ranking
- Embedding model is lightweight — larger models would improve retrieval accuracy
- No query rewriting — rephrasing questions sometimes needed for best results
- Could add source citations showing which page the answer came from