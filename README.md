# RAG Pipeline - Document Question Answering System

A production-ready Retrieval-Augmented Generation (RAG) pipeline for querying documents using local LLMs.

## Features

- 📄 **Document Ingestion**: Process text files with optimized chunking
- 🔍 **Vector Search**: ChromaDB/FAISS for fast similarity search  
- 🤖 **Local LLM**: Uses Ollama (free, no API costs)
- ⚡ **Optimized**: Fast embeddings with sentence-transformers
- 🎨 **Web UI**: Simple Gradio interface

## Installation
```bash
# Clone repository
git clone <your-repo-url>
cd "rag pipeline01.py"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama
# Download from https://ollama.ai/download

# Pull LLM model
ollama pull phi3:mini
```

## Quick Start
```bash
# 1. Add your documents to DOCS/ folder (create it first)
mkdir DOCS
# Add your .txt files to DOCS/

# 2. Run ingestion pipeline
python ingestion_clean.py

# 3. Query your documents
python retrieval_ollama.py

# Or use web interface
python app.py  # Then open http://localhost:7860
```

## Configuration

- **Chunk size**: Edit `chunk_size=1000` in `ingestion_clean.py`
- **Model**: Change `model="phi3:mini"` in `retrieval_ollama.py`
- **Retrieval count**: Adjust `k=4` for more/fewer context chunks

## Supported Models

### Embedding Models (HuggingFace)
- `all-MiniLM-L6-v2` (default, fast)
- `BAAI/bge-small-en-v1.5` (better quality)
- `BAAI/bge-base-en-v1.5` (best quality)

### LLM Models (Ollama)
- `phi3:mini` (recommended, fast)
- `llama3.2` (better quality, slower)
- `tinyllama` (fastest)

## Project Structure
```
rag pipeline01.py/
├── ingestion_clean.py      # Document ingestion pipeline
├── retrieval_ollama.py     # Query/retrieval pipeline
├── app.py                  # Gradio web interface (optional)
├── requirements.txt        # Python dependencies
├── DOCS/                   # Your documents (not tracked)
├── chroma_db/             # Vector store (not tracked)
└── README.md
```

## Requirements

- Python 3.8+
- 4GB+ RAM
- Ollama installed locally

## License

MIT

## Contributing
