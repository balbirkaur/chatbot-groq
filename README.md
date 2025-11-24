# LangChain Groq RAG - PDF Question Answering System

A Retrieval-Augmented Generation (RAG) system that enables intelligent Q&A over PDF documents using LangChain, Groq's fast inference API, and Chroma vector database.

## Features

- **PDF Ingestion**: Automatically load and process PDF files from the `pdfs/` directory
- **Semantic Chunking**: Split documents intelligently using RecursiveCharacterTextSplitter
- **Vector Embeddings**: Generate embeddings using HuggingFace's sentence-transformers
- **Persistent Storage**: Store embeddings in Chroma vector database for fast retrieval
- **Fast Inference**: Leverage Groq's API with Llama 3.1 8B for rapid response generation
- **Context-Aware Answers**: Answers are grounded exclusively in your PDF content

## Prerequisites

- Python 3.9+
- Groq API key (get one at [console.groq.com](https://console.groq.com))

## Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd langchain_groq
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   ```bash
   pip freeze | findstr /R /C:"langchain" /C:"chromadb" /C:"groq" /C:"python-dotenv" /C:"sentence" /C:"pypdf" > requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

## Quick Start

### 1. Ingest PDF Documents

Place your PDF files in the `pdfs/` directory, then run:

```bash
python ingest.py
```

This will:

- Load all PDFs from `pdfs/`
- Split documents into semantic chunks
- Generate embeddings
- Store in Chroma database (`db/`)

### 2. Ask Questions

**Interactive mode:**

```bash
python main.py
```

**Programmatic usage:**

```python
from chat import rag

answer = rag("What is the main topic of the document?")
print(answer)
```

## Project Structure

```
langchain_groq/
├── chat.py           # RAG chat function and Groq integration
├── ingest.py         # PDF ingestion and Chroma DB setup
├── main.py           # Interactive CLI entry point
├── requirements.txt  # Project dependencies
├── .env.example      # Environment variables template
├── .gitignore        # Git ignore rules
├── pdfs/             # Place your PDF files here
└── db/               # Chroma vector database (auto-created)
```

## Configuration

Edit `.env` to customize:

```env
GROQ_API_KEY=your_api_key_here
```

## Usage Examples

### Example 1: Basic Question

```python
from chat import rag

result = rag("What are the key points?")
print(result)
```

### Example 2: Specific Query

```python
from chat import rag

result = rag("How does feature X work?")
print(result)
```

## How It Works

1. **Ingestion Phase** (`ingest.py`):

   - Loads PDFs using PyPDFLoader
   - Splits content into 1000-char chunks with 200-char overlap
   - Generates embeddings using sentence-transformers
   - Stores in Chroma vector database

2. **Query Phase** (`chat.py`):
   - Retrieves top 4 relevant chunks using similarity search
   - Constructs a prompt with context from PDFs
   - Sends to Groq API (Llama 3.1 8B)
   - Returns answer grounded in PDF content

## Dependencies

- **langchain** - LLM framework
- **langchain-community** - Community integrations
- **langchain-chroma** - Chroma vector store integration
- **langchain-huggingface** - HuggingFace embeddings
- **chromadb** - Vector database
- **sentence-transformers** - Embedding model
- **pypdf** - PDF parsing
- **groq** - Groq API client
- **python-dotenv** - Environment variable management

## Troubleshooting

### "Not found in PDF"

The answer isn't in your PDF documents. Try:

- Rephrasing the question
- Ensuring PDFs are properly ingested
- Checking retriever with `k=4` parameter

### Slow ingestion

- Large PDFs may take time. Be patient.
- Chunk size (1000) can be adjusted in `ingest.py`

### API errors

- Verify `GROQ_API_KEY` is set correctly in `.env`
- Check internet connection
- Ensure API key has sufficient credits

## License

MIT

## Contributing

Contributions welcome! Please feel free to submit pull requests.
