# LangChain Groq RAG - PDF Question Answering System

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using the content of your PDF documents. Combines LangChain, Groq's fast LLM API, and Chroma vector database for efficient, context-aware responses.

## Features

- **PDF Ingestion:** Load and process PDFs from the `pdfs/` folder
- **Semantic Chunking:** Split documents for better retrieval
- **Embeddings:** Use HuggingFace sentence-transformers
- **Vector Database:** Store and search chunks in Chroma
- **LLM Integration:** Query Groq's Llama 3.1 8B model
- **Environment Management:** Uses `.env` for secrets

## Installation

1. **Clone the repository:**
   ```powershell
   git clone git@github.com:balbirkaur/chatbot-groq.git
   cd chatbot-groq
   ```
2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Add your Groq API key:
     ```env
     GROQ_API_KEY=your_groq_api_key_here
     ```

## Usage

1. **Ingest PDFs:**
   - Place your PDF files in the `pdfs/` directory
   - Run:
     ```powershell
     python ingest.py
     ```
2. **Ask questions (CLI):**

   ```powershell
   python main.py
   ```

   - Type your question and get answers based on your PDFs

3. **Programmatic usage:**
   ```python
   from chat import rag
   answer = rag("What is the main topic?")
   print(answer)
   ```

## Project Structure

```
chatbot-groq/
├── chat.py         # RAG chat logic
├── ingest.py       # PDF ingestion and embedding
├── main.py         # Interactive CLI
├── requirements.txt
├── .env.example
├── .gitignore
├── pdfs/           # Place PDFs here
├── db/             # Chroma DB files
```

## Troubleshooting

- If you see "Not found in PDF", the answer isn't in your documents.
- If you get API errors, check your `.env` and Groq API key.
- For merge/push issues, ensure your local branch is up to date with remote.

## License

MIT

## Contributing

Pull requests welcome!

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
