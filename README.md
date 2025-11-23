# PDF Q&A Chatbot using Groq + LangChain
LangChain Groq RAG - PDF Question Answering System
A Retrieval-Augmented Generation (RAG) system that enables intelligent Q&A over PDF documents using LangChain, Groq's fast inference API, and Chroma vector database.

## Features
- **PDF Ingestion**: Automatically load and process PDF files from the `pdfs/` directory
- **Semantic Chunking**: Split documents intelligently using RecursiveCharacterTextSplitter
- **Vector Embeddings**: Generate embeddings using HuggingFace's sentence-transformers
- **Persistent Storage**: Store embeddings in Chroma vector database for fast retrieval
- **Fast Inference**: Leverage Groq's API with Llama 3.1 8B for rapid response generation
- **Context-Aware Answers**: Answers are grounded exclusively in your PDF content

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set up environment:**
   echo "GROQ_API_KEY=your_key_here" > .env
   
4. **Full Installation Command (COPY–PASTE & RUN)**
  ```bash
pip install langchain langchain-core langchain-community chromadb sentence-transformers pypdf python-dotenv groq fastapi uvicorn[standard] --upgrade
   ```
| Package                   | Purpose                   |
| ------------------------- | ------------------------- |
| **langchain**             | Core RAG tools            |
| **langchain-core**        | Base LangChain components |
| **langchain-community**   | Chroma + PDF loaders      |
| **chromadb**              | Vector database           |
| **sentence-transformers** | Embeddings                |
| **pypdf**                 | PDF parsing               |
| **python-dotenv**         | Load API keys             |
| **groq**                  | Groq LLM client           |
| **fastapi**               | Optional API server       |
| **uvicorn**               | FastAPI server runner     |

**Optional: Create Virtual Environment (Recommended)**
**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```
**Mac**
```bash
python3 -m venv venv
source venv/bin/activate
```
Then run the install command:
```bash
pip install langchain langchain-core langchain-community chromadb sentence-transformers pypdf python-dotenv groq fastapi uvicorn[standard] --upgrade
```

5. **Ingest PDFs:**
   python ingest.py
6. **Run the RAG Chatbot:**
   python app.py


## Requirements

Python 3.9+
LangChain ecosystem
HuggingFace embeddings
Groq API key
Chroma vector database
   
   
