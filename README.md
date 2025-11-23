# chatbot-groq
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
2. **Set up environment:**
   echo "GROQ_API_KEY=your_key_here" > .env
3. **Ingest PDFs:**
   python ingest.py

## Requirements

Python 3.9+
LangChain ecosystem
HuggingFace embeddings
Groq API key
Chroma vector database
   
   
