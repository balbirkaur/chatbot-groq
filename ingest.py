import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_DIR = "pdfs"

def load_pdfs():
    documents = []
    for file in os.listdir(PDF_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_DIR, file))
            documents.extend(loader.load())
    return documents

def ingest():
    print("📄 Loading PDFs...")
    docs = load_pdfs()

    print("🔪 Chunking text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    print("🧠 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("💾 Storing in Chroma...")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="chroma")
    vectordb.persist()

    print("✅ Ingestion complete!")

if __name__ == "__main__":
    ingest()
