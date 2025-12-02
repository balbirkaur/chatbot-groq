import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_DIR = "pdfs"
CHROMA_DIR = "chroma"


def load_pdfs():
    if not os.path.exists(PDF_DIR) or not os.listdir(PDF_DIR):
        raise ValueError("No PDFs found in the 'pdfs/' folder. Please upload and try again.")

    documents = []
    for file in os.listdir(PDF_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_DIR, file))
            documents.extend(loader.load())
    return documents


def ingest():
    print("📄 Loading PDFs...")
    docs = load_pdfs()

    print("🔪 Splitting text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    print("🧠 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("💾 Saving to Chroma DB...")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()

    print("✨ Ingestion complete!")


if __name__ == "__main__":
    ingest()
