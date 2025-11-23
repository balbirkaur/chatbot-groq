import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PDF_DIR = "pdfs"
CHROMA_DIR = "db"

print("📄 Loading PDF...")
docs = []

for file in os.listdir(PDF_DIR):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(PDF_DIR, file))
        docs.extend(loader.load())

print(f"Loaded {len(docs)} pages.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

print(f"Split into {len(chunks)} chunks.")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

db.add_documents(chunks)

print("✅ Ingestion complete! Chroma DB saved.")
