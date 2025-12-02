import os
import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "chroma"

# API key: Streamlit Secrets (Cloud) → .env fallback (Local)
groq_api_key = None
try:
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not groq_api_key:
    groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("❌ Missing GROQ_API_KEY. Add it to .env or Streamlit Secrets.")

# Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
)

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Basic RAG Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use retrieved document context. Include short citations."),
    ("human", "{question}")
])

# RAG pipeline
rag_with_memory = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
