import os
from dotenv import load_dotenv

from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector DB
vectordb = Chroma(
    persist_directory="chroma",
    embedding_function=embeddings
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# LLM - Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt Template (Updated for LangChain 0.3.x)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Use the context and chat history to answer."),
    ("system", "Context:\n{context}"),
    ("system", "Chat History:\n{chat_history}"),
    ("human", "{question}")
])

# Context builder
def build_context(inputs):
    docs = retriever.invoke(inputs["question"])
    return {"context": "\n\n".join(d.page_content for d in docs)}

# Main RAG Chain
rag_chain = (
    {
        "context": build_context,
        "question": lambda x: x["question"],
        "chat_history": lambda x: x.get("chat_history", []),
    }
    | prompt
    | llm
)

# Chat Memory
store = {}

def get_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Chain with Memory
rag_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)