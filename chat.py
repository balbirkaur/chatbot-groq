import os
from dotenv import load_dotenv
from groq import Groq

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = "db"

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Embeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load Chroma
db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=emb
)

retriever = db.as_retriever(search_kwargs={"k": 4})


def rag(question):
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer the question ONLY using the PDF context below.
If the answer is not in the PDF, say "Not found in PDF".

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )



    return response.choices[0].message.content


print("🤖 Chatbot Ready! Ask anything about the PDF.\n")

while True:
    q = input("You: ")
    if q.lower() in ["exit", "quit"]:
        break
    print("\nBot:", rag(q), "\n")
