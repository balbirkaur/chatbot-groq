import os
import streamlit as st
from dotenv import load_dotenv

from chat import rag_with_memory   # uses your existing RAG + memory chain
import ingest                      # your ingest.py

load_dotenv()

st.set_page_config(
    page_title="Groq PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ---- SESSION STATE ----
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_streamlit"  # could randomize if you want

if "messages" not in st.session_state:
    st.session_state.messages = []  # for UI only


# ---- SIDEBAR ----
st.sidebar.title("📄 PDF Settings")

st.sidebar.markdown("**Upload PDFs to chat with them**")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    pdf_dir = "pdfs"
    os.makedirs(pdf_dir, exist_ok=True)

    for file in uploaded_files:
        save_path = os.path.join(pdf_dir, file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("✅ PDFs uploaded. Click 'Rebuild Vector DB' below.")

if st.sidebar.button("🔁 Rebuild Vector DB (Ingest PDFs)"):
    with st.spinner("Ingesting PDFs and rebuilding Chroma…"):
        try:
            ingest.ingest()
            st.sidebar.success("✅ Ingestion complete!")
        except Exception as e:
            st.sidebar.error(f"❌ Ingestion failed: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** `llama-3.3-70b-versatile` (Groq)")
st.sidebar.markdown("Make sure your `.env` has `GROQ_API_KEY` set.")


# ---- MAIN CHAT UI ----
st.title("🤖 Groq PDF RAG Chatbot (with Memory)")

# Display previous messages
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)


# Chat input
user_input = st.chat_input("Ask a question about your PDFs…")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call RAG chain with memory
    with st.chat_message("assistant"):
        with st.spinner("Thinking with Groq…"):
            try:
                response = rag_with_memory.invoke(
                    {"question": user_input, "chat_history": []},
                    config={"configurable": {"session_id": st.session_state.session_id}},
                )

                answer = response.content if hasattr(response, "content") else str(response)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"❌ Error: {e}")
