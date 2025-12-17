from chat import rag_with_memory

def main():
    session_id = "user123"
    print("💬 Groq RAG Chatbot with Memory")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = rag_with_memory.invoke(
            {"question": q, "chat_history": []},
            config={"configurable": {"session_id": session_id}}
        )

        print("Bot:", response.content)

if __name__ == "__main__":
    main()