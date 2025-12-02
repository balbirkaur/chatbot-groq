from chat import rag_with_memory

def main():
    session_id = "user_cli_test"
    chat_history = []

    print("🤖 Groq RAG Chatbot (CLI with Memory)")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        chat_history.append({"role": "user", "content": q})

        response = rag_with_memory.invoke(
            {"question": q, "chat_history": chat_history},
            config={"configurable": {"session_id": session_id}}
        )

        answer = response.content if hasattr(response, "content") else str(response)
        chat_history.append({"role": "assistant", "content": answer})

        print("Bot:", answer)


if __name__ == "__main__":
    main()
