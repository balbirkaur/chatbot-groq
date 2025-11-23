#!/usr/bin/env python3
"""
Interactive CLI for the LangChain Groq RAG system.
Ask questions about your PDF documents interactively.
"""

import os
from dotenv import load_dotenv
from chat import rag

load_dotenv()

def main():
    """Run the interactive chat interface."""
    print("=" * 60)
    print("🚀 LangChain Groq RAG - PDF Question Answering System")
    print("=" * 60)
    print("\n📚 Ask questions about your PDF documents.")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    # Check if Chroma DB exists
    if not os.path.exists("db") or not os.path.exists("db/chroma.sqlite3"):
        print("⚠️  Warning: Chroma database not found.")
        print("   Please run 'python ingest.py' first to ingest PDFs.\n")
    
    while True:
        try:
            question = input("💬 You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ["exit", "quit"]:
                print("\n👋 Goodbye!")
                break
            
            print("\n⏳ Thinking...\n")
            answer = rag(question)
            print(f"🤖 Assistant: {answer}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")

if __name__ == "__main__":
    main()
