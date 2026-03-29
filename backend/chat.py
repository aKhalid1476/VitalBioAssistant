"""
Interactive CLI for the VitalBio bloodborne pathogen RAG assistant.
Usage: python chat.py
"""

from dotenv import load_dotenv
load_dotenv()

from rag import rag_chain

SESSION_ID = "cli-session"

print("VitalBio Assistant — OSHA 29 CFR § 1910.1030")
print("Type your question and press Enter. Type 'exit' to quit.\n")

while True:
    try:
        question = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        break

    if not question:
        continue
    if question.lower() in ("exit", "quit"):
        print("Goodbye.")
        break

    result = rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": SESSION_ID}},
    )

    print(f"\nAssistant: {result['answer']}\n")
