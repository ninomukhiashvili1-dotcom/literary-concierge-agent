import os
import getpass
from langgraph.types import Command
from agent import build_graph

def main():
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = getpass.getpass("შეიყვანეთ GROQ_API_KEY: ")

    thread_id = "cli-session-1"
    graph = build_graph(thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    print("📚 Literary Time-Machine Concierge (გასასვლელად აკრიფეთ 'exit')")
    while True:
        user_input = input("\nთქვენ: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("ნახვამდის!")
            break
        if not user_input:
            continue

        try:
            res = graph.invoke({"messages": [("user", user_input)]}, config)
            state = graph.get_state(config)
            if state.next:
                prompt_text = state.tasks[0].interrupts[0].value
                ans = input(f"⚠️ {prompt_text} (yes/no): ").strip().lower()
                res = graph.invoke(Command(resume=(ans in ("yes", "y", "კი"))), config)
            print(f"\nასისტენტი: {res['messages'][-1].content}")
        except Exception as e:
            print(f"შეცდომა: {e}")

if __name__ == "__main__":
    main()
