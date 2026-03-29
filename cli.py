from app.agent.graph import app

def run_cli():
    print("=" * 60)
    print(" Agent-Based Energy Management Assistant (CLI Mode) ")
    print("=" * 60)
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break
        if not user_input: continue

        prompt = user_input
        if user_input.lower().startswith("import "):
            filename = user_input.split(" ", 1)[1].strip()
            prompt = f"Import the data from {filename}"

        initial_state = {"user_request": prompt, "messages": []}
        try:
            result = app.invoke(initial_state)
            print(f"Assistant: {result.get('drafted_response', 'Error')}")
        except Exception as e:
            print(f"\n[Execution Error]: {e}")

if __name__ == "__main__":
    run_cli()