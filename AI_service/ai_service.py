from google import genai
from google.genai import types
import time

client = genai.Client()

SYSTEM_PROMPT = """
You are a helpful and friendly AI assistant.
Answer questions clearly and accurately.
Keep answers simple and easy to understand.
If you do not know something, say so honestly.
"""
MODEL = "gemini-3.5-flash-lite"


def ask_ai(question):
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        )
    )
    return response.text


def run_cli():
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        )
    )

    print("=" * 45)
    print("🤖 AI CHATBOT")
    print("=" * 45)
    print("Type 'exit' to stop.")
    print("Type 'clear' to start a new conversation.")
    print()

    while True:
        message = input("You: ").strip()

        if not message:
            print("Bot: Please enter a message.")
            print()
            continue

        if message.lower() == "exit":
            print("Bot: Goodbye! 👋")
            break

        if message.lower() == "clear":
            chat = client.chats.create(
                model=MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                )
            )
            print("Bot: Chat cleared! 🧹")
            print()
            continue

        print("Bot: Thinking... ⏳")

        for attempt in range(3):
            try:
                response = chat.send_message(message)
                print("Bot:", response.text)
                print()
                break
            except Exception as e:
                if "503" in str(e) and attempt < 2:
                    print("Bot: Server busy. Retrying... 🔄")
                    time.sleep(3)
                else:
                    print("Bot: Sorry, something went wrong. ❌")
                    print("Error:", e)
                    print()
                    break


if __name__ == "__main__":
    run_cli()
