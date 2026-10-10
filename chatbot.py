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

    # Empty input validation
    if not message:
        print("Bot: Please enter a message.")
        print()
        continue

    # Exit
    if message.lower() == "exit":
        print("Bot: Goodbye! 👋")
        break

    # Clear conversation
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

    # Retry system for temporary 503 errors
    for attempt in range(3):

        try:
            response = chat.send_message(message)

            print("Bot:", response.text)
            print()
            break

        except Exception as e:

            error_text = str(e)

            if "503" in error_text and attempt < 2:
                print("Bot: Server busy. Retrying... 🔄")
                time.sleep(3)
            else:
                print("Bot: Sorry, something went wrong. ❌")
                print("Error:", e)
                print()
                break