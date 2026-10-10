from google import genai

client = genai.Client()

print("AI Chatbot started!")

while True:
    message = input("You: ")

    if message.lower() == "exit":
        print("Bot: Bye!")
        break

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=message
        )
        print("Bot:", response.text)

    except Exception as e:
        print("Server/API error:", e)