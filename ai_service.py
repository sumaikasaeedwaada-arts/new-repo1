from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
import time

app = FastAPI()

client = genai.Client()


# -----------------------------
# Request structure
# -----------------------------

class ChatRequest(BaseModel):
    message: str

    user: str
    business: str
    customer: str
    task: str
    notes: str


# -----------------------------
# Home
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "HisabDo AI Chat API is running"
    }


# -----------------------------
# AI Chat
# -----------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    business_context = f"""
USER:
{request.user}

BUSINESS:
{request.business}

CUSTOMER:
{request.customer}

TASK:
{request.task}

NOTES:
{request.notes}
"""

    prompt = f"""
You are HisabDo AI, a professional business assistant.

Use the business context provided below to answer the user's question.

IMPORTANT RULES:
1. Give answers based on the provided business context.
2. Do not invent customer, task, or business information.
3. If the required information is not available in the context, clearly say that the information is not available.
4. Keep the answer clear and useful.
5. Use the user's business data to give a grounded answer.

BUSINESS CONTEXT:
{business_context}

USER QUESTION:
{request.message}
"""

    for attempt in range(3):
        try:

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="""
You are HisabDo AI.
You are a helpful and professional business assistant.
Always ground your answers in the business context provided by the user.
Never make up business data.
"""
                )
            )

            print("QUESTION:", request.message)
            print("BUSINESS CONTEXT:", business_context)
            print("AI RESPONSE:", response.text)

            return {
                "question": request.message,
                "context": {
                    "user": request.user,
                    "business": request.business,
                    "customer": request.customer,
                    "task": request.task,
                    "notes": request.notes
                },
                "response": response.text
            }

        except Exception as e:

            if attempt < 2:
                time.sleep(3)

            else:
                return {
                    "error": "Gemini temporarily unavailable",
                    "details": str(e)
                }