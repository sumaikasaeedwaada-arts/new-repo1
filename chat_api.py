from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
import time
from datetime import datetime

app = FastAPI()

client = genai.Client()

# -----------------------------
# Conversation / Message Storage
# -----------------------------

conversations = {}

# -----------------------------
# Knowledge Base
# -----------------------------

knowledge_base = [
    {
        "id": 1,
        "title": "Refund Policy",
        "content": "Customers can request a refund within 7 days of purchase."
    },
    {
        "id": 2,
        "title": "Business Hours",
        "content": "HisabDo business support is available Monday to Friday, 9 AM to 6 PM."
    },
    {
        "id": 3,
        "title": "Quotation Process",
        "content": "Customers can request a quotation. The sales team should follow up with the customer after sending the quotation."
    }
]


# -----------------------------
# Request Models
# -----------------------------

class ChatRequest(BaseModel):
    conversation_id: str
    message: str

    user: str
    business: str
    customer: str
    task: str
    notes: str


class KnowledgeRequest(BaseModel):
    title: str
    content: str


# -----------------------------
# Home
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "HisabDo AI Chat API is running"
    }


# -----------------------------
# Knowledge Base
# -----------------------------

@app.post("/knowledge")
def add_knowledge(request: KnowledgeRequest):

    new_item = {
        "id": len(knowledge_base) + 1,
        "title": request.title,
        "content": request.content
    }

    knowledge_base.append(new_item)

    return {
        "message": "Knowledge added successfully",
        "knowledge": new_item
    }


@app.get("/knowledge")
def get_knowledge():

    return {
        "knowledge_base": knowledge_base
    }


# -----------------------------
# Chat + History + RAG
# -----------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    # Create conversation if it doesn't exist
    if request.conversation_id not in conversations:
        conversations[request.conversation_id] = []

    # Save user message
    conversations[request.conversation_id].append({
        "role": "user",
        "message": request.message,
        "time": datetime.now().isoformat()
    })

    # -----------------------------
    # Basic Knowledge Retrieval
    # -----------------------------

    relevant_knowledge = []

    question_words = request.message.lower().split()

    for item in knowledge_base:

        searchable_text = (
            item["title"] + " " + item["content"]
        ).lower()

        for word in question_words:

            if len(word) > 3 and word in searchable_text:
                relevant_knowledge.append(item)
                break

    # Remove duplicates
    unique_knowledge = []

    for item in relevant_knowledge:

        if item not in unique_knowledge:
            unique_knowledge.append(item)

    # -----------------------------
    # Conversation History
    # -----------------------------

    history = conversations[request.conversation_id]

    history_text = ""

    for message in history[-10:]:

        history_text += (
            f'{message["role"]}: '
            f'{message["message"]}\n'
        )

    # -----------------------------
    # Knowledge Context
    # -----------------------------

    knowledge_text = ""

    for item in unique_knowledge:

        knowledge_text += (
            f'Title: {item["title"]}\n'
            f'Information: {item["content"]}\n\n'
        )

    if not knowledge_text:
        knowledge_text = "No relevant knowledge was found."

    # -----------------------------
    # Business Context
    # -----------------------------

    business_context = f"""
User: {request.user}
Business: {request.business}
Customer: {request.customer}
Task: {request.task}
Notes: {request.notes}
"""

    # -----------------------------
    # RAG Prompt
    # -----------------------------

    prompt = f"""
You are HisabDo AI, a professional business assistant.

Answer the user's question using the available business
context, conversation history and retrieved knowledge.

IMPORTANT:
- Do not invent business information.
- Use the retrieved knowledge when it is relevant.
- If information is not available, clearly say so.
- Keep the answer clear and useful.

BUSINESS CONTEXT:
{business_context}

CONVERSATION HISTORY:
{history_text}

RETRIEVED KNOWLEDGE:
{knowledge_text}

CURRENT USER QUESTION:
{request.message}
"""

    # -----------------------------
    # Gemini
    # -----------------------------

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="""
You are HisabDo AI.
You are a helpful professional business assistant.
Ground your answers in the provided business context,
conversation history and knowledge base.
Never make up information.
"""
                )
            )

            # Save AI response
            conversations[request.conversation_id].append({
                "role": "assistant",
                "message": response.text,
                "time": datetime.now().isoformat()
            })

            print("QUESTION:", request.message)
            print("AI RESPONSE:", response.text)

            return {
                "conversation_id": request.conversation_id,
                "question": request.message,
                "retrieved_knowledge": unique_knowledge,
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


# -----------------------------
# Conversation History
# -----------------------------

@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):

    return {
        "conversation_id": conversation_id,
        "messages": conversations.get(
            conversation_id,
            []
        )
    }