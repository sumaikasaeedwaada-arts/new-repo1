from fastapi import FastAPI
from AI_service.ai_service import ask_ai

app = FastAPI()

@app.get("/")
def home():
    return {"message": "HisabDo AI Service is running"}

@app.get("/ask")
def ask(question: str):
    answer = ask_ai(question)
    return {"question": question, "answer": answer}