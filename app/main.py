import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

CHAT_MODEL = "gemini-1.5-flash"

app = FastAPI()

class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatOut)
async def chat(in_: ChatIn):
    model = genai.GenerativeModel(CHAT_MODEL)
    resp = model.generate_content(in_.message)
    return ChatOut(answer=resp.text or "")

