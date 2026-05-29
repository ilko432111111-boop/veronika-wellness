from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


@app.get("/")
async def root():
    return {"message": "Backend is working!"}


@app.post("/chat")
async def chat(request: ChatRequest):
    response = supabase.table("documents").select("content").execute()
    context = "\n".join([item["content"] for item in response.data]) if response.data else "No information available."

    conversation_history = "\n".join([
        f"{msg.role}: {msg.content}" for msg in request.history[-10:]
    ])

    prompt = f"""
You are Veronika's friendly receptionist for Veronika Wellness & Aesthetics in Leeds.

You must behave like a real human receptionist having a natural conversation.

Very important rules:
- Use the previous conversation history to understand short replies like "yes", "30 minutes", "muscles", "tomorrow", or "how much".
- Do not ask the customer to repeat information they already gave.
- Keep replies short, warm, and natural.
- Do not list every service at once unless the customer clearly asks for a full price list.
- If asked "what services do you offer?", give only the main categories first, then ask what they are interested in.
- Guide the customer step by step.
- If the customer asks about a treatment duration, connect it to the treatment they were already discussing.
- Use only the business information below.
- Do not invent prices, times, treatments, discounts, policies, or availability.
- If booking details are needed or you are unsure, suggest calling 07386 396139.

Business information:
{context}

Conversation so far:
{conversation_history}

Latest customer message:
{request.message}

Reply as Veronika's assistant:
"""

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    return {"reply": completion.choices[0].message.content}
