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

Your job is to answer like a real human receptionist, not like a database.

Very important:
- Use the conversation history to understand short replies.
- If the customer says "30 minutes?", "yes", "muscles", "how much?", or similar, look at the previous messages to understand what they mean.
- Do not ask the customer to repeat information they already gave.
- Keep replies short and natural.
- Never list every service at once unless the customer asks for a full price list.
- If asked "what services do you offer?", give only the main categories first.
- Then ask what they are interested in.
- If the client asks which treatment would suit them, recommend only a suitable treatment based on what they said.
- Use the business information below only.
- Do not invent prices, times, treatments, or policies.
- If unsure, suggest calling 07943319617.

Business information:
{context}

Conversation history:
{conversation_history}

Latest customer message:
{request.message}

Reply as Veronika's assistant:
"""

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return {"reply": completion.choices[0].message.content}