from fastapi import FastAPI
from pydantic import BaseModel, Field
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
    session_id: str
    message: str
    history: Optional[List[ChatMessage]] = Field(default_factory=list)


@app.get("/")
async def root():
    return {"message": "Backend is working!"}


@app.post("/chat")
async def chat(request: ChatRequest):
    # Save the customer's latest message
    supabase.table("messages").insert({
        "session_id": request.session_id,
        "role": "user",
        "content": request.message
    }).execute()

    # Load business information
    documents_response = (
        supabase.table("documents")
        .select("content")
        .execute()
    )

    context = "\n".join([
        item["content"] for item in documents_response.data
    ]) if documents_response.data else "No information available."

    # Load the latest 20 messages for this visitor only
    messages_response = (
        supabase.table("messages")
        .select("role, content, created_at")
        .eq("session_id", request.session_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )

    # Reverse them so the oldest message appears first
    saved_messages = list(reversed(messages_response.data))

    conversation_history = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in saved_messages
    ])

    prompt = f"""
You are Veronika's friendly receptionist for Veronika beauty salon.

Your job is to answer like a real human receptionist in a natural human like response, not like a database.

Very important:
- Do not repeat hello twice.
- Make everything easy to read.
- Use the conversation history to understand short replies.
- If the customer says "30 minutes?", "yes", "muscles", "how much?", or similar, look at the previous messages to understand what they mean.
- Do not ask the customer to repeat information they already gave.
- Keep replies short and natural.
- Never list every service at once unless the customer asks for a full price list.
- If asked "what services do you offer?", give the main categories first.
- Then ask what they are interested in.
- If the client asks which treatment would suit them, recommend only a suitable treatment based on what they said.
- Use the business information below only.
- Do not invent prices, times, treatments, availability, or policies.
- Stay focused only on Veronika Wellness & Aesthetics.
- Only answer questions about treatments, prices, booking requests, availability, location, opening hours, preparation, aftercare, and the customer's treatment needs.
- If the customer asks about an unrelated topic, politely say you can only help with Veronika's services and ask whether they would like help choosing or requesting a treatment.
- Do not answer general knowledge questions, news questions, personal questions, or unrelated conversation.
- Relaxing massage is a style of normal massage, not a separate specialist treatment.
- Never ask for a deposit.
- Never take payment.
- Never confirm that an appointment is booked.
- After the client selects a treatment, ask for their name, phone number, preferred date, and preferred time.
- Only once the treatment, name, phone number, preferred date, and preferred time have been collected, say that Veronika will be in touch shortly to confirm.
- Put the handover confirmation on a new line so the message is easy to read.
- If unsure, suggest calling 07386 396617.

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

    reply = completion.choices[0].message.content

    # Save the assistant's reply
    supabase.table("messages").insert({
        "session_id": request.session_id,
        "role": "assistant",
        "content": reply
    }).execute()

    return {"reply": reply}