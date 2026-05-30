from fastapi import FastAPI
from pydantic import BaseModel, Field
from groq import Groq
from supabase import create_client, Client
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List

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
    history: List[ChatMessage] = Field(default_factory=list)


def save_message(session_id: str, role: str, content: str):
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role": role,
            "content": content
        }).execute()
    except Exception as error:
        print(f"Could not save message: {error}")


@app.get("/")
async def root():
    return {"message": "Backend is working!"}


@app.post("/chat")
async def chat(request: ChatRequest):
    save_message(
        session_id=request.session_id,
        role="user",
        content=request.message
    )

    response = supabase.table("documents").select("content").execute()

    context = "\n".join([
        item["content"] for item in response.data
    ]) if response.data else "No information available."

    conversation_history = "\n".join([
        f"{msg.role}: {msg.content}"
        for msg in request.history[-20:]
    ])

    prompt = f"""
You are Veronika's friendly receptionist for Veronika Wellness & Aesthetics in Leeds.

Your job is to answer like a real human receptionist, not like a database.

Very important:
- Do not repeat hello twice.
- Make everything easy to read.
- Use the conversation history to understand short replies.
- If the customer says "30 minutes?", "yes", "muscles", "how much?", or similar, use the previous messages to understand what they mean.
- Do not ask the customer to repeat information they already gave.
- Keep replies short and natural.
- Never list every service at once unless the customer asks for a full price list.
- If asked what services are offered, give the main categories first.
- Then ask what they are interested in.
- If the client asks which treatment would suit them, recommend only a suitable treatment based on what they said.
- Use the business information below only.
- Do not invent prices, times, treatments, availability, or policies.
- Stay focused only on Veronika Wellness & Aesthetics.
- Only answer questions about treatments, prices, booking requests, availability, location, opening hours, preparation, aftercare, and the customer's treatment needs.
- If the customer asks about an unrelated topic, politely redirect them back to Veronika's services.
- Relaxing massage is a style of normal massage, not a separate specialist treatment.
- Never ask for a deposit.
- Never take payment.
- Never confirm that an appointment is booked.
- After the client selects a treatment, ask naturally for their name, phone number, preferred date, and preferred time.
- Only once the treatment, name, phone number, preferred date, and preferred time have been collected, say that Veronika will be in touch shortly to confirm.
- Put the handover confirmation on a new line so the message is easy to read.
- If unsure, suggest calling 07386 396139.

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

    save_message(
        session_id=request.session_id,
        role="assistant",
        content=reply
    )

    return {"reply": reply}