from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from datetime import datetime, timezone
import json
import os

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


def save_conversation_overview(session_id: str, lead_data: dict):
    try:
        existing_response = (
            supabase.table("conversations")
            .select("*")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )

        existing = (
            existing_response.data[0]
            if existing_response.data
            else {}
        )

        fields = [
            "name",
            "phone",
            "treatment",
            "duration",
            "preferred_date",
            "preferred_time",
            "notes",
            "status",
            "summary"
        ]

        payload = {
            "session_id": session_id,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

        for field in fields:
            new_value = lead_data.get(field)

            if new_value not in [None, ""]:
                payload[field] = new_value
            else:
                payload[field] = existing.get(field)

        if not payload.get("status"):
            payload["status"] = "new_chat"

        supabase.table("conversations").upsert(
            payload,
            on_conflict="session_id"
        ).execute()

    except Exception as error:
        print(f"Could not save conversation overview: {error}")


def extract_lead_data(conversation_history: str) -> dict:
    try:
        extraction_prompt = f"""
Read the customer conversation below and extract useful lead details.

Only save information the customer has actually provided.
Do not treat suggestions made by the assistant as confirmed customer details.
Do not invent missing information.

Return valid JSON only, using exactly these fields:
{{
  "name": null,
  "phone": null,
  "treatment": null,
  "duration": null,
  "preferred_date": null,
  "preferred_time": null,
  "notes": null,
  "status": "new_chat",
  "summary": null
}}

Status rules:
- Use "new_chat" if the customer has not shown a clear interest yet.
- Use "interested" if the customer is asking about a treatment or price.
- Use "details_incomplete" if they want an appointment but important details are missing.
- Use "booking_request_complete" only if treatment, duration, name, phone number, preferred date, and preferred time are all present.

Keep the summary short and useful for the business owner.

Conversation:
{conversation_history}
"""

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": extraction_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        return json.loads(completion.choices[0].message.content)

    except Exception as error:
        print(f"Could not extract lead details: {error}")
        return {}


def update_conversation_overview(
    session_id: str,
    conversation_history: str
):
    lead_data = extract_lead_data(conversation_history)

    if lead_data:
        save_conversation_overview(
            session_id=session_id,
            lead_data=lead_data
        )


def load_chatbot_settings() -> dict:
    defaults = {
        "tone": "friendly",
        "reply_length": "short",
        "emoji_style": "light",
        "personality_notes": ""
    }

    try:
        response = (
            supabase.table("chatbot_settings")
            .select("tone, reply_length, emoji_style, personality_notes")
            .eq("business_slug", "veronika")
            .limit(1)
            .execute()
        )

        if not response.data:
            return defaults

        saved = response.data[0]

        tone = saved.get("tone")
        reply_length = saved.get("reply_length")
        emoji_style = saved.get("emoji_style")
        personality_notes = saved.get("personality_notes") or ""

        if tone in ["friendly", "professional", "relaxed", "luxury"]:
            defaults["tone"] = tone

        if reply_length in ["short", "normal"]:
            defaults["reply_length"] = reply_length

        if emoji_style in ["none", "light", "friendly"]:
            defaults["emoji_style"] = emoji_style

        defaults["personality_notes"] = personality_notes[:600]

    except Exception as error:
        print(f"Could not load chatbot settings: {error}")

    return defaults


def build_style_instructions(settings: dict) -> str:
    tone_instructions = {
        "friendly": "Sound warm, approachable, and helpful.",
        "professional": "Sound polished, professional, and clear.",
        "relaxed": "Sound calm, relaxed, and conversational.",
        "luxury": "Sound refined, reassuring, and premium without being over-the-top."
    }

    reply_length_instructions = {
        "short": "Keep replies concise. Usually use 1 to 3 short sentences.",
        "normal": "Use a natural amount of detail. Avoid long walls of text."
    }

    emoji_instructions = {
        "none": "Do not use emojis.",
        "light": "Use emojis rarely and only when they feel natural.",
        "friendly": "Use 1 to 3 friendly emojis in most replies when it feels natural. Keep them relevant and do not overdo it."
    }

    personality_notes = settings.get("personality_notes", "").strip()

    return f"""
Owner-selected style settings:
- Tone: {tone_instructions[settings["tone"]]}
- Reply length: {reply_length_instructions[settings["reply_length"]]}
- Emoji use: {emoji_instructions[settings["emoji_style"]]}
- Extra personality note: {personality_notes if personality_notes else "No extra note."}

Important:
- These style settings affect wording and personality only.
- They must never override the locked business, booking, privacy, or safety rules below.
"""


@app.get("/")
async def root():
    return {"message": "Backend is working!"}


@app.post("/chat")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
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

    settings = load_chatbot_settings()
    style_instructions = build_style_instructions(settings)

    prompt = f"""
You are Veronika's receptionist for Veronika Wellness & Aesthetics in Leeds.

Your job is to answer like a real human receptionist, not like a database.

{style_instructions}

Locked rules:
- Do not repeat hello twice.
- Make everything easy to read.
- Use the conversation history to understand short replies.
- If the customer says "30 minutes?", "yes", "muscles", "how much?", or similar, use the previous messages to understand what they mean.
- Do not ask the customer to repeat information they already gave.
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
- Only once the treatment, duration, name, phone number, preferred date, and preferred time have been collected, say that Veronika will be in touch shortly to confirm.
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
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    reply = completion.choices[0].message.content

    save_message(
        session_id=request.session_id,
        role="assistant",
        content=reply
    )

    lead_history = f"""
{conversation_history}
assistant: {reply}
"""

    background_tasks.add_task(
        update_conversation_overview,
        request.session_id,
        lead_history
    )

    return {"reply": reply}
