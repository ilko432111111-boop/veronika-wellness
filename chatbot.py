from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from groq import Groq
from supabase import create_client, Client
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from datetime import datetime, timezone, timedelta
from html import escape
from urllib import request as urllib_request
from urllib import error as urllib_error
import json
import os
import re
from zoneinfo import ZoneInfo

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

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
NOTIFICATION_EMAIL = os.getenv("NOTIFICATION_EMAIL")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")
DASHBOARD_URL = "https://veronika-wellness.onrender.com/admin.html"
BUSINESS_TIMEZONE = ZoneInfo("Europe/London")

REQUIRED_BOOKING_FIELDS = [
    "treatment",
    "duration",
    "name",
    "phone",
    "preferred_date",
    "preferred_time",
]


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


def get_existing_conversation(session_id: str) -> dict:
    try:
        response = (
            supabase.table("conversations")
            .select("*")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )

        if response.data:
            return response.data[0]

    except Exception as error:
        print(f"Could not load existing conversation: {error}")

    return {}


def merge_lead_data(existing: dict, extracted: dict) -> dict:
    fields = [
        "name",
        "phone",
        "treatment",
        "duration",
        "preferred_date",
        "preferred_time",
        "notes",
        "status",
        "summary",
    ]

    merged = {}

    for field in fields:
        new_value = extracted.get(field)
        old_value = existing.get(field)

        if new_value not in [None, ""]:
            merged[field] = new_value
        else:
            merged[field] = old_value

    complete = all(merged.get(field) not in [None, ""] for field in REQUIRED_BOOKING_FIELDS)

    if complete:
        merged["status"] = "booking_request_complete"
    elif not merged.get("status"):
        merged["status"] = "new_chat"

    return merged


def normalise_preferred_date(value: str | None) -> str | None:
    if value in [None, ""]:
        return value

    cleaned = str(value).strip()
    lower_value = cleaned.lower()
    today = datetime.now(BUSINESS_TIMEZONE).date()

    if lower_value == "today":
        return today.isoformat()

    if lower_value == "tomorrow":
        return (today + timedelta(days=1)).isoformat()

    return cleaned


def apply_simple_date_time_fallback(
    lead_data: dict,
    conversation_history: str
) -> dict:
    """
    Preserve obvious customer date/time preferences even if the AI extractor
    misses them. Relative dates such as today and tomorrow are converted into
    real calendar dates so the dashboard remains clear later.
    """
    result = dict(lead_data)
    lower_history = conversation_history.lower()
    today = datetime.now(BUSINESS_TIMEZONE).date()

    result["preferred_date"] = normalise_preferred_date(
        result.get("preferred_date")
    )

    if result.get("preferred_date") in [None, ""]:
        if re.search(r"\btoday\b", lower_history):
            result["preferred_date"] = today.isoformat()
        elif re.search(r"\btomorrow\b", lower_history):
            result["preferred_date"] = (
                today + timedelta(days=1)
            ).isoformat()

    if result.get("preferred_time") in [None, ""]:
        time_match = re.search(
            r"\b(?:at\s*)?([01]?\d|2[0-3])[:.]([0-5]\d)\b",
            lower_history
        )

        if time_match:
            result["preferred_time"] = (
                f"{int(time_match.group(1)):02d}:{time_match.group(2)}"
            )
        else:
            am_pm_match = re.search(
                r"\b(?:at\s*)?(1[0-2]|0?[1-9])(?:[:.]([0-5]\d))?\s*(am|pm)\b",
                lower_history
            )

            if am_pm_match:
                hour = int(am_pm_match.group(1))
                minute = am_pm_match.group(2) or "00"
                suffix = am_pm_match.group(3)

                if suffix == "pm" and hour != 12:
                    hour += 12
                elif suffix == "am" and hour == 12:
                    hour = 0

                result["preferred_time"] = f"{hour:02d}:{minute}"

    return result


def save_conversation_overview(session_id: str, lead_data: dict):
    try:
        payload = {
            "session_id": session_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **lead_data,
        }

        supabase.table("conversations").upsert(
            payload,
            on_conflict="session_id"
        ).execute()

    except Exception as error:
        print(f"Could not save conversation overview: {error}")


def extract_lead_data(conversation_history: str) -> dict:
    try:
        current_leeds_date = datetime.now(BUSINESS_TIMEZONE).date().isoformat()

        extraction_prompt = f"""
Read the customer conversation below and extract useful lead details.

Only save information the customer has actually provided or clearly confirmed.
Do not treat suggestions made by the assistant as confirmed customer details.
Do not invent missing information.
Treat clear relative dates such as "today" and "tomorrow" as valid preferred dates.
Convert relative dates into an ISO calendar date in YYYY-MM-DD format using the current Leeds date.

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

Current Leeds date:
{current_leeds_date}

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


def format_confirmed_details(lead_data: dict) -> str:
    lines = []

    for field in REQUIRED_BOOKING_FIELDS:
        value = lead_data.get(field)

        if value not in [None, ""]:
            label = field.replace("_", " ").title()
            lines.append(f"- {label}: {value}")

    return "\n".join(lines) if lines else "- No confirmed booking details yet."


def send_booking_notification(session_id: str):
    if not RESEND_API_KEY or not NOTIFICATION_EMAIL:
        print("Email notification skipped: missing Resend environment variables.")
        return

    try:
        lead_response = (
            supabase.table("conversations")
            .select(
                "session_id, name, phone, treatment, duration, "
                "preferred_date, preferred_time, notes, summary, "
                "status, notification_sent_at"
            )
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )

        if not lead_response.data:
            return

        lead = lead_response.data[0]

        if lead.get("status") != "booking_request_complete":
            return

        if lead.get("notification_sent_at"):
            return

        sent_at = datetime.now(timezone.utc).isoformat()

        claim_response = (
            supabase.table("conversations")
            .update({"notification_sent_at": sent_at})
            .eq("session_id", session_id)
            .is_("notification_sent_at", "null")
            .execute()
        )

        if not claim_response.data:
            return

        name = escape(str(lead.get("name") or "Not provided"))
        phone = escape(str(lead.get("phone") or "Not provided"))
        treatment = escape(str(lead.get("treatment") or "Not provided"))
        duration = escape(str(lead.get("duration") or "Not provided"))
        preferred_date = escape(str(lead.get("preferred_date") or "Not provided"))
        preferred_time = escape(str(lead.get("preferred_time") or "Not provided"))
        notes = escape(str(lead.get("notes") or "None"))
        summary = escape(str(lead.get("summary") or "No summary available"))

        email_payload = {
            "from": RESEND_FROM_EMAIL,
            "to": [NOTIFICATION_EMAIL],
            "subject": f"New booking request: {name}",
            "html": f"""
                <h2>New booking request</h2>
                <p><strong>Name:</strong> {name}</p>
                <p><strong>Phone:</strong> {phone}</p>
                <p><strong>Treatment:</strong> {treatment}</p>
                <p><strong>Duration:</strong> {duration}</p>
                <p><strong>Preferred date:</strong> {preferred_date}</p>
                <p><strong>Preferred time:</strong> {preferred_time}</p>
                <p><strong>Notes:</strong> {notes}</p>
                <p><strong>Summary:</strong> {summary}</p>
                <p><a href="{DASHBOARD_URL}">Open the lead dashboard</a></p>
            """
        }

        api_request = urllib_request.Request(
            "https://api.resend.com/emails",
            data=json.dumps(email_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": "veronika-chatbot/1.0",
            },
            method="POST"
        )

        with urllib_request.urlopen(api_request, timeout=15) as response:
            response.read()

        print(f"Booking notification sent for session {session_id}")

    except urllib_error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        print(f"Could not send booking notification: HTTP {error.code} {error_body}")

        try:
            (
                supabase.table("conversations")
                .update({"notification_sent_at": None})
                .eq("session_id", session_id)
                .execute()
            )
        except Exception as reset_error:
            print(f"Could not reset notification marker: {reset_error}")

    except Exception as error:
        print(f"Could not send booking notification: {error}")

        try:
            (
                supabase.table("conversations")
                .update({"notification_sent_at": None})
                .eq("session_id", session_id)
                .execute()
            )
        except Exception as reset_error:
            print(f"Could not reset notification marker: {reset_error}")


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

    history_lines = [
        f"{msg.role}: {msg.content}"
        for msg in request.history[-20:]
    ]

    latest_already_present = (
        bool(request.history)
        and request.history[-1].content.strip() == request.message.strip()
    )

    if not latest_already_present:
        history_lines.append(f"customer: {request.message}")

    conversation_history = "\n".join(history_lines[-20:])

    extracted_lead = extract_lead_data(conversation_history)
    existing_lead = get_existing_conversation(request.session_id)
    merged_lead = merge_lead_data(existing_lead, extracted_lead)
    merged_lead = apply_simple_date_time_fallback(
        merged_lead,
        conversation_history
    )

    save_conversation_overview(
        session_id=request.session_id,
        lead_data=merged_lead
    )

    missing_fields = [
        field
        for field in REQUIRED_BOOKING_FIELDS
        if merged_lead.get(field) in [None, ""]
    ]

    confirmed_details = format_confirmed_details(merged_lead)
    missing_details = (
        ", ".join(field.replace("_", " ") for field in missing_fields)
        if missing_fields
        else "None"
    )

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
- Do not ask the customer to repeat information they already gave.
- Never ask again for any detail listed under CONFIRMED CUSTOMER DETAILS.
- Ask naturally for only the missing details. Ask for no more than two missing details in one reply.
- If no required details are missing, tell the customer that Veronika will check availability and contact them shortly to confirm.
- Never say a date or time is available, unavailable, free, fully booked, reserved, or confirmed unless that exact availability information exists in the business information.
- Treat any requested date and time as the customer's preference only.
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
- Put the handover confirmation on a new line so the message is easy to read.
- If unsure, suggest calling 07386 396139.

CONFIRMED CUSTOMER DETAILS:
{confirmed_details}

MISSING REQUIRED DETAILS:
{missing_details}

Business information:
{context}

Conversation history:
{conversation_history}

Reply as Veronika's assistant:
"""

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    reply = completion.choices[0].message.content

    save_message(
        session_id=request.session_id,
        role="assistant",
        content=reply
    )

    background_tasks.add_task(
        send_booking_notification,
        request.session_id
    )

    return {"reply": reply}
