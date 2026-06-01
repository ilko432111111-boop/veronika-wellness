from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
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
from urllib import parse as urllib_parse
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
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
BUSINESS_SLUG = "veronika"

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
GOOGLE_OAUTH_STATE_SECRET = os.getenv("GOOGLE_OAUTH_STATE_SECRET")
GOOGLE_CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.freebusy"
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")

# Working hours currently shown on the public website.
# Monday=0 ... Sunday=6
BUSINESS_HOURS = {
    0: (9, 20),
    1: (9, 20),
    2: (9, 20),
    3: (9, 20),
    4: (9, 20),
    5: (10, 18),
    6: None,
}
SLOT_STEP_MINUTES = 30
NEXT_SLOT_LIMIT = 3
NEXT_SLOT_SEARCH_DAYS = 7

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


def calculate_lead_status(lead_data: dict, extracted_status: str | None) -> str:
    complete = all(
        lead_data.get(field) not in [None, ""]
        for field in REQUIRED_BOOKING_FIELDS
    )

    if complete:
        return "booking_request_complete"

    booking_progress_fields = [
        "duration",
        "preferred_date",
        "preferred_time",
        "name",
        "phone",
    ]

    has_booking_progress = any(
        lead_data.get(field) not in [None, ""]
        for field in booking_progress_fields
    )

    if extracted_status in ["details_incomplete", "booking_request_complete"]:
        return "details_incomplete"

    if lead_data.get("treatment") and has_booking_progress:
        return "details_incomplete"

    if extracted_status == "interested" or lead_data.get("treatment"):
        return "interested"

    return "new_chat"


def merge_lead_data(existing: dict, extracted: dict) -> dict:
    fields = [
        "name",
        "phone",
        "treatment",
        "duration",
        "preferred_date",
        "preferred_time",
        "notes",
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

    merged["status"] = calculate_lead_status(
        merged,
        extracted.get("status"),
    )

    # If a previous buggy version sent an alert too early, clear the marker
    # while the lead is incomplete so a correct alert can be sent later.
    if merged["status"] != "booking_request_complete":
        merged["notification_sent_at"] = None

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


def next_weekday_date(target_weekday: int):
    today = datetime.now(BUSINESS_TIMEZONE).date()
    days_ahead = (target_weekday - today.weekday()) % 7
    return today + timedelta(days=days_ahead)


def parse_relative_date_from_text(message: str) -> str | None:
    lower_message = message.lower()
    today = datetime.now(BUSINESS_TIMEZONE).date()

    if re.search(r"\btoday\b", lower_message):
        return today.isoformat()

    if re.search(r"\btomorrow\b", lower_message):
        return (today + timedelta(days=1)).isoformat()

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    for weekday_name, weekday_number in weekdays.items():
        if re.search(rf"\b{weekday_name}\b", lower_message):
            return next_weekday_date(weekday_number).isoformat()

    iso_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", message)

    if iso_match:
        return iso_match.group(1)

    return None


def parse_time_from_text(message: str) -> str | None:
    lower_message = message.lower()

    twenty_four_hour_match = re.search(
        r"\b(?:at\s*)?([01]?\d|2[0-3])[:.]([0-5]\d)\b",
        lower_message,
    )

    if twenty_four_hour_match:
        return (
            f"{int(twenty_four_hour_match.group(1)):02d}:"
            f"{twenty_four_hour_match.group(2)}"
        )

    am_pm_match = re.search(
        r"\b(?:at\s*)?(1[0-2]|0?[1-9])(?:[:.]([0-5]\d))?\s*(am|pm)\b",
        lower_message,
    )

    if am_pm_match:
        hour = int(am_pm_match.group(1))
        minute = am_pm_match.group(2) or "00"
        suffix = am_pm_match.group(3)

        if suffix == "pm" and hour != 12:
            hour += 12
        elif suffix == "am" and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute}"

    return None


def parse_phone_from_text(message: str) -> str | None:
    digit_groups = re.findall(r"\+?\d[\d\s()-]{6,}\d", message)

    for candidate in digit_groups:
        digits = re.sub(r"\D", "", candidate)

        if 7 <= len(digits) <= 15:
            return candidate.strip()

    return None


def looks_like_customer_name(message: str) -> bool:
    cleaned = message.strip()
    lower_message = cleaned.lower()

    rejected = {
        "yes",
        "no",
        "sure",
        "okay",
        "ok",
        "today",
        "tomorrow",
        "massage",
        "swedish massage",
        "relaxing massage",
        "deep tissue massage",
    }

    if lower_message in rejected:
        return False

    if any(char.isdigit() for char in cleaned):
        return False

    words = cleaned.split()

    return (
        1 <= len(words) <= 4
        and all(re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ'-]+", word) for word in words)
    )


def apply_latest_message_fallbacks(
    lead_data: dict,
    latest_message: str,
    previous_assistant_message: str,
) -> dict:
    """
    Use the latest customer message as the strongest source for obvious details.
    This avoids reusing stale dates or times from earlier messages.
    """
    result = dict(lead_data)

    result["preferred_date"] = normalise_preferred_date(
        result.get("preferred_date")
    )

    latest_date = parse_relative_date_from_text(latest_message)
    latest_time = parse_time_from_text(latest_message)
    latest_phone = parse_phone_from_text(latest_message)

    if latest_date:
        result["preferred_date"] = latest_date

    if latest_time:
        result["preferred_time"] = latest_time

    if latest_phone:
        result["phone"] = latest_phone

    if (
        result.get("name") in [None, ""]
        and "name" in previous_assistant_message.lower()
        and looks_like_customer_name(latest_message)
    ):
        result["name"] = latest_message.strip()

    result["status"] = calculate_lead_status(
        result,
        result.get("status"),
    )

    if result["status"] != "booking_request_complete":
        result["notification_sent_at"] = None

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


def format_customer_date(value: str | None) -> str:
    """
    Keep ISO dates in the database, but speak naturally to customers.
    """
    if value in [None, ""]:
        return "the requested date"

    try:
        parsed_date = datetime.fromisoformat(str(value)).date()
        today = datetime.now(BUSINESS_TIMEZONE).date()

        if parsed_date == today:
            return "today"

        if parsed_date == today + timedelta(days=1):
            return "tomorrow"

        return (
            f"{parsed_date.strftime('%A')} "
            f"{parsed_date.day} "
            f"{parsed_date.strftime('%B')}"
        )

    except ValueError:
        return str(value)


def format_customer_time(value: str | None) -> str:
    if value in [None, ""]:
        return "the requested time"

    try:
        parsed_time = datetime.strptime(str(value), "%H:%M")
        return parsed_time.strftime("%H:%M")

    except ValueError:
        return str(value)


def format_customer_slot(
    preferred_date: str | None,
    preferred_time: str | None,
) -> str:
    return (
        f"{format_customer_date(preferred_date)} "
        f"at {format_customer_time(preferred_time)}"
    )


def is_simple_acknowledgement(message: str) -> bool:
    cleaned = re.sub(r"[^a-z]+", " ", message.lower()).strip()

    acknowledgements = {
        "ok",
        "okay",
        "thanks",
        "thank you",
        "great",
        "perfect",
        "fine",
        "sounds good",
        "sure",
        "alright",
    }

    return cleaned in acknowledgements


def build_next_question_hint(missing_fields: list[str]) -> str:
    if not missing_fields:
        return (
            "No booking details are missing. Do not ask another question. "
            "If the customer is only acknowledging the previous message, "
            "reply briefly: Thanks. The therapist will confirm the appointment shortly."
        )

    next_field = missing_fields[0]

    questions = {
        "treatment": "Ask only which treatment they would like.",
        "duration": "Ask only how long they would like the session for.",
        "name": "Ask only for their name.",
        "phone": "Ask only for their phone number.",
        "preferred_date": "Ask only which date they would prefer.",
        "preferred_time": "Ask only which time they would prefer.",
    }

    return questions.get(
        next_field,
        "Ask only for the next missing booking detail."
    )


def format_confirmed_details(lead_data: dict) -> str:
    lines = []

    for field in REQUIRED_BOOKING_FIELDS:
        value = lead_data.get(field)

        if value not in [None, ""]:
            label = field.replace("_", " ").title()

            if field == "preferred_date":
                value = format_customer_date(value)

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

        missing_email_fields = [
            field
            for field in REQUIRED_BOOKING_FIELDS
            if lead.get(field) in [None, ""]
        ]

        if missing_email_fields:
            print(
                "Email notification skipped because required fields are missing: "
                + ", ".join(missing_email_fields)
            )
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


def google_oauth_is_configured() -> bool:
    return all([
        GOOGLE_CLIENT_ID,
        GOOGLE_CLIENT_SECRET,
        GOOGLE_REDIRECT_URI,
        GOOGLE_OAUTH_STATE_SECRET,
    ])


def create_google_oauth_state() -> str:
    if not GOOGLE_OAUTH_STATE_SECRET:
        raise RuntimeError("GOOGLE_OAUTH_STATE_SECRET is missing.")

    payload = f"{int(time.time())}:{secrets.token_urlsafe(18)}"
    signature = hmac.new(
        GOOGLE_OAUTH_STATE_SECRET.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    token = f"{payload}:{signature}".encode("utf-8")
    return base64.urlsafe_b64encode(token).decode("utf-8")


def verify_google_oauth_state(state: str, max_age_seconds: int = 900) -> bool:
    if not GOOGLE_OAUTH_STATE_SECRET:
        return False

    try:
        decoded = base64.urlsafe_b64decode(state.encode("utf-8")).decode("utf-8")
        timestamp_text, nonce, signature = decoded.split(":", 2)
        payload = f"{timestamp_text}:{nonce}"

        expected_signature = hmac.new(
            GOOGLE_OAUTH_STATE_SECRET.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
            return False

        age_seconds = int(time.time()) - int(timestamp_text)
        return 0 <= age_seconds <= max_age_seconds

    except Exception:
        return False


def save_google_refresh_token(refresh_token: str):
    supabase.table("google_calendar_tokens").upsert(
        {
            "business_slug": BUSINESS_SLUG,
            "refresh_token": refresh_token,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        on_conflict="business_slug",
    ).execute()


def get_google_refresh_token() -> str | None:
    try:
        response = (
            supabase.table("google_calendar_tokens")
            .select("refresh_token")
            .eq("business_slug", BUSINESS_SLUG)
            .limit(1)
            .execute()
        )

        if response.data:
            return response.data[0].get("refresh_token")

    except Exception as error:
        print(f"Could not load Google refresh token: {error}")

    return None


def exchange_google_code_for_tokens(code: str) -> dict:
    if not google_oauth_is_configured():
        raise RuntimeError("Google OAuth environment variables are incomplete.")

    token_payload = urllib_parse.urlencode({
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }).encode("utf-8")

    token_request = urllib_request.Request(
        "https://oauth2.googleapis.com/token",
        data=token_payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "veronika-chatbot/1.0",
        },
        method="POST",
    )

    with urllib_request.urlopen(token_request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def refresh_google_access_token() -> str | None:
    refresh_token = get_google_refresh_token()

    if not refresh_token:
        return None

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        print("Could not refresh Google access token: missing OAuth credentials.")
        return None

    try:
        token_payload = urllib_parse.urlencode({
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }).encode("utf-8")

        token_request = urllib_request.Request(
            "https://oauth2.googleapis.com/token",
            data=token_payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "veronika-chatbot/1.0",
            },
            method="POST",
        )

        with urllib_request.urlopen(token_request, timeout=15) as response:
            token_data = json.loads(response.read().decode("utf-8"))

        return token_data.get("access_token")

    except urllib_error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        print(f"Could not refresh Google access token: HTTP {error.code} {body}")

    except Exception as error:
        print(f"Could not refresh Google access token: {error}")

    return None


def parse_duration_minutes(value: str | None) -> int | None:
    if value in [None, ""]:
        return None

    cleaned = str(value).strip().lower()
    total_minutes = 0

    hours_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs)\b",
        cleaned,
    )

    minutes_match = re.search(
        r"(\d+)\s*(?:minute|minutes|min|mins)\b",
        cleaned,
    )

    if hours_match:
        total_minutes += round(float(hours_match.group(1)) * 60)

    if minutes_match:
        total_minutes += int(minutes_match.group(1))

    if total_minutes > 0:
        return total_minutes

    if re.fullmatch(r"\d+", cleaned):
        return int(cleaned)

    return None


def build_requested_slot(lead_data: dict):
    preferred_date = lead_data.get("preferred_date")
    preferred_time = lead_data.get("preferred_time")
    duration_minutes = parse_duration_minutes(lead_data.get("duration"))

    if not preferred_date or not preferred_time:
        return None

    try:
        start_time = datetime.fromisoformat(
            f"{preferred_date}T{preferred_time}"
        ).replace(tzinfo=BUSINESS_TIMEZONE)

        # If duration is not known yet, check whether the requested start
        # itself already clashes with a busy period.
        check_minutes = duration_minutes or 1
        end_time = start_time + timedelta(minutes=check_minutes)

        return start_time, end_time, duration_minutes

    except ValueError:
        return None


def query_google_freebusy(
    start_time: datetime,
    end_time: datetime,
) -> list[dict] | None:
    access_token = refresh_google_access_token()

    if not access_token:
        return None

    payload = {
        "timeMin": start_time.isoformat(),
        "timeMax": end_time.isoformat(),
        "timeZone": "Europe/London",
        "items": [
            {"id": GOOGLE_CALENDAR_ID}
        ],
    }

    try:
        freebusy_request = urllib_request.Request(
            "https://www.googleapis.com/calendar/v3/freeBusy",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "User-Agent": "receptionist-chatbot/1.0",
            },
            method="POST",
        )

        with urllib_request.urlopen(freebusy_request, timeout=15) as response:
            freebusy_data = json.loads(response.read().decode("utf-8"))

        calendar_data = (
            freebusy_data
            .get("calendars", {})
            .get(GOOGLE_CALENDAR_ID, {})
        )

        if calendar_data.get("errors"):
            print(f"Google Calendar freeBusy returned errors: {calendar_data['errors']}")
            return None

        return calendar_data.get("busy", [])

    except urllib_error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        print(f"Google Calendar freeBusy failed: HTTP {error.code} {body}")

    except Exception as error:
        print(f"Google Calendar freeBusy failed: {error}")

    return None


def parse_google_datetime(value: str) -> datetime:
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    ).astimezone(BUSINESS_TIMEZONE)


def slot_overlaps_busy_period(
    start_time: datetime,
    end_time: datetime,
    busy_periods: list[dict],
) -> bool:
    for busy_period in busy_periods:
        busy_start = parse_google_datetime(busy_period["start"])
        busy_end = parse_google_datetime(busy_period["end"])

        if start_time < busy_end and end_time > busy_start:
            return True

    return False


def round_up_to_slot_step(value: datetime) -> datetime:
    value = value.replace(second=0, microsecond=0)
    remainder = value.minute % SLOT_STEP_MINUTES

    if remainder:
        value += timedelta(minutes=SLOT_STEP_MINUTES - remainder)

    return value


def find_next_available_slots(
    duration_minutes: int,
    search_start: datetime | None = None,
) -> list[str] | None:
    now = datetime.now(BUSINESS_TIMEZONE)
    search_start = max(search_start or now, now)
    search_end = search_start + timedelta(days=NEXT_SLOT_SEARCH_DAYS)

    busy_periods = query_google_freebusy(search_start, search_end)

    if busy_periods is None:
        return None

    suggestions = []

    for day_offset in range(NEXT_SLOT_SEARCH_DAYS + 1):
        candidate_date = (search_start + timedelta(days=day_offset)).date()
        hours = BUSINESS_HOURS.get(candidate_date.weekday())

        if not hours:
            continue

        open_hour, close_hour = hours

        open_time = datetime.combine(
            candidate_date,
            datetime.min.time(),
            tzinfo=BUSINESS_TIMEZONE,
        ).replace(hour=open_hour)

        close_time = datetime.combine(
            candidate_date,
            datetime.min.time(),
            tzinfo=BUSINESS_TIMEZONE,
        ).replace(hour=close_hour)

        candidate = round_up_to_slot_step(max(open_time, search_start))

        while candidate + timedelta(minutes=duration_minutes) <= close_time:
            candidate_end = candidate + timedelta(minutes=duration_minutes)

            if not slot_overlaps_busy_period(
                candidate,
                candidate_end,
                busy_periods,
            ):
                suggestions.append(
                    format_customer_slot(
                        candidate.date().isoformat(),
                        candidate.strftime("%H:%M"),
                    )
                )

                if len(suggestions) >= NEXT_SLOT_LIMIT:
                    return suggestions

            candidate += timedelta(minutes=SLOT_STEP_MINUTES)

    return suggestions


def asks_for_next_available_time(message: str) -> bool:
    lower_message = message.lower()

    phrases = [
        "when is she free",
        "when are you free",
        "next free",
        "next available",
        "when is the therapist free",
        "what time is free",
        "what times are free",
        "alternative time",
        "another time",
    ]

    return any(phrase in lower_message for phrase in phrases)


def asks_about_booking_or_availability(message: str) -> bool:
    lower_message = message.lower()

    keywords = [
        "available",
        "availability",
        "free",
        "book",
        "booked",
        "appointment",
        "slot",
        "come in",
        "confirm",
    ]

    return any(keyword in lower_message for keyword in keywords)


def should_surface_calendar_result(
    latest_message: str,
    existing_lead: dict,
    merged_lead: dict,
) -> bool:
    date_changed = (
        existing_lead.get("preferred_date")
        != merged_lead.get("preferred_date")
    )

    time_changed = (
        existing_lead.get("preferred_time")
        != merged_lead.get("preferred_time")
    )

    return (
        date_changed
        or time_changed
        or asks_about_booking_or_availability(latest_message)
        or asks_for_next_available_time(latest_message)
    )


def check_requested_calendar_slot(
    lead_data: dict,
    latest_message: str,
) -> dict:
    slot = build_requested_slot(lead_data)

    if not slot:
        return {
            "status": "not_checked",
            "message": (
                "A date and time are not both available yet. "
                "Ask only for whichever one is missing."
            ),
        }

    start_time, end_time, duration_minutes = slot
    customer_slot = format_customer_slot(
        start_time.date().isoformat(),
        start_time.strftime("%H:%M"),
    )

    if end_time <= datetime.now(BUSINESS_TIMEZONE):
        return {
            "status": "past",
            "message": (
                "The requested time is in the past. "
                "Ask for a future preferred time."
            ),
        }

    busy_periods = query_google_freebusy(start_time, end_time)

    if busy_periods is None:
        return {
            "status": "unknown",
            "message": (
                "Calendar availability could not be checked. "
                "Do not guess. Say the therapist will check and confirm."
            ),
        }

    is_busy = slot_overlaps_busy_period(
        start_time,
        end_time,
        busy_periods,
    )

    if is_busy:
        suggestions = None

        if duration_minutes:
            suggestions = find_next_available_slots(
                duration_minutes,
                search_start=end_time,
            )

        suggestion_text = ""

        if suggestions:
            suggestion_text = (
                " Suggested alternatives: "
                + "; ".join(suggestions)
                + "."
            )

        return {
            "status": "busy",
            "message": (
                f"{customer_slot} is unavailable. "
                "Say this once, briefly, and ask whether another preferred "
                "time would suit the customer."
                + suggestion_text
            ),
        }

    if not duration_minutes:
        return {
            "status": "provisional_free",
            "message": (
                f"{customer_slot} currently appears free, but the "
                "duration is still missing. Ask for the duration so the full "
                "slot can be checked. Do not overexplain."
            ),
        }

    if asks_for_next_available_time(latest_message):
        suggestions = find_next_available_slots(
            duration_minutes,
            search_start=datetime.now(BUSINESS_TIMEZONE),
        )

        if suggestions:
            return {
                "status": "suggestions",
                "message": (
                    "Offer these currently free options briefly: "
                    + "; ".join(suggestions)
                    + ". State that the therapist still needs to confirm."
                ),
            }

    return {
        "status": "free",
        "message": (
            f"{customer_slot} currently appears free. "
            "Say this once briefly and state that the therapist still needs "
            "to confirm the booking."
        ),
    }


def format_calendar_result(
    calendar_result: dict,
    should_surface: bool,
) -> str:
    if not should_surface:
        return (
            "Visibility: silent\n"
            "Instruction: Do not mention calendar availability in this reply "
            "unless the customer directly asks about it."
        )

    return (
        "Visibility: mention briefly\n"
        f"Status: {calendar_result.get('status', 'unknown')}\n"
        f"Instruction: {calendar_result.get('message', 'No calendar result available.')}"
    )


@app.get("/")
async def root():
    return {"message": "Backend is working!"}


@app.get("/google-calendar/connect")
async def google_calendar_connect():
    if not google_oauth_is_configured():
        return HTMLResponse(
            """
            <h2>Google Calendar connection is not configured yet.</h2>
            <p>Check the Google OAuth environment variables in Render.</p>
            """,
            status_code=500,
        )

    state = create_google_oauth_state()

    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib_parse.urlencode({
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": GOOGLE_CALENDAR_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    })

    return RedirectResponse(auth_url)


@app.get("/google-calendar/callback")
async def google_calendar_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
):
    if error:
        return HTMLResponse(
            f"""
            <h2>Google Calendar connection was cancelled.</h2>
            <p>Error: {escape(error)}</p>
            """,
            status_code=400,
        )

    if not code or not state or not verify_google_oauth_state(state):
        return HTMLResponse(
            """
            <h2>Google Calendar connection failed.</h2>
            <p>The security check failed or the connection link expired. Please open the connect link again.</p>
            """,
            status_code=400,
        )

    try:
        tokens = exchange_google_code_for_tokens(code)
        refresh_token = tokens.get("refresh_token")

        if not refresh_token:
            return HTMLResponse(
                """
                <h2>No refresh token was returned.</h2>
                <p>Please open the connect link again and approve access when Google asks.</p>
                """,
                status_code=400,
            )

        save_google_refresh_token(refresh_token)

        return HTMLResponse(
            """
            <h2>Google Calendar connected successfully.</h2>
            <p>You can close this page. The chatbot backend now has availability-only access.</p>
            """
        )

    except urllib_error.HTTPError as oauth_error:
        body = oauth_error.read().decode("utf-8", errors="replace")
        print(f"Google OAuth token exchange failed: HTTP {oauth_error.code} {body}")

        return HTMLResponse(
            """
            <h2>Google Calendar connection failed.</h2>
            <p>The Google token exchange was rejected. Check the Render logs for details.</p>
            """,
            status_code=500,
        )

    except Exception as oauth_error:
        print(f"Google Calendar connection failed: {oauth_error}")

        return HTMLResponse(
            """
            <h2>Google Calendar connection failed.</h2>
            <p>Check the Render logs for details.</p>
            """,
            status_code=500,
        )


@app.get("/google-calendar/status")
async def google_calendar_status():
    return {
        "connected": bool(get_google_refresh_token()),
        "calendar_id": GOOGLE_CALENDAR_ID,
        "scope": GOOGLE_CALENDAR_SCOPE,
    }


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

    previous_assistant_message = next(
        (
            msg.content
            for msg in reversed(request.history[:-1])
            if msg.role == "assistant"
        ),
        "",
    )

    merged_lead = apply_latest_message_fallbacks(
        merged_lead,
        request.message,
        previous_assistant_message,
    )

    save_conversation_overview(
        session_id=request.session_id,
        lead_data=merged_lead
    )

    calendar_result = check_requested_calendar_slot(
        merged_lead,
        request.message,
    )

    calendar_should_be_visible = should_surface_calendar_result(
        request.message,
        existing_lead,
        merged_lead,
    )

    calendar_context = format_calendar_result(
        calendar_result,
        calendar_should_be_visible,
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

    next_question_hint = build_next_question_hint(missing_fields)
    customer_date_label = format_customer_date(
        merged_lead.get("preferred_date")
    )

    if is_simple_acknowledgement(request.message) and not missing_fields:
        reply = "Thanks. The therapist will confirm the appointment shortly."

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

    settings = load_chatbot_settings()
    style_instructions = build_style_instructions(settings)

    prompt = f"""
You are Receptionist, the virtual receptionist for Veronika Wellness & Aesthetics in Leeds.

Your job is to answer like a real human receptionist, not like a database.
Never present yourself as Veronika. If you need to refer to the person providing the treatment, say "the therapist".

{style_instructions}

Locked rules:
- Keep replies extremely concise and natural. Usually use one short sentence. Use two only when necessary.
- Answer the customer's question first, then ask only for the single next missing detail.
- Follow NEXT QUESTION TARGET closely.
- Do not repeat a treatment name immediately after the customer has just chosen it unless clarification is genuinely needed.
- Do not repeat prices, dates, times, durations, or availability unless the customer asks or the information has changed.
- Do not list duration options unless duration is the next missing detail.
- If duration is the next missing detail, prefer a reply like: "How long would you like the session for: 30, 60, 90, or 120 minutes?"
- Do not repeat hello twice.
- Avoid filler such as "I've noted that down", "just to confirm", "we offer", "to proceed", and "the requested time currently looks available" unless genuinely needed.
- Use the conversation history to understand short replies.
- Do not ask the customer to repeat information they already gave.
- Never ask again for any detail listed under CONFIRMED CUSTOMER DETAILS.
- Ask naturally for only the missing details. Ask for no more than two missing details in one reply.
- Follow the CALENDAR AVAILABILITY RESULT exactly.
- If CALENDAR AVAILABILITY RESULT says Visibility: silent, do not mention calendar availability.
- When speaking to the customer, use CUSTOMER-FACING DATE LABEL. Say "today" or "tomorrow" where appropriate. Never show an ISO date such as 2026-06-01 to the customer unless they explicitly ask for the exact date.
- If a requested slot is busy, say it once briefly and offer the listed alternatives when available.
- If a requested slot appears free, say it once briefly and clearly state that the therapist still needs to confirm.
- If the calendar cannot be checked, do not guess. Say the therapist will check and confirm.
- Never reveal private calendar event details.
- Never say a booking is confirmed, reserved, or completed.
- If asked "am I booked?", answer clearly: "Not yet. The therapist still needs to confirm the appointment."
- If the customer only says "ok", "thanks", or a similar acknowledgement after giving all details, do not repeat that they are not booked. Reply briefly: "Thanks. The therapist will confirm the appointment shortly."
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

NEXT QUESTION TARGET:
{next_question_hint}

CUSTOMER-FACING DATE LABEL:
{customer_date_label}

CALENDAR AVAILABILITY RESULT:
{calendar_context}

Business information:
{context}

Conversation history:
{conversation_history}

Reply as Receptionist:
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
