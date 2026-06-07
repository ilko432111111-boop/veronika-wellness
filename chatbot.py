from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse
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
import asyncio
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

# AI model configuration
# The live receptionist uses the stronger reasoning model.
# Lead extraction remains separate so it can be upgraded independently later.
LIVE_CHAT_MODEL = os.getenv(
    "LIVE_CHAT_MODEL",
    "openai/gpt-oss-120b",
).strip()

LIVE_CHAT_REASONING_EFFORT = os.getenv(
    "LIVE_CHAT_REASONING_EFFORT",
    "low",
).strip().lower()

if LIVE_CHAT_REASONING_EFFORT not in {"low", "medium", "high"}:
    LIVE_CHAT_REASONING_EFFORT = "low"


def read_positive_int_environment_variable(
    name: str,
    default: int,
) -> int:
    try:
        value = int(os.getenv(name, str(default)))

        if value > 0:
            return value

    except (TypeError, ValueError):
        pass

    return default


LIVE_CHAT_MAX_COMPLETION_TOKENS = read_positive_int_environment_variable(
    "LIVE_CHAT_MAX_COMPLETION_TOKENS",
    1200,
)

LEAD_EXTRACTION_MODEL = os.getenv(
    "LEAD_EXTRACTION_MODEL",
    "llama-3.1-8b-instant",
).strip()

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
BUSINESS_PHONE = "07943319617"
BUSINESS_ADDRESS = "25 Albion Place, Leeds, LS1 6JS"

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
GOOGLE_OAUTH_STATE_SECRET = os.getenv("GOOGLE_OAUTH_STATE_SECRET")
GOOGLE_CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.freebusy"
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")

INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
INSTAGRAM_ACCOUNT_ID = os.getenv("INSTAGRAM_ACCOUNT_ID")
INSTAGRAM_APP_SECRET = os.getenv("INSTAGRAM_APP_SECRET")
INSTAGRAM_VERIFY_TOKEN = os.getenv("INSTAGRAM_VERIFY_TOKEN")
INSTAGRAM_GRAPH_VERSION = os.getenv("INSTAGRAM_GRAPH_VERSION", "v25.0")

# Instagram session and moderation safeguards.
# A completed or stale Instagram conversation starts a fresh lead session.
INSTAGRAM_SESSION_IDLE_HOURS = 12
INSTAGRAM_MAX_MESSAGES_PER_10_MINUTES = 12
INSTAGRAM_MAX_MESSAGES_PER_DAY = 80
INSTAGRAM_RATE_LIMIT_NOTICE_COOLDOWN_MINUTES = 60

SLOT_STEP_MINUTES = 30
NEXT_SLOT_LIMIT = 3
NEXT_SLOT_SEARCH_DAYS = 7

DEFAULT_BUSINESS_HOURS = {
    0: (9, 20),
    1: (9, 20),
    2: (9, 20),
    3: (9, 20),
    4: (9, 20),
    5: (10, 18),
    6: None,
}

REQUIRED_BOOKING_FIELDS = [
    "treatment",
    "duration",
    "preferred_date",
    "preferred_time",
    "name",
    "phone",
]


def booking_request_is_complete(lead_data: dict) -> bool:
    """
    A booking request is complete only when every required field is present.
    Never trust a status label on its own.
    """
    return all(
        lead_data.get(field) not in [None, ""]
        for field in REQUIRED_BOOKING_FIELDS
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    source: str = "website"


def normalise_source(source: str | None) -> str:
    allowed_sources = {"website", "instagram", "whatsapp"}

    cleaned = (source or "website").strip().lower()

    return cleaned if cleaned in allowed_sources else "website"


def save_message(
    session_id: str,
    role: str,
    content: str,
    source: str = "website",
):
    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "role": role,
            "content": content,
            "source": normalise_source(source),
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
    complete = booking_request_is_complete(lead_data)

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


def next_weekday_date(
    target_weekday: int,
    reference_date=None,
    force_next_week: bool = False,
):
    """
    Resolve weekday requests in context.
    If the customer is already discussing a future week, keep later weekday
    changes inside that same week where possible.
    """
    today = datetime.now(BUSINESS_TIMEZONE).date()
    anchor = reference_date or today

    if force_next_week:
        start_of_next_week = today + timedelta(days=(7 - today.weekday()))
        return start_of_next_week + timedelta(days=target_weekday)

    days_ahead = (target_weekday - anchor.weekday()) % 7

    # If there is no future reference and the customer names today after the
    # day has already begun, keep the date resolver simple. Working-hours logic
    # later decides whether the slot is still usable.
    return anchor + timedelta(days=days_ahead)


def parse_relative_date_from_text(
    message: str,
    reference_date: str | None = None,
) -> str | None:
    lower_message = str(message or "").lower()
    today = datetime.now(BUSINESS_TIMEZONE).date()

    reference = None

    if reference_date:
        try:
            reference = datetime.fromisoformat(str(reference_date)).date()
        except ValueError:
            reference = None

    if re.search(r"\btoday\b", lower_message):
        return today.isoformat()

    if re.search(r"\btomorrow\b", lower_message):
        return (today + timedelta(days=1)).isoformat()

    iso_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", lower_message)

    if iso_match:
        return iso_match.group(1)

    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    # Day-first written dates:
    # "14 June", "14th June", "14th of June", "the 14th of June 2026"
    written_date_match = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+"
        r"(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)"
        r"(?:\s+(20\d{2}))?\b",
        lower_message,
    )

    if written_date_match:
        day = int(written_date_match.group(1))
        month = months[written_date_match.group(2)]
        year = int(written_date_match.group(3) or today.year)

        try:
            parsed = datetime(year, month, day).date()

            if not written_date_match.group(3) and parsed < today:
                parsed = datetime(year + 1, month, day).date()

            return parsed.isoformat()

        except ValueError:
            return None

    # Month-first written dates:
    # "June 14", "June 14th", "June 14th 2026"
    month_first_match = re.search(
        r"\b(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+"
        r"(\d{1,2})(?:st|nd|rd|th)?"
        r"(?:\s+(20\d{2}))?\b",
        lower_message,
    )

    if month_first_match:
        month = months[month_first_match.group(1)]
        day = int(month_first_match.group(2))
        year = int(month_first_match.group(3) or today.year)

        try:
            parsed = datetime(year, month, day).date()

            if not month_first_match.group(3) and parsed < today:
                parsed = datetime(year + 1, month, day).date()

            return parsed.isoformat()

        except ValueError:
            return None

    # Numeric UK-style dates:
    # "14/06", "14/06/2026", "14-06-2026"
    numeric_date_match = re.search(
        r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](20\d{2}))?\b",
        lower_message,
    )

    if numeric_date_match:
        day = int(numeric_date_match.group(1))
        month = int(numeric_date_match.group(2))
        year = int(numeric_date_match.group(3) or today.year)

        try:
            parsed = datetime(year, month, day).date()

            if not numeric_date_match.group(3) and parsed < today:
                parsed = datetime(year + 1, month, day).date()

            return parsed.isoformat()

        except ValueError:
            return None

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sauturday": 5,
        "satarday": 5,
        "saterday": 5,
        "sunday": 6,
    }

    for weekday_name, weekday_number in weekdays.items():
        if re.search(rf"\bnext\s+week\s+{weekday_name}\b", lower_message):
            return next_weekday_date(
                weekday_number,
                force_next_week=True,
            ).isoformat()

    for weekday_name, weekday_number in weekdays.items():
        if re.search(rf"\bnext\s+{weekday_name}\b", lower_message):
            resolved = next_weekday_date(
                weekday_number,
                reference_date=today + timedelta(days=1),
            )

            if resolved <= today:
                resolved += timedelta(days=7)

            return resolved.isoformat()

    for weekday_name, weekday_number in weekdays.items():
        if re.search(rf"\b{weekday_name}\b", lower_message):
            anchor = reference if reference and reference >= today else today

            return next_weekday_date(
                weekday_number,
                reference_date=anchor,
            ).isoformat()

    return None


def parse_time_from_text(message: str) -> str | None:
    lower_message = str(message or "").lower()

    # Parse explicit am/pm first. Previously "1.30 pm" was incorrectly caught
    # by the 24-hour pattern and stored as 01:30.
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

    twenty_four_hour_match = re.search(
        r"\b(?:at\s*)?([01]?\d|2[0-3])[:.]([0-5]\d)\b",
        lower_message,
    )

    if twenty_four_hour_match:
        hour = int(twenty_four_hour_match.group(1))
        minute = twenty_four_hour_match.group(2)

        # For daytime appointment businesses, an ambiguous "1.30" normally
        # means 13:30 rather than 01:30. Explicit am/pm always overrides this.
        if 1 <= hour <= 7:
            hour += 12

        return f"{hour:02d}:{minute}"

    return None


def parse_phone_from_text(message: str) -> str | None:
    digit_groups = re.findall(r"\+?\d[\d\s()-]{6,}\d", message)

    for candidate in digit_groups:
        digits = re.sub(r"\D", "", candidate)

        if 10 <= len(digits) <= 15:
            return candidate.strip()

    return None


BOOKING_INTENT_PATTERNS = [
    r"\bbook(?:ing|ed)?\b",
    r"\bappointment\b",
    r"\bschedule\b",
    r"\bslot\b",
    r"\bavailability\b",
    r"\bavailable\b",
    r"\bcan i come(?: in)?\b",
    r"\bi(?:'d| would) like to come(?: in)?\b",
    r"\bi want to come(?: in)?\b",
    r"\bmake (?:an|a) appointment\b",
]


def customer_history_text(conversation_history: str) -> str:
    lines = []

    for line in str(conversation_history or "").splitlines():
        lower_line = line.lower().lstrip()

        if lower_line.startswith(("customer:", "user:")):
            lines.append(line.split(":", 1)[1])

    return "\n".join(lines)


def is_affirmative_reply(message: str) -> bool:
    cleaned = re.sub(r"[^a-z]+", " ", str(message or "").lower()).strip()

    return cleaned in {
        "yes",
        "yes please",
        "yeah",
        "yep",
        "sure",
        "okay",
        "ok",
        "please",
        "sounds good",
        "that works",
    }


def conversation_has_booking_intent(
    conversation_history: str,
    latest_message: str,
    previous_assistant_message: str,
    current_lead: dict,
) -> bool:
    """
    Start the booking-detail flow only when the customer actually indicates
    that they want an appointment. Asking for information about a treatment is
    not enough.
    """
    if current_lead.get("status") in {
        "details_incomplete",
        "booking_request_complete",
    }:
        return True

    customer_text = customer_history_text(conversation_history)
    searchable_text = f"{customer_text}\n{latest_message}".lower()

    if any(
        re.search(pattern, searchable_text)
        for pattern in BOOKING_INTENT_PATTERNS
    ):
        return True

    previous_lower = str(previous_assistant_message or "").lower()

    if (
        is_affirmative_reply(latest_message)
        and any(
            phrase in previous_lower
            for phrase in [
                "would you like to book",
                "would you like me to take your booking request",
                "would you like to request an appointment",
                "shall i take your booking details",
            ]
        )
    ):
        return True

    # Voluntarily providing a date or time while a treatment is already known
    # in the current message or saved lead is also a clear sign that the
    # customer wants to arrange an appointment.
    if current_lead.get("treatment") and (
        parse_relative_date_from_text(latest_message)
        or parse_time_from_text(latest_message)
    ):
        return True

    return False


def normalise_misplaced_preferred_fields(
    lead_data: dict,
    latest_message: str,
) -> dict:
    """
    Correct extraction mistakes such as preferred_time='today'. Relative date
    words are always stored in preferred_date as ISO dates, never as a time.
    """
    result = dict(lead_data)

    raw_date = result.get("preferred_date")
    raw_time = result.get("preferred_time")

    misplaced_date = parse_relative_date_from_text(str(raw_time or ""))

    if misplaced_date and not parse_time_from_text(str(raw_time or "")):
        result["preferred_date"] = misplaced_date
        result["preferred_time"] = None

    misplaced_time = parse_time_from_text(str(raw_date or ""))

    if misplaced_time and not parse_relative_date_from_text(str(raw_date or "")):
        result["preferred_time"] = misplaced_time
        result["preferred_date"] = None

    latest_date = parse_relative_date_from_text(
        latest_message,
        reference_date=result.get("preferred_date"),
    )
    latest_time = parse_time_from_text(latest_message)

    if latest_date:
        result["preferred_date"] = latest_date

        if not latest_time:
            current_time_value = str(result.get("preferred_time") or "").strip().lower()

            if parse_relative_date_from_text(current_time_value):
                result["preferred_time"] = None

    if latest_time:
        result["preferred_time"] = latest_time

    result["preferred_date"] = normalise_preferred_date(
        result.get("preferred_date")
    )

    return result


def apply_authoritative_schedule_fields(
    lead_data: dict,
    existing_lead: dict,
    latest_message: str,
) -> dict:
    """
    Date and time must come from the customer, not from model inference.

    Preserve a previously accepted value from the saved conversation row.
    Replace it only when the latest customer message explicitly supplies a
    new date or time. If no accepted value exists and the latest message does
    not contain one, clear any value invented by extraction.
    """
    result = dict(lead_data)

    existing_date = existing_lead.get("preferred_date")
    existing_time = existing_lead.get("preferred_time")

    latest_date = parse_relative_date_from_text(
        latest_message,
        reference_date=existing_date,
    )
    latest_time = parse_time_from_text(latest_message)

    if latest_date:
        result["preferred_date"] = latest_date
    elif existing_date not in [None, ""]:
        result["preferred_date"] = existing_date
    else:
        result["preferred_date"] = None

    if latest_time:
        result["preferred_time"] = latest_time
    elif existing_time not in [None, ""]:
        result["preferred_time"] = existing_time
    else:
        result["preferred_time"] = None

    return result


def align_status_with_booking_intent(
    lead_data: dict,
    booking_flow_active: bool,
) -> dict:
    result = dict(lead_data)

    if booking_flow_active:
        result["status"] = calculate_lead_status(
            result,
            result.get("status"),
        )
    else:
        result["status"] = (
            "interested"
            if result.get("treatment")
            else "new_chat"
        )
        result["notification_sent_at"] = None

    return result


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

    latest_date = parse_relative_date_from_text(
        latest_message,
        reference_date=result.get("preferred_date"),
    )
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


FALLBACK_MASSAGE_SERVICES = [
    {
        "service_name": "Relaxing Massage",
        "aliases": ["relaxing massage", "relaxation massage"],
        "allowed_durations_minutes": [30, 60, 90, 120],
        "price_by_duration": {
            "30": 3500,
            "60": 5000,
            "90": 7500,
            "120": 9500,
        },
    },
    {
        "service_name": "Swedish Massage",
        "aliases": ["swedish massage", "swedish"],
        "allowed_durations_minutes": [30, 60, 90, 120],
        "price_by_duration": {
            "30": 3500,
            "60": 5000,
            "90": 7500,
            "120": 9500,
        },
    },
    {
        "service_name": "Hot Stone Massage",
        "aliases": ["hot stone massage", "hot stones", "hot stone"],
        "allowed_durations_minutes": [60, 90, 120],
        "price_by_duration": {
            "60": 5500,
            "90": 7500,
            "120": 9500,
        },
    },
    {
        "service_name": "Deep Tissue Massage",
        "aliases": ["deep tissue massage", "deep tissue", "strong massage"],
        "allowed_durations_minutes": [30, 60, 90, 120],
        "price_by_duration": {
            "30": 4000,
            "60": 5500,
            "90": 7500,
            "120": 9500,
        },
    },
]

_MASSAGE_SERVICE_CACHE: dict = {
    "loaded_at": 0.0,
    "rows": None,
    "from_supabase": False,
}
MASSAGE_SERVICE_CACHE_SECONDS = 60


def normalise_treatment_name(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def load_massage_services() -> tuple[list[dict], bool]:
    """
    Load Veronika's live massage catalogue from public.services.

    Returns:
        (rows, from_supabase)

    If Supabase is temporarily unavailable or the table is empty, use a safe
    fallback so the live chatbot does not stop working.
    """
    now = time.time()
    cached_rows = _MASSAGE_SERVICE_CACHE.get("rows")

    if (
        cached_rows is not None
        and now - float(_MASSAGE_SERVICE_CACHE.get("loaded_at") or 0)
        < MASSAGE_SERVICE_CACHE_SECONDS
    ):
        return (
            cached_rows,
            bool(_MASSAGE_SERVICE_CACHE.get("from_supabase")),
        )

    try:
        response = (
            supabase.table("services")
            .select(
                "service_name, aliases, allowed_durations_minutes, "
                "price_by_duration, booking_mode, is_active, display_order"
            )
            .eq("business_slug", BUSINESS_SLUG)
            .eq("category", "massage")
            .eq("is_active", True)
            .order("display_order")
            .execute()
        )

        rows = []

        for row in response.data or []:
            service_name = str(row.get("service_name") or "").strip()
            durations = row.get("allowed_durations_minutes") or []

            if not service_name or not durations:
                continue

            cleaned_durations = sorted({
                int(value)
                for value in durations
                if str(value).isdigit() and int(value) > 0
            })

            if not cleaned_durations:
                continue

            rows.append({
                "service_name": service_name,
                "aliases": list(row.get("aliases") or []),
                "allowed_durations_minutes": cleaned_durations,
                "price_by_duration": dict(row.get("price_by_duration") or {}),
                "booking_mode": row.get("booking_mode") or "choose_duration",
                "display_order": row.get("display_order") or 0,
            })

        if rows:
            _MASSAGE_SERVICE_CACHE.update({
                "loaded_at": now,
                "rows": rows,
                "from_supabase": True,
            })
            return rows, True

    except Exception as error:
        print(f"Could not load massage services from Supabase: {error}")

    fallback_rows = [dict(row) for row in FALLBACK_MASSAGE_SERVICES]

    _MASSAGE_SERVICE_CACHE.update({
        "loaded_at": now,
        "rows": fallback_rows,
        "from_supabase": False,
    })

    return fallback_rows, False


def massage_service_match_score(
    treatment: str | None,
    service: dict,
) -> int:
    cleaned = normalise_service_match_text(treatment)

    if not cleaned:
        return 0

    padded = f" {cleaned} "
    candidates = [
        service.get("service_name"),
        *(service.get("aliases") or []),
    ]

    best_score = 0

    for candidate in candidates:
        cleaned_candidate = normalise_service_match_text(candidate)

        if not cleaned_candidate:
            continue

        if f" {cleaned_candidate} " in padded:
            best_score = max(best_score, len(cleaned_candidate))

    return best_score


def find_massage_service(
    treatment: str | None,
) -> dict | None:
    services, _ = load_massage_services()
    matches = []

    for service in services:
        score = massage_service_match_score(
            treatment,
            service,
        )

        if score:
            matches.append((score, service))

    if not matches:
        return None

    return max(matches, key=lambda item: item[0])[1]


def allowed_durations_for_treatment(
    treatment: str | None,
) -> set[int] | None:
    service = find_massage_service(treatment)

    if not service:
        return None

    return {
        int(value)
        for value in service.get("allowed_durations_minutes") or []
    }


def apply_dynamic_massage_details(
    lead_data: dict,
    latest_message: str,
) -> dict:
    """
    Canonicalise recognised massage names using the services table.

    This keeps casual phrases such as "strong massage" or "hot stones"
    connected to the owner-editable Supabase service row.
    """
    result = dict(lead_data)

    service = (
        find_massage_service(latest_message)
        or find_massage_service(result.get("treatment"))
    )

    if not service:
        return result

    result["treatment"] = service["service_name"]
    result["status"] = calculate_lead_status(
        result,
        result.get("status"),
    )

    if result["status"] != "booking_request_complete":
        result["notification_sent_at"] = None

    return result


def format_price_pence(value) -> str:
    try:
        pence = int(value)
    except (TypeError, ValueError):
        return ""

    pounds = pence / 100

    if pounds.is_integer():
        return f"£{int(pounds)}"

    return f"£{pounds:.2f}"


def format_live_massage_services_context() -> tuple[str, bool]:
    """
    Produce the massage catalogue block sent to the LLM.

    The Supabase services table is authoritative when available.
    """
    services, from_supabase = load_massage_services()
    lines = [
        "Massage services (authoritative live catalogue):"
    ]

    for service in services:
        durations = service.get("allowed_durations_minutes") or []
        prices = service.get("price_by_duration") or {}
        duration_parts = []

        for duration in durations:
            price = format_price_pence(
                prices.get(str(duration))
            )

            if price:
                duration_parts.append(
                    f"{duration} minutes for {price}"
                )
            else:
                duration_parts.append(
                    f"{duration} minutes"
                )

        lines.append(
            f"- {service.get('service_name')}: "
            + ", ".join(duration_parts)
        )

    return "\n".join(lines), from_supabase


FALLBACK_VITAMIN_SHOT_SERVICES = [
    {
        "service_name": "Vitamin Shot",
        "booking_mode": "choose_variant",
        "fixed_duration_minutes": None,
        "price_pence": None,
        "aliases": [
            "vitamin shot",
            "vitamin shots",
            "vitamin injection",
            "vitamin injections",
        ],
        "description": (
            "Ask whether the customer would like a B12 Vitamin Shot "
            "or a Vitamin C Shot."
        ),
    },
    {
        "service_name": "B12 Vitamin Shot",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 15,
        "price_pence": 2000,
        "aliases": [
            "b12 vitamin shot",
            "b12 shot",
            "b12",
            "vitamin b12",
        ],
        "description": "One B12 Vitamin Shot session.",
    },
    {
        "service_name": "Vitamin C Shot",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 15,
        "price_pence": 2000,
        "aliases": [
            "vitamin c shot",
            "c vitamin shot",
            "vitamin c",
            "c shot",
        ],
        "description": "One Vitamin C Shot session.",
    },
]

_VITAMIN_SHOT_SERVICE_CACHE: dict = {
    "loaded_at": 0.0,
    "rows": None,
    "from_supabase": False,
}
VITAMIN_SHOT_SERVICE_CACHE_SECONDS = 60


def load_vitamin_shot_services() -> tuple[list[dict], bool]:
    """
    Load Veronika's live vitamin-shot catalogue from public.services.

    If Supabase temporarily fails or the category is empty, use the fallback
    catalogue so the live chatbot remains available.
    """
    now = time.time()
    cached_rows = _VITAMIN_SHOT_SERVICE_CACHE.get("rows")

    if (
        cached_rows is not None
        and now - float(_VITAMIN_SHOT_SERVICE_CACHE.get("loaded_at") or 0)
        < VITAMIN_SHOT_SERVICE_CACHE_SECONDS
    ):
        return (
            cached_rows,
            bool(_VITAMIN_SHOT_SERVICE_CACHE.get("from_supabase")),
        )

    try:
        response = (
            supabase.table("services")
            .select(
                "service_name, aliases, fixed_duration_minutes, "
                "price_pence, booking_mode, description, is_active, "
                "display_order"
            )
            .eq("business_slug", BUSINESS_SLUG)
            .eq("category", "vitamin_shots")
            .eq("is_active", True)
            .order("display_order")
            .execute()
        )

        rows = []

        for row in response.data or []:
            service_name = str(row.get("service_name") or "").strip()
            booking_mode = str(row.get("booking_mode") or "").strip()

            if not service_name or not booking_mode:
                continue

            fixed_duration = row.get("fixed_duration_minutes")

            if fixed_duration not in [None, ""]:
                try:
                    fixed_duration = int(fixed_duration)
                except (TypeError, ValueError):
                    fixed_duration = None

            rows.append({
                "service_name": service_name,
                "aliases": list(row.get("aliases") or []),
                "fixed_duration_minutes": fixed_duration,
                "price_pence": row.get("price_pence"),
                "booking_mode": booking_mode,
                "description": str(row.get("description") or "").strip(),
                "display_order": row.get("display_order") or 0,
            })

        if rows:
            _VITAMIN_SHOT_SERVICE_CACHE.update({
                "loaded_at": now,
                "rows": rows,
                "from_supabase": True,
            })
            return rows, True

    except Exception as error:
        print(f"Could not load vitamin-shot services from Supabase: {error}")

    fallback_rows = [dict(row) for row in FALLBACK_VITAMIN_SHOT_SERVICES]

    _VITAMIN_SHOT_SERVICE_CACHE.update({
        "loaded_at": now,
        "rows": fallback_rows,
        "from_supabase": False,
    })

    return fallback_rows, False


def vitamin_shot_service_match_score(
    treatment: str | None,
    service: dict,
) -> int:
    cleaned = normalise_service_match_text(treatment)

    if not cleaned:
        return 0

    padded = f" {cleaned} "
    candidates = [
        service.get("service_name"),
        *(service.get("aliases") or []),
    ]

    best_score = 0

    for candidate in candidates:
        cleaned_candidate = normalise_service_match_text(candidate)

        if not cleaned_candidate:
            continue

        if f" {cleaned_candidate} " in padded:
            best_score = max(best_score, len(cleaned_candidate))

    return best_score


def find_vitamin_shot_service(
    treatment: str | None,
) -> dict | None:
    services, _ = load_vitamin_shot_services()
    matches = []

    for service in services:
        score = vitamin_shot_service_match_score(
            treatment,
            service,
        )

        if score:
            matches.append((score, service))

    if not matches:
        return None

    return max(matches, key=lambda item: item[0])[1]


def format_minutes_as_duration(minutes: int | None) -> str | None:
    if minutes in [None, ""]:
        return None

    try:
        minutes = int(minutes)
    except (TypeError, ValueError):
        return None

    if minutes <= 0:
        return None

    hours, remainder = divmod(minutes, 60)

    if hours and remainder:
        hour_label = "hour" if hours == 1 else "hours"
        return f"{hours} {hour_label} {remainder} minutes"

    if hours:
        hour_label = "hour" if hours == 1 else "hours"
        return f"{hours} {hour_label}"

    return f"{minutes} minutes"


def apply_dynamic_vitamin_shot_details(
    lead_data: dict,
    latest_message: str,
) -> dict:
    """
    Canonicalise vitamin-shot variants from public.services.

    Generic "vitamin shot" stays unresolved until the customer chooses B12 or
    Vitamin C. Specific fixed-duration variants receive their duration
    automatically from Supabase.
    """
    result = dict(lead_data)

    service = (
        find_vitamin_shot_service(latest_message)
        or find_vitamin_shot_service(result.get("treatment"))
    )

    if not service:
        return result

    result["treatment"] = service["service_name"]

    if service.get("booking_mode") == "fixed_duration":
        duration = format_minutes_as_duration(
            service.get("fixed_duration_minutes")
        )

        if duration:
            result["duration"] = duration

    else:
        result["duration"] = None

    result["status"] = calculate_lead_status(
        result,
        result.get("status"),
    )

    if result["status"] != "booking_request_complete":
        result["notification_sent_at"] = None

    return result


def needs_vitamin_shot_variant(lead_data: dict) -> bool:
    """
    Generic vitamin-shot enquiries require a B12 or Vitamin C choice before
    the schedule flow continues.
    """
    service = find_vitamin_shot_service(
        lead_data.get("treatment")
    )

    return bool(
        service
        and service.get("booking_mode") == "choose_variant"
    )


def is_dynamic_fixed_duration_treatment(
    treatment: str | None,
) -> bool:
    service = find_vitamin_shot_service(treatment)

    return bool(
        service
        and service.get("booking_mode") == "fixed_duration"
        and service.get("fixed_duration_minutes")
    )


def format_live_vitamin_shot_services_context() -> tuple[str, bool]:
    """
    Produce the vitamin-shot catalogue block sent to the LLM.
    """
    services, from_supabase = load_vitamin_shot_services()
    lines = [
        "Vitamin-shot services (authoritative live catalogue):"
    ]

    for service in services:
        booking_mode = service.get("booking_mode")
        duration = format_minutes_as_duration(
            service.get("fixed_duration_minutes")
        )
        price = format_price_pence(
            service.get("price_pence")
        )
        details = []

        if duration:
            details.append(duration)

        if price:
            details.append(price)

        if service.get("description"):
            details.append(service["description"])

        suffix = "; ".join(details) if details else booking_mode

        lines.append(
            f"- {service.get('service_name')} "
            f"[{booking_mode}]: {suffix}"
        )

    return "\n".join(lines), from_supabase


FALLBACK_ULTRASOUND_SERVICES = [
    {
        "service_name": "Ultrasound — Single Session",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 45,
        "price_pence": 5000,
        "aliases": [
            "ultrasound session",
            "single ultrasound",
            "ultrasound single session",
        ],
        "description": "One ultrasound session lasting 45 minutes.",
    },
    {
        "service_name": "Ultrasound — Package Deal",
        "booking_mode": "package",
        "fixed_duration_minutes": 45,
        "price_pence": 22000,
        "aliases": [
            "ultrasound package",
            "ultrasound package deal",
            "five ultrasound sessions",
            "5 ultrasound sessions",
            "ultrasound course",
        ],
        "description": (
            "Package of 5 separate ultrasound sessions. Each session lasts "
            "45 minutes. Collect a preferred date and time for the first "
            "session only. The therapist arranges the remaining 4 sessions."
        ),
    },
]

_ULTRASOUND_SERVICE_CACHE: dict = {
    "loaded_at": 0.0,
    "rows": None,
    "from_supabase": False,
}
ULTRASOUND_SERVICE_CACHE_SECONDS = 60


def load_ultrasound_services() -> tuple[list[dict], bool]:
    """
    Load Veronika's ultrasound catalogue from public.services.

    If Supabase temporarily fails or the category is empty, retain a safe
    fallback catalogue so the live chatbot remains available.
    """
    now = time.time()
    cached_rows = _ULTRASOUND_SERVICE_CACHE.get("rows")

    if (
        cached_rows is not None
        and now - float(_ULTRASOUND_SERVICE_CACHE.get("loaded_at") or 0)
        < ULTRASOUND_SERVICE_CACHE_SECONDS
    ):
        return (
            cached_rows,
            bool(_ULTRASOUND_SERVICE_CACHE.get("from_supabase")),
        )

    try:
        response = (
            supabase.table("services")
            .select(
                "service_name, aliases, fixed_duration_minutes, "
                "price_pence, booking_mode, description, is_active, "
                "display_order"
            )
            .eq("business_slug", BUSINESS_SLUG)
            .eq("category", "ultrasound")
            .eq("is_active", True)
            .order("display_order")
            .execute()
        )

        rows = []

        for row in response.data or []:
            service_name = str(row.get("service_name") or "").strip()
            booking_mode = str(row.get("booking_mode") or "").strip()

            if not service_name or not booking_mode:
                continue

            fixed_duration = row.get("fixed_duration_minutes")

            if fixed_duration not in [None, ""]:
                try:
                    fixed_duration = int(fixed_duration)
                except (TypeError, ValueError):
                    fixed_duration = None

            rows.append({
                "service_name": service_name,
                "aliases": list(row.get("aliases") or []),
                "fixed_duration_minutes": fixed_duration,
                "price_pence": row.get("price_pence"),
                "booking_mode": booking_mode,
                "description": str(row.get("description") or "").strip(),
                "display_order": row.get("display_order") or 0,
            })

        if rows:
            _ULTRASOUND_SERVICE_CACHE.update({
                "loaded_at": now,
                "rows": rows,
                "from_supabase": True,
            })
            return rows, True

    except Exception as error:
        print(f"Could not load ultrasound services from Supabase: {error}")

    fallback_rows = [dict(row) for row in FALLBACK_ULTRASOUND_SERVICES]

    _ULTRASOUND_SERVICE_CACHE.update({
        "loaded_at": now,
        "rows": fallback_rows,
        "from_supabase": False,
    })

    return fallback_rows, False


def ultrasound_service_match_score(
    treatment: str | None,
    service: dict,
) -> int:
    cleaned = normalise_service_match_text(treatment)

    if not cleaned:
        return 0

    padded = f" {cleaned} "
    candidates = [
        service.get("service_name"),
        *(service.get("aliases") or []),
    ]

    best_score = 0

    for candidate in candidates:
        cleaned_candidate = normalise_service_match_text(candidate)

        if not cleaned_candidate:
            continue

        if f" {cleaned_candidate} " in padded:
            best_score = max(best_score, len(cleaned_candidate))

    return best_score


def find_ultrasound_service(
    treatment: str | None,
) -> dict | None:
    services, _ = load_ultrasound_services()
    matches = []

    for service in services:
        score = ultrasound_service_match_score(
            treatment,
            service,
        )

        if score:
            matches.append((score, service))

    if not matches:
        return None

    return max(matches, key=lambda item: item[0])[1]


def ultrasound_message_is_generic(value: str | None) -> bool:
    cleaned = normalise_service_match_text(value)

    if not cleaned:
        return False

    return (
        "ultrasound" in cleaned
        and not any(
            phrase in cleaned
            for phrase in [
                "package",
                "course",
                "five sessions",
                "5 sessions",
                "single session",
                "one session",
            ]
        )
    )


def apply_dynamic_ultrasound_details(
    lead_data: dict,
    latest_message: str,
) -> dict:
    """
    Handle ultrasound dynamically from public.services.

    A generic ultrasound request remains unresolved until the customer chooses
    a single session or the 5-session package. Both options use a 45-minute
    slot for calendar availability; the package checks the first session only.
    """
    result = dict(lead_data)
    current_treatment = normalise_treatment_name(
        result.get("treatment")
    )
    latest_cleaned = normalise_service_match_text(
        latest_message
    )

    if (
        current_treatment == "ultrasound"
        and any(
            phrase in latest_cleaned
            for phrase in [
                "package",
                "package deal",
                "course",
                "five sessions",
                "5 sessions",
            ]
        )
    ):
        service = next(
            (
                item
                for item in load_ultrasound_services()[0]
                if item.get("booking_mode") == "package"
            ),
            None,
        )

    elif (
        current_treatment == "ultrasound"
        and any(
            phrase in latest_cleaned
            for phrase in [
                "single",
                "single session",
                "one session",
                "just one",
            ]
        )
    ):
        service = next(
            (
                item
                for item in load_ultrasound_services()[0]
                if item.get("booking_mode") == "fixed_duration"
            ),
            None,
        )

    elif current_treatment == "ultrasound":
        # Keep the generic ultrasound state stable until the customer chooses
        # single session or package. Do not let extraction guess a variant.
        result["treatment"] = "Ultrasound"
        result["duration"] = None
        result["status"] = calculate_lead_status(
            result,
            result.get("status"),
        )
        result["notification_sent_at"] = None
        return result

    elif ultrasound_message_is_generic(latest_message):
        result["treatment"] = "Ultrasound"
        result["duration"] = None
        result["status"] = calculate_lead_status(
            result,
            result.get("status"),
        )
        result["notification_sent_at"] = None
        return result

    else:
        service = (
            find_ultrasound_service(latest_message)
            or find_ultrasound_service(result.get("treatment"))
        )

    if not service:
        return result

    result["treatment"] = service["service_name"]

    duration = format_minutes_as_duration(
        service.get("fixed_duration_minutes")
    )

    if duration:
        result["duration"] = duration

    result["status"] = calculate_lead_status(
        result,
        result.get("status"),
    )

    if result["status"] != "booking_request_complete":
        result["notification_sent_at"] = None

    return result


def needs_ultrasound_variant(lead_data: dict) -> bool:
    return (
        normalise_treatment_name(
            lead_data.get("treatment")
        )
        == "ultrasound"
    )


def is_dynamic_ultrasound_fixed_duration(
    treatment: str | None,
) -> bool:
    service = find_ultrasound_service(treatment)

    return bool(
        service
        and service.get("booking_mode") in {
            "fixed_duration",
            "package",
        }
        and service.get("fixed_duration_minutes")
    )


def format_live_ultrasound_services_context() -> tuple[str, bool]:
    """
    Produce the ultrasound catalogue block sent to the LLM.
    """
    services, from_supabase = load_ultrasound_services()
    lines = [
        "Ultrasound services (authoritative live catalogue):"
    ]

    for service in services:
        duration = format_minutes_as_duration(
            service.get("fixed_duration_minutes")
        )
        price = format_price_pence(
            service.get("price_pence")
        )
        details = []

        if duration:
            if service.get("booking_mode") == "package":
                details.append(
                    f"{duration} for the first appointment"
                )
            else:
                details.append(duration)

        if price:
            details.append(price)

        if service.get("description"):
            details.append(service["description"])

        lines.append(
            f"- {service.get('service_name')} "
            f"[{service.get('booking_mode')}]: "
            + "; ".join(details)
        )

    return "\n".join(lines), from_supabase


FALLBACK_BODY_TREATMENT_SERVICES = [
    {
        "service_name": "Body Sculpting Treatment",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 30,
        "price_pence": 4500,
        "aliases": [
            "body sculpting",
            "body sculpting treatment",
            "sculpting treatment",
        ],
        "description": "One body sculpting treatment lasting 30 minutes.",
    },
    {
        "service_name": "Electronic Muscle Stimulation (EMS)",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 60,
        "price_pence": 20000,
        "aliases": [
            "electronic muscle stimulation",
            "ems",
            "muscle stimulation",
        ],
        "description": (
            "One Electronic Muscle Stimulation session lasting 60 minutes."
        ),
    },
]

_BODY_TREATMENT_SERVICE_CACHE: dict = {
    "loaded_at": 0.0,
    "rows": None,
    "from_supabase": False,
}
BODY_TREATMENT_SERVICE_CACHE_SECONDS = 60


def load_body_treatment_services() -> tuple[list[dict], bool]:
    """
    Load Veronika's body-treatment catalogue from public.services.

    If Supabase temporarily fails or the category is empty, use the fallback
    rows so the live chatbot remains available.
    """
    now = time.time()
    cached_rows = _BODY_TREATMENT_SERVICE_CACHE.get("rows")

    if (
        cached_rows is not None
        and now - float(
            _BODY_TREATMENT_SERVICE_CACHE.get("loaded_at") or 0
        )
        < BODY_TREATMENT_SERVICE_CACHE_SECONDS
    ):
        return (
            cached_rows,
            bool(_BODY_TREATMENT_SERVICE_CACHE.get("from_supabase")),
        )

    try:
        response = (
            supabase.table("services")
            .select(
                "service_name, aliases, fixed_duration_minutes, "
                "price_pence, booking_mode, description, is_active, "
                "display_order"
            )
            .eq("business_slug", BUSINESS_SLUG)
            .eq("category", "body_treatments")
            .eq("is_active", True)
            .order("display_order")
            .execute()
        )

        rows = []

        for row in response.data or []:
            service_name = str(row.get("service_name") or "").strip()
            booking_mode = str(row.get("booking_mode") or "").strip()

            if not service_name or not booking_mode:
                continue

            fixed_duration = row.get("fixed_duration_minutes")

            if fixed_duration not in [None, ""]:
                try:
                    fixed_duration = int(fixed_duration)
                except (TypeError, ValueError):
                    fixed_duration = None

            rows.append({
                "service_name": service_name,
                "aliases": list(row.get("aliases") or []),
                "fixed_duration_minutes": fixed_duration,
                "price_pence": row.get("price_pence"),
                "booking_mode": booking_mode,
                "description": str(row.get("description") or "").strip(),
                "display_order": row.get("display_order") or 0,
            })

        if rows:
            _BODY_TREATMENT_SERVICE_CACHE.update({
                "loaded_at": now,
                "rows": rows,
                "from_supabase": True,
            })
            return rows, True

    except Exception as error:
        print(f"Could not load body treatments from Supabase: {error}")

    fallback_rows = [
        dict(row)
        for row in FALLBACK_BODY_TREATMENT_SERVICES
    ]

    _BODY_TREATMENT_SERVICE_CACHE.update({
        "loaded_at": now,
        "rows": fallback_rows,
        "from_supabase": False,
    })

    return fallback_rows, False


def body_treatment_match_score(
    treatment: str | None,
    service: dict,
) -> int:
    cleaned = normalise_service_match_text(treatment)

    if not cleaned:
        return 0

    padded = f" {cleaned} "
    candidates = [
        service.get("service_name"),
        *(service.get("aliases") or []),
    ]

    best_score = 0

    for candidate in candidates:
        cleaned_candidate = normalise_service_match_text(candidate)

        if not cleaned_candidate:
            continue

        if f" {cleaned_candidate} " in padded:
            best_score = max(best_score, len(cleaned_candidate))

    return best_score


def find_body_treatment_service(
    treatment: str | None,
) -> dict | None:
    services, _ = load_body_treatment_services()
    matches = []

    for service in services:
        score = body_treatment_match_score(
            treatment,
            service,
        )

        if score:
            matches.append((score, service))

    if not matches:
        return None

    return max(matches, key=lambda item: item[0])[1]


def apply_dynamic_body_treatment_details(
    lead_data: dict,
    latest_message: str,
) -> dict:
    """
    Canonicalise body sculpting and EMS using public.services.

    Both are fixed-duration treatments, so duration is filled automatically
    from Supabase.
    """
    result = dict(lead_data)

    service = (
        find_body_treatment_service(latest_message)
        or find_body_treatment_service(result.get("treatment"))
    )

    if not service:
        return result

    result["treatment"] = service["service_name"]

    duration = format_minutes_as_duration(
        service.get("fixed_duration_minutes")
    )

    if duration:
        result["duration"] = duration

    result["status"] = calculate_lead_status(
        result,
        result.get("status"),
    )

    if result["status"] != "booking_request_complete":
        result["notification_sent_at"] = None

    return result


def is_dynamic_body_treatment_fixed_duration(
    treatment: str | None,
) -> bool:
    service = find_body_treatment_service(treatment)

    return bool(
        service
        and service.get("booking_mode") == "fixed_duration"
        and service.get("fixed_duration_minutes")
    )


def format_live_body_treatment_services_context() -> tuple[str, bool]:
    """
    Produce the body-treatment catalogue block sent to the LLM.
    """
    services, from_supabase = load_body_treatment_services()
    lines = [
        "Body-treatment services (authoritative live catalogue):"
    ]

    for service in services:
        duration = format_minutes_as_duration(
            service.get("fixed_duration_minutes")
        )
        price = format_price_pence(
            service.get("price_pence")
        )
        details = []

        if duration:
            details.append(duration)

        if price:
            details.append(price)

        if service.get("description"):
            details.append(service["description"])

        lines.append(
            f"- {service.get('service_name')} "
            f"[{service.get('booking_mode')}]: "
            + "; ".join(details)
        )

    return "\n".join(lines), from_supabase


FALLBACK_MICRONEEDLING_SERVICES = [
    {
        "service_name": "Microneedling for 1 Area — Stretch Marks",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 75,
        "price_pence": 6000,
        "aliases": [
            "microneedling stretch marks",
            "microneedling for stretch marks",
            "stretch mark microneedling",
            "microneedling for 1 area stretch marks",
        ],
        "description": (
            "Microneedling treatment for one stretch-mark area. "
            "Session lasts 75 minutes."
        ),
    },
    {
        "service_name": "Microneedling for 1 Area — Face and Neck",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 75,
        "price_pence": 7000,
        "aliases": [
            "microneedling face and neck",
            "microneedling for face and neck",
            "face and neck microneedling",
            "microneedling for 1 area face and neck",
        ],
        "description": (
            "Microneedling treatment for the face and neck. "
            "Session lasts 75 minutes."
        ),
    },
]

_MICRONEEDLING_SERVICE_CACHE: dict = {
    "loaded_at": 0.0,
    "rows": None,
    "from_supabase": False,
}
MICRONEEDLING_SERVICE_CACHE_SECONDS = 60


def load_microneedling_services() -> tuple[list[dict], bool]:
    """
    Load Veronika's microneedling catalogue from public.services.

    If Supabase temporarily fails or the category is empty, use the fallback
    rows so the live chatbot remains available.
    """
    now = time.time()
    cached_rows = _MICRONEEDLING_SERVICE_CACHE.get("rows")

    if (
        cached_rows is not None
        and now - float(
            _MICRONEEDLING_SERVICE_CACHE.get("loaded_at") or 0
        )
        < MICRONEEDLING_SERVICE_CACHE_SECONDS
    ):
        return (
            cached_rows,
            bool(_MICRONEEDLING_SERVICE_CACHE.get("from_supabase")),
        )

    try:
        response = (
            supabase.table("services")
            .select(
                "service_name, aliases, fixed_duration_minutes, "
                "price_pence, booking_mode, description, is_active, "
                "display_order"
            )
            .eq("business_slug", BUSINESS_SLUG)
            .eq("category", "microneedling")
            .eq("is_active", True)
            .order("display_order")
            .execute()
        )

        rows = []

        for row in response.data or []:
            service_name = str(row.get("service_name") or "").strip()
            booking_mode = str(row.get("booking_mode") or "").strip()

            if not service_name or not booking_mode:
                continue

            fixed_duration = row.get("fixed_duration_minutes")

            if fixed_duration not in [None, ""]:
                try:
                    fixed_duration = int(fixed_duration)
                except (TypeError, ValueError):
                    fixed_duration = None

            rows.append({
                "service_name": service_name,
                "aliases": list(row.get("aliases") or []),
                "fixed_duration_minutes": fixed_duration,
                "price_pence": row.get("price_pence"),
                "booking_mode": booking_mode,
                "description": str(row.get("description") or "").strip(),
                "display_order": row.get("display_order") or 0,
            })

        if rows:
            _MICRONEEDLING_SERVICE_CACHE.update({
                "loaded_at": now,
                "rows": rows,
                "from_supabase": True,
            })
            return rows, True

    except Exception as error:
        print(f"Could not load microneedling services from Supabase: {error}")

    fallback_rows = [
        dict(row)
        for row in FALLBACK_MICRONEEDLING_SERVICES
    ]

    _MICRONEEDLING_SERVICE_CACHE.update({
        "loaded_at": now,
        "rows": fallback_rows,
        "from_supabase": False,
    })

    return fallback_rows, False


def microneedling_match_score(
    treatment: str | None,
    service: dict,
) -> int:
    cleaned = normalise_service_match_text(treatment)

    if not cleaned:
        return 0

    padded = f" {cleaned} "
    candidates = [
        service.get("service_name"),
        *(service.get("aliases") or []),
    ]

    best_score = 0

    for candidate in candidates:
        cleaned_candidate = normalise_service_match_text(candidate)

        if not cleaned_candidate:
            continue

        if f" {cleaned_candidate} " in padded:
            best_score = max(best_score, len(cleaned_candidate))

    return best_score


def find_microneedling_service(
    treatment: str | None,
) -> dict | None:
    services, _ = load_microneedling_services()
    matches = []

    for service in services:
        score = microneedling_match_score(
            treatment,
            service,
        )

        if score:
            matches.append((score, service))

    if not matches:
        return None

    return max(matches, key=lambda item: item[0])[1]


def microneedling_message_is_generic(value: str | None) -> bool:
    cleaned = normalise_service_match_text(value)

    if not cleaned:
        return False

    return (
        "microneedling" in cleaned
        and not any(
            phrase in cleaned
            for phrase in [
                "stretch mark",
                "stretch marks",
                "face and neck",
                "face neck",
            ]
        )
    )


def apply_dynamic_microneedling_details(
    lead_data: dict,
    latest_message: str,
) -> dict:
    """
    Handle microneedling dynamically from public.services.

    A generic microneedling request remains unresolved until the customer
    chooses stretch marks or face and neck. Specific variants receive their
    fixed duration automatically from Supabase.
    """
    result = dict(lead_data)
    current_treatment = normalise_treatment_name(
        result.get("treatment")
    )

    # The newest customer message takes priority. This allows short replies
    # such as "stretch marks" or "face and neck" to resolve the variant after
    # the chatbot has asked the customer to choose one.
    service = find_microneedling_service(
        latest_message
    )

    if not service and current_treatment == "microneedling":
        service = find_service_from_short_variant_reply(
            latest_message,
            load_microneedling_services()[0],
            category_tokens={"microneedling"},
        )

    if service:
        result["treatment"] = service["service_name"]

        duration = format_minutes_as_duration(
            service.get("fixed_duration_minutes")
        )

        if duration:
            result["duration"] = duration

        result["status"] = calculate_lead_status(
            result,
            result.get("status"),
        )

        if result["status"] != "booking_request_complete":
            result["notification_sent_at"] = None

        return result

    if current_treatment == "microneedling":
        result["duration"] = None
        result["status"] = calculate_lead_status(
            result,
            result.get("status"),
        )
        result["notification_sent_at"] = None
        return result

    if microneedling_message_is_generic(latest_message):
        result["treatment"] = "Microneedling"
        result["duration"] = None
        result["status"] = calculate_lead_status(
            result,
            result.get("status"),
        )
        result["notification_sent_at"] = None
        return result

    service = find_microneedling_service(
        result.get("treatment")
    )

    if not service:
        return result

    result["treatment"] = service["service_name"]

    duration = format_minutes_as_duration(
        service.get("fixed_duration_minutes")
    )

    if duration:
        result["duration"] = duration

    result["status"] = calculate_lead_status(
        result,
        result.get("status"),
    )

    if result["status"] != "booking_request_complete":
        result["notification_sent_at"] = None

    return result


def needs_microneedling_variant(lead_data: dict) -> bool:
    return (
        normalise_treatment_name(
            lead_data.get("treatment")
        )
        == "microneedling"
    )


def is_dynamic_microneedling_fixed_duration(
    treatment: str | None,
) -> bool:
    service = find_microneedling_service(treatment)

    return bool(
        service
        and service.get("booking_mode") == "fixed_duration"
        and service.get("fixed_duration_minutes")
    )


def format_live_microneedling_services_context() -> tuple[str, bool]:
    """
    Produce the microneedling catalogue block sent to the LLM.
    """
    services, from_supabase = load_microneedling_services()
    lines = [
        "Microneedling services (authoritative live catalogue):"
    ]

    for service in services:
        duration = format_minutes_as_duration(
            service.get("fixed_duration_minutes")
        )
        price = format_price_pence(
            service.get("price_pence")
        )
        details = []

        if duration:
            details.append(duration)

        if price:
            details.append(price)

        if service.get("description"):
            details.append(service["description"])

        lines.append(
            f"- {service.get('service_name')} "
            f"[{service.get('booking_mode')}]: "
            + "; ".join(details)
        )

    return "\n".join(lines), from_supabase


FALLBACK_FACIAL_SERVICES = [
    {
        "service_name": "Deep Facial Cleanse",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 60,
        "price_pence": 6000,
        "aliases": [
            "deep facial cleanse",
            "facial cleanse",
            "deep cleanse facial",
        ],
        "description": "Deep facial cleanse treatment lasting 60 minutes.",
    },
    {
        "service_name": "Radio Frequency Facial",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 60,
        "price_pence": 6000,
        "aliases": [
            "radio frequency facial",
            "rf facial",
            "radio frequency treatment",
        ],
        "description": "Radio Frequency Facial treatment lasting 60 minutes.",
    },
    {
        "service_name": "Facial Treatment for Young Skin",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 60,
        "price_pence": 4500,
        "aliases": [
            "facial treatment for young skin",
            "young skin facial",
            "facial for young skin",
        ],
        "description": "Facial treatment for young skin lasting 60 minutes.",
    },
    {
        "service_name": "Hydraface Facial Treatment",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 60,
        "price_pence": 6000,
        "aliases": [
            "hydraface facial treatment",
            "hydraface facial",
            "hydraface",
        ],
        "description": "Hydraface Facial Treatment lasting 60 minutes.",
    },
    {
        "service_name": "Hydrafacial",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 90,
        "price_pence": 8000,
        "aliases": [
            "hydrafacial",
            "hydrafacial 90 minutes",
            "90 minute hydrafacial",
        ],
        "description": "Hydrafacial treatment lasting 90 minutes.",
    },
]

_FACIAL_SERVICE_CACHE: dict = {
    "loaded_at": 0.0,
    "rows": None,
    "from_supabase": False,
}
FACIAL_SERVICE_CACHE_SECONDS = 60


def load_facial_services() -> tuple[list[dict], bool]:
    """
    Load Veronika's facial catalogue from public.services.

    If Supabase temporarily fails or the category is empty, use the fallback
    rows so the live chatbot remains available.
    """
    now = time.time()
    cached_rows = _FACIAL_SERVICE_CACHE.get("rows")

    if (
        cached_rows is not None
        and now - float(
            _FACIAL_SERVICE_CACHE.get("loaded_at") or 0
        )
        < FACIAL_SERVICE_CACHE_SECONDS
    ):
        return (
            cached_rows,
            bool(_FACIAL_SERVICE_CACHE.get("from_supabase")),
        )

    try:
        response = (
            supabase.table("services")
            .select(
                "service_name, aliases, fixed_duration_minutes, "
                "price_pence, booking_mode, description, is_active, "
                "display_order"
            )
            .eq("business_slug", BUSINESS_SLUG)
            .eq("category", "facials")
            .eq("is_active", True)
            .order("display_order")
            .execute()
        )

        rows = []

        for row in response.data or []:
            service_name = str(row.get("service_name") or "").strip()
            booking_mode = str(row.get("booking_mode") or "").strip()

            if not service_name or not booking_mode:
                continue

            fixed_duration = row.get("fixed_duration_minutes")

            if fixed_duration not in [None, ""]:
                try:
                    fixed_duration = int(fixed_duration)
                except (TypeError, ValueError):
                    fixed_duration = None

            rows.append({
                "service_name": service_name,
                "aliases": list(row.get("aliases") or []),
                "fixed_duration_minutes": fixed_duration,
                "price_pence": row.get("price_pence"),
                "booking_mode": booking_mode,
                "description": str(row.get("description") or "").strip(),
                "display_order": row.get("display_order") or 0,
            })

        if rows:
            _FACIAL_SERVICE_CACHE.update({
                "loaded_at": now,
                "rows": rows,
                "from_supabase": True,
            })
            return rows, True

    except Exception as error:
        print(f"Could not load facial services from Supabase: {error}")

    fallback_rows = [
        dict(row)
        for row in FALLBACK_FACIAL_SERVICES
    ]

    _FACIAL_SERVICE_CACHE.update({
        "loaded_at": now,
        "rows": fallback_rows,
        "from_supabase": False,
    })

    return fallback_rows, False


def facial_match_score(
    treatment: str | None,
    service: dict,
) -> int:
    cleaned = normalise_service_match_text(treatment)

    if not cleaned:
        return 0

    padded = f" {cleaned} "
    candidates = [
        service.get("service_name"),
        *(service.get("aliases") or []),
    ]

    best_score = 0

    for candidate in candidates:
        cleaned_candidate = normalise_service_match_text(candidate)

        if not cleaned_candidate:
            continue

        if f" {cleaned_candidate} " in padded:
            best_score = max(best_score, len(cleaned_candidate))

    return best_score


def find_facial_service(
    treatment: str | None,
) -> dict | None:
    services, _ = load_facial_services()
    matches = []

    for service in services:
        score = facial_match_score(
            treatment,
            service,
        )

        if score:
            matches.append((score, service))

    if not matches:
        return None

    return max(matches, key=lambda item: item[0])[1]


def facial_message_is_generic(value: str | None) -> bool:
    cleaned = normalise_service_match_text(value)

    if not cleaned:
        return False

    if "facial" not in cleaned:
        return False

    return find_facial_service(value) is None


def apply_dynamic_facial_details(
    lead_data: dict,
    latest_message: str,
) -> dict:
    """
    Handle facials dynamically from public.services.

    A generic facial request remains unresolved until the customer chooses a
    specific facial. Specific services receive their fixed duration
    automatically from Supabase.
    """
    result = dict(lead_data)
    current_treatment = normalise_treatment_name(
        result.get("treatment")
    )

    service = find_facial_service(
        latest_message
    )

    if not service and current_treatment == "facial":
        service = find_service_from_short_variant_reply(
            latest_message,
            load_facial_services()[0],
            category_tokens={"facial"},
        )

    if service:
        result["treatment"] = service["service_name"]

        duration = format_minutes_as_duration(
            service.get("fixed_duration_minutes")
        )

        if duration:
            result["duration"] = duration

        result["status"] = calculate_lead_status(
            result,
            result.get("status"),
        )

        if result["status"] != "booking_request_complete":
            result["notification_sent_at"] = None

        return result

    if current_treatment == "facial":
        result["duration"] = None
        result["status"] = calculate_lead_status(
            result,
            result.get("status"),
        )
        result["notification_sent_at"] = None
        return result

    if facial_message_is_generic(latest_message):
        result["treatment"] = "Facial"
        result["duration"] = None
        result["status"] = calculate_lead_status(
            result,
            result.get("status"),
        )
        result["notification_sent_at"] = None
        return result

    service = find_facial_service(
        result.get("treatment")
    )

    if not service:
        return result

    result["treatment"] = service["service_name"]

    duration = format_minutes_as_duration(
        service.get("fixed_duration_minutes")
    )

    if duration:
        result["duration"] = duration

    result["status"] = calculate_lead_status(
        result,
        result.get("status"),
    )

    if result["status"] != "booking_request_complete":
        result["notification_sent_at"] = None

    return result


def needs_facial_variant(lead_data: dict) -> bool:
    return (
        normalise_treatment_name(
            lead_data.get("treatment")
        )
        == "facial"
    )


def is_dynamic_facial_fixed_duration(
    treatment: str | None,
) -> bool:
    service = find_facial_service(treatment)

    return bool(
        service
        and service.get("booking_mode") == "fixed_duration"
        and service.get("fixed_duration_minutes")
    )


def format_live_facial_services_context() -> tuple[str, bool]:
    """
    Produce the facial catalogue block sent to the LLM.
    """
    services, from_supabase = load_facial_services()
    lines = [
        "Facial services (authoritative live catalogue):"
    ]

    for service in services:
        duration = format_minutes_as_duration(
            service.get("fixed_duration_minutes")
        )
        price = format_price_pence(
            service.get("price_pence")
        )
        details = []

        if duration:
            details.append(duration)

        if price:
            details.append(price)

        if service.get("description"):
            details.append(service["description"])

        lines.append(
            f"- {service.get('service_name')} "
            f"[{service.get('booking_mode')}]: "
            + "; ".join(details)
        )

    return "\n".join(lines), from_supabase


FALLBACK_DERMAL_FILLER_SERVICES = [
    {
        "service_name": "Dermal Filler — Lip Filler 0.5 ml",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 45,
        "price_pence": 7000,
        "aliases": [
            "lip filler 0.5 ml",
            "0.5 ml lip filler",
            "lip filler 0.5ml",
            "0.5ml lip filler",
            "lip 0.5 ml",
            "lip 0.5ml",
        ],
        "description": "Lip filler treatment using 0.5 ml. Session lasts 45 minutes.",
    },
    {
        "service_name": "Dermal Filler — Marionette Lines 0.5 ml",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 45,
        "price_pence": 7000,
        "aliases": [
            "marionette lines 0.5 ml",
            "0.5 ml marionette lines",
            "marionette lines 0.5ml",
            "0.5ml marionette lines",
            "marionette 0.5 ml",
            "marionette 0.5ml",
        ],
        "description": "Dermal filler treatment for marionette lines using 0.5 ml. Session lasts 45 minutes.",
    },
    {
        "service_name": "Dermal Filler — Nasolabial Folds 0.5 ml",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 45,
        "price_pence": 7000,
        "aliases": [
            "nasolabial folds 0.5 ml",
            "0.5 ml nasolabial folds",
            "nasolabial folds 0.5ml",
            "0.5ml nasolabial folds",
            "nasolabial 0.5 ml",
            "nasolabial 0.5ml",
        ],
        "description": "Dermal filler treatment for nasolabial folds using 0.5 ml. Session lasts 45 minutes.",
    },
    {
        "service_name": "Dermal Filler — Lip Filler 1 ml",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 60,
        "price_pence": 10000,
        "aliases": [
            "lip filler 1 ml",
            "1 ml lip filler",
            "lip filler 1ml",
            "1ml lip filler",
            "lip 1 ml",
            "lip 1ml",
        ],
        "description": "Lip filler treatment using 1 ml. Session lasts 60 minutes.",
    },
    {
        "service_name": "Dermal Filler — Marionette Lines 1 ml",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 60,
        "price_pence": 10000,
        "aliases": [
            "marionette lines 1 ml",
            "1 ml marionette lines",
            "marionette lines 1ml",
            "1ml marionette lines",
            "marionette 1 ml",
            "marionette 1ml",
        ],
        "description": "Dermal filler treatment for marionette lines using 1 ml. Session lasts 60 minutes.",
    },
    {
        "service_name": "Dermal Filler — Nasolabial Folds 1 ml",
        "booking_mode": "fixed_duration",
        "fixed_duration_minutes": 60,
        "price_pence": 10000,
        "aliases": [
            "nasolabial folds 1 ml",
            "1 ml nasolabial folds",
            "nasolabial folds 1ml",
            "1ml nasolabial folds",
            "nasolabial 1 ml",
            "nasolabial 1ml",
        ],
        "description": "Dermal filler treatment for nasolabial folds using 1 ml. Session lasts 60 minutes.",
    },
]

_DERMAL_FILLER_SERVICE_CACHE: dict = {
    "loaded_at": 0.0,
    "rows": None,
    "from_supabase": False,
}
DERMAL_FILLER_SERVICE_CACHE_SECONDS = 60


def load_dermal_filler_services() -> tuple[list[dict], bool]:
    """
    Load Veronika's dermal-filler catalogue from public.services.

    If Supabase temporarily fails or the category is empty, use the fallback
    rows so the live chatbot remains available.
    """
    now = time.time()
    cached_rows = _DERMAL_FILLER_SERVICE_CACHE.get("rows")

    if (
        cached_rows is not None
        and now - float(
            _DERMAL_FILLER_SERVICE_CACHE.get("loaded_at") or 0
        )
        < DERMAL_FILLER_SERVICE_CACHE_SECONDS
    ):
        return (
            cached_rows,
            bool(_DERMAL_FILLER_SERVICE_CACHE.get("from_supabase")),
        )

    try:
        response = (
            supabase.table("services")
            .select(
                "service_name, aliases, fixed_duration_minutes, "
                "price_pence, booking_mode, description, is_active, "
                "display_order"
            )
            .eq("business_slug", BUSINESS_SLUG)
            .eq("category", "dermal_fillers")
            .eq("is_active", True)
            .order("display_order")
            .execute()
        )

        rows = []

        for row in response.data or []:
            service_name = str(row.get("service_name") or "").strip()
            booking_mode = str(row.get("booking_mode") or "").strip()

            if not service_name or not booking_mode:
                continue

            fixed_duration = row.get("fixed_duration_minutes")

            if fixed_duration not in [None, ""]:
                try:
                    fixed_duration = int(fixed_duration)
                except (TypeError, ValueError):
                    fixed_duration = None

            rows.append({
                "service_name": service_name,
                "aliases": list(row.get("aliases") or []),
                "fixed_duration_minutes": fixed_duration,
                "price_pence": row.get("price_pence"),
                "booking_mode": booking_mode,
                "description": str(row.get("description") or "").strip(),
                "display_order": row.get("display_order") or 0,
            })

        if rows:
            _DERMAL_FILLER_SERVICE_CACHE.update({
                "loaded_at": now,
                "rows": rows,
                "from_supabase": True,
            })
            return rows, True

    except Exception as error:
        print(f"Could not load dermal-filler services from Supabase: {error}")

    fallback_rows = [
        dict(row)
        for row in FALLBACK_DERMAL_FILLER_SERVICES
    ]

    _DERMAL_FILLER_SERVICE_CACHE.update({
        "loaded_at": now,
        "rows": fallback_rows,
        "from_supabase": False,
    })

    return fallback_rows, False


def dermal_filler_match_score(
    treatment: str | None,
    service: dict,
) -> int:
    cleaned = normalise_service_match_text(treatment)

    if not cleaned:
        return 0

    padded = f" {cleaned} "
    candidates = [
        service.get("service_name"),
        *(service.get("aliases") or []),
    ]

    best_score = 0

    for candidate in candidates:
        cleaned_candidate = normalise_service_match_text(candidate)

        if not cleaned_candidate:
            continue

        if f" {cleaned_candidate} " in padded:
            best_score = max(best_score, len(cleaned_candidate))

    return best_score


def find_dermal_filler_service(
    treatment: str | None,
) -> dict | None:
    services, _ = load_dermal_filler_services()
    matches = []

    for service in services:
        score = dermal_filler_match_score(
            treatment,
            service,
        )

        if score:
            matches.append((score, service))

    if not matches:
        return None

    return max(matches, key=lambda item: item[0])[1]


DERMAL_FILLER_AREA_LABELS = {
    "lip": "Lip Filler",
    "marionette": "Marionette Lines",
    "nasolabial": "Nasolabial Folds",
}


def dermal_filler_area_from_text(
    value: str | None,
) -> str | None:
    """
    Resolve the treatment area even when the customer has not chosen the
    amount yet.

    Examples:
      "lip filler instead" -> lip
      "marionette lines" -> marionette
      "nasolabial folds" -> nasolabial
    """
    cleaned = normalise_service_match_text(value)

    if not cleaned:
        return None

    if "lip" in cleaned:
        return "lip"

    if "marionette" in cleaned:
        return "marionette"

    if "nasolabial" in cleaned:
        return "nasolabial"

    return None


def dermal_filler_partial_treatment(
    area: str,
) -> str:
    label = DERMAL_FILLER_AREA_LABELS.get(area)

    if not label:
        return "Dermal Filler"

    return f"Dermal Filler — {label}"


def dermal_filler_partial_area(
    treatment: str | None,
) -> str | None:
    cleaned = normalise_service_match_text(treatment)

    if not cleaned.startswith("dermal filler"):
        return None

    return dermal_filler_area_from_text(cleaned)


def is_partial_dermal_filler_treatment(
    treatment: str | None,
) -> bool:
    cleaned = normalise_service_match_text(treatment)

    return bool(
        cleaned.startswith("dermal filler")
        and dermal_filler_partial_area(treatment)
        and find_dermal_filler_service(treatment) is None
    )


def dermal_filler_services_for_area(
    area: str | None,
) -> list[dict]:
    if not area:
        return []

    matches = []

    for service in load_dermal_filler_services()[0]:
        if dermal_filler_area_from_text(
            service.get("service_name")
        ) == area:
            matches.append(service)

    return matches


def find_dermal_filler_service_for_partial_reply(
    current_treatment: str | None,
    latest_message: str,
) -> dict | None:
    """
    Resolve short amount-only replies after the area has already been chosen.

    Example:
      active partial choice: Lip Filler
      newest message: 0.5 ml
      resolved service: Dermal Filler — Lip Filler 0.5 ml
    """
    area = dermal_filler_partial_area(
        current_treatment
    )

    if not area:
        return None

    return find_service_from_short_variant_reply(
        latest_message,
        dermal_filler_services_for_area(area),
        category_tokens={
            "dermal",
            "filler",
            "fillers",
            "lip",
            "marionette",
            "lines",
            "nasolabial",
            "folds",
        },
    )


def customer_asks_dermal_filler_areas(
    message: str,
) -> bool:
    cleaned = normalise_service_match_text(message)

    return any(
        phrase in cleaned
        for phrase in [
            "which area",
            "which areas",
            "what area",
            "what areas",
            "areas do you do",
            "where can",
            "what do you do",
        ]
    )


def build_dermal_filler_variant_prompt(
    lead_data: dict,
    latest_message: str | None = None,
) -> str:
    area = dermal_filler_partial_area(
        lead_data.get("treatment")
    )

    if area:
        label = DERMAL_FILLER_AREA_LABELS[area]
        amount_question = (
            f"For {label.lower()}, would you like 0.5 ml "
            "(£70, 45 minutes) or 1 ml (£100, 1 hour)?"
        )

        if customer_asks_dermal_filler_areas(
            latest_message or ""
        ):
            return (
                "Dermal fillers are available for the lips, marionette "
                "lines, or nasolabial folds. "
                + amount_question
            )

        return amount_question

    return (
        "Dermal fillers are available for the lips, marionette lines, "
        "or nasolabial folds. A 0.5 ml treatment is £70 and takes "
        "45 minutes. A 1 ml treatment is £100 and takes 1 hour. "
        "Which area and amount would you like?"
    )


def dermal_filler_message_is_generic(value: str | None) -> bool:
    cleaned = normalise_service_match_text(value)

    if not cleaned:
        return False

    mentions_fillers = (
        "dermal filler" in cleaned
        or "dermal fillers" in cleaned
        or cleaned == "filler"
        or cleaned == "fillers"
        or "lip filler" in cleaned
    )

    return bool(
        mentions_fillers
        and find_dermal_filler_service(value) is None
    )


def apply_dynamic_dermal_filler_details(
    lead_data: dict,
    latest_message: str,
) -> dict:
    """
    Handle dermal fillers dynamically from public.services.

    Generic filler enquiries remain unresolved until the customer chooses both
    the treatment area and amount. Specific variants receive a fixed duration
    automatically from Supabase.
    """
    result = dict(lead_data)
    current_treatment = normalise_treatment_name(
        result.get("treatment")
    )

    service = find_dermal_filler_service(
        latest_message
    )

    if not service and is_partial_dermal_filler_treatment(
        result.get("treatment")
    ):
        service = find_dermal_filler_service_for_partial_reply(
            result.get("treatment"),
            latest_message,
        )

    if not service and current_treatment == "dermal filler":
        service = find_service_from_short_variant_reply(
            latest_message,
            load_dermal_filler_services()[0],
            category_tokens={"dermal", "filler", "fillers"},
        )

    if service:
        result["treatment"] = service["service_name"]

        duration = format_minutes_as_duration(
            service.get("fixed_duration_minutes")
        )

        if duration:
            result["duration"] = duration

        result["status"] = calculate_lead_status(
            result,
            result.get("status"),
        )

        if result["status"] != "booking_request_complete":
            result["notification_sent_at"] = None

        return result

    latest_area = dermal_filler_area_from_text(
        latest_message
    )

    if latest_area:
        result["treatment"] = dermal_filler_partial_treatment(
            latest_area
        )
        result["duration"] = None
        result["status"] = calculate_lead_status(
            result,
            result.get("status"),
        )
        result["notification_sent_at"] = None
        return result

    if (
        current_treatment == "dermal filler"
        or is_partial_dermal_filler_treatment(
            result.get("treatment")
        )
    ):
        result["duration"] = None
        result["status"] = calculate_lead_status(
            result,
            result.get("status"),
        )
        result["notification_sent_at"] = None
        return result

    if dermal_filler_message_is_generic(latest_message):
        result["treatment"] = "Dermal Filler"
        result["duration"] = None
        result["status"] = calculate_lead_status(
            result,
            result.get("status"),
        )
        result["notification_sent_at"] = None
        return result

    service = find_dermal_filler_service(
        result.get("treatment")
    )

    if not service:
        return result

    result["treatment"] = service["service_name"]

    duration = format_minutes_as_duration(
        service.get("fixed_duration_minutes")
    )

    if duration:
        result["duration"] = duration

    result["status"] = calculate_lead_status(
        result,
        result.get("status"),
    )

    if result["status"] != "booking_request_complete":
        result["notification_sent_at"] = None

    return result


def needs_dermal_filler_variant(lead_data: dict) -> bool:
    treatment = lead_data.get("treatment")

    return (
        normalise_treatment_name(treatment) == "dermal filler"
        or is_partial_dermal_filler_treatment(treatment)
    )


def is_dynamic_dermal_filler_fixed_duration(
    treatment: str | None,
) -> bool:
    service = find_dermal_filler_service(treatment)

    return bool(
        service
        and service.get("booking_mode") == "fixed_duration"
        and service.get("fixed_duration_minutes")
    )


def format_live_dermal_filler_services_context() -> tuple[str, bool]:
    """
    Produce the dermal-filler catalogue block sent to the LLM.
    """
    services, from_supabase = load_dermal_filler_services()
    lines = [
        "Dermal-filler services (authoritative live catalogue):"
    ]

    for service in services:
        duration = format_minutes_as_duration(
            service.get("fixed_duration_minutes")
        )
        price = format_price_pence(
            service.get("price_pence")
        )
        details = []

        if duration:
            details.append(duration)

        if price:
            details.append(price)

        if service.get("description"):
            details.append(service["description"])

        lines.append(
            f"- {service.get('service_name')} "
            f"[{service.get('booking_mode')}]: "
            + "; ".join(details)
        )

    return "\n".join(lines), from_supabase


GENERIC_STRUCTURED_CATEGORIES = {
    "vitamin shot": "vitamin_shots",
    "ultrasound": "ultrasound",
    "microneedling": "microneedling",
    "facial": "facials",
    "dermal filler": "dermal_fillers",
}


def structured_service_identity(
    treatment: str | None,
) -> dict | None:
    """
    Return one normalised service identity for any migrated category.

    This gives the conversation controller one shared way to compare services
    instead of adding another category-specific switch rule each time.
    """
    if treatment in [None, ""]:
        return None

    lookups = [
        ("massage", find_massage_service),
        ("vitamin_shots", find_vitamin_shot_service),
        ("ultrasound", find_ultrasound_service),
        ("body_treatments", find_body_treatment_service),
        ("microneedling", find_microneedling_service),
        ("facials", find_facial_service),
        ("dermal_fillers", find_dermal_filler_service),
    ]

    for category, finder in lookups:
        service = finder(treatment)

        if service:
            return {
                "category": category,
                "service_name": service.get("service_name"),
                "booking_mode": service.get("booking_mode"),
                "fixed_duration_minutes": service.get(
                    "fixed_duration_minutes"
                ),
                "allowed_durations_minutes": service.get(
                    "allowed_durations_minutes"
                ),
            }

    cleaned = normalise_treatment_name(treatment)

    if cleaned in GENERIC_STRUCTURED_CATEGORIES:
        return {
            "category": GENERIC_STRUCTURED_CATEGORIES[cleaned],
            "service_name": cleaned.title(),
            "booking_mode": "choose_variant",
            "fixed_duration_minutes": None,
            "allowed_durations_minutes": None,
        }

    if is_partial_dermal_filler_treatment(treatment):
        return {
            "category": "dermal_fillers",
            "service_name": str(treatment),
            "booking_mode": "choose_variant",
            "fixed_duration_minutes": None,
            "allowed_durations_minutes": None,
        }

    return None


def resolve_latest_generic_or_partial_service(
    latest_message: str,
) -> dict | None:
    cleaned = normalise_service_match_text(
        latest_message
    )

    if not cleaned:
        return None

    area = dermal_filler_area_from_text(
        latest_message
    )

    if area and (
        "filler" in cleaned
        or "dermal" in cleaned
        or "instead" in cleaned
        or "change" in cleaned
        or "switch" in cleaned
    ):
        return {
            "category": "dermal_fillers",
            "service_name": dermal_filler_partial_treatment(area),
            "booking_mode": "choose_variant",
            "fixed_duration_minutes": None,
            "allowed_durations_minutes": None,
        }

    generic_checks = [
        ("vitamin_shots", "Vitamin Shot", ["vitamin shot"]),
        ("ultrasound", "Ultrasound", ["ultrasound"]),
        ("microneedling", "Microneedling", ["microneedling"]),
        ("facials", "Facial", ["facial"]),
        (
            "dermal_fillers",
            "Dermal Filler",
            ["dermal filler", "dermal fillers", "fillers"],
        ),
    ]

    for category, service_name, phrases in generic_checks:
        if any(phrase in cleaned for phrase in phrases):
            return {
                "category": category,
                "service_name": service_name,
                "booking_mode": "choose_variant",
                "fixed_duration_minutes": None,
                "allowed_durations_minutes": None,
            }

    return None


def resolve_latest_structured_service(
    latest_message: str,
    existing_treatment: str | None = None,
) -> dict | None:
    """
    Resolve a service explicitly named in the newest customer message.

    The newest customer message is authoritative when a customer changes their
    mind. Relaxed short-reply matching is used only while a known category
    choice is pending.
    """
    direct_lookups = [
        ("massage", find_massage_service),
        ("vitamin_shots", find_vitamin_shot_service),
        ("ultrasound", find_ultrasound_service),
        ("body_treatments", find_body_treatment_service),
        ("microneedling", find_microneedling_service),
        ("facials", find_facial_service),
        ("dermal_fillers", find_dermal_filler_service),
    ]

    for category, finder in direct_lookups:
        service = finder(latest_message)

        if service:
            return {
                "category": category,
                "service_name": service.get("service_name"),
                "booking_mode": service.get("booking_mode"),
                "fixed_duration_minutes": service.get(
                    "fixed_duration_minutes"
                ),
                "allowed_durations_minutes": service.get(
                    "allowed_durations_minutes"
                ),
            }

    generic_or_partial = resolve_latest_generic_or_partial_service(
        latest_message
    )

    if generic_or_partial:
        return generic_or_partial

    current = normalise_treatment_name(existing_treatment)

    relaxed_categories = {
        "microneedling": (
            "microneedling",
            load_microneedling_services,
            {"microneedling"},
        ),
        "facial": (
            "facials",
            load_facial_services,
            {"facial"},
        ),
        "dermal filler": (
            "dermal_fillers",
            load_dermal_filler_services,
            {"dermal", "filler", "fillers"},
        ),
    }

    if current in relaxed_categories:
        category, loader, category_tokens = relaxed_categories[current]
        service = find_service_from_short_variant_reply(
            latest_message,
            loader()[0],
            category_tokens=category_tokens,
        )

        if service:
            return {
                "category": category,
                "service_name": service.get("service_name"),
                "booking_mode": service.get("booking_mode"),
                "fixed_duration_minutes": service.get(
                    "fixed_duration_minutes"
                ),
                "allowed_durations_minutes": service.get(
                    "allowed_durations_minutes"
                ),
            }

    return None


def duration_for_structured_service(
    service_identity: dict | None,
) -> str | None:
    if not service_identity:
        return None

    return format_minutes_as_duration(
        service_identity.get("fixed_duration_minutes")
    )


def apply_generic_service_switch_controller(
    existing_lead: dict,
    lead_data: dict,
    latest_message: str,
) -> tuple[dict, dict | None]:
    """
    Detect a customer changing service and clear only the details invalidated
    by that switch.

    Preserved:
      - preferred date
      - preferred time
      - name
      - phone
      - notes

    Recalculated:
      - treatment
      - duration
      - calendar slot status
      - next required detail
    """
    result = dict(lead_data)

    previous_identity = structured_service_identity(
        existing_lead.get("treatment")
    )
    selected_identity = resolve_latest_structured_service(
        latest_message,
        existing_treatment=existing_lead.get("treatment"),
    )

    if not selected_identity:
        current_identity = structured_service_identity(
            result.get("treatment")
        )

        if current_identity:
            result["active_category"] = current_identity["category"]

        return result, None

    previous_name = (
        previous_identity.get("service_name")
        if previous_identity
        else None
    )
    selected_name = selected_identity.get("service_name")

    if not selected_name:
        return result, None

    result["treatment"] = selected_name
    result["active_category"] = selected_identity.get("category")

    fixed_duration = duration_for_structured_service(
        selected_identity
    )

    if fixed_duration:
        result["duration"] = fixed_duration
    else:
        # Selectable-duration services, such as massages, must not inherit a
        # stale fixed duration from the previous treatment.
        result["duration"] = None

    switched = bool(
        previous_name
        and normalise_treatment_name(previous_name)
        != normalise_treatment_name(selected_name)
    )

    if not switched:
        result["status"] = calculate_lead_status(
            result,
            result.get("status"),
        )

        if result["status"] != "booking_request_complete":
            result["notification_sent_at"] = None

        return result, None

    switched_at = datetime.now(timezone.utc).isoformat()

    result["slot_status"] = "not_checked"
    result["last_service_switch_at"] = switched_at
    result["notification_sent_at"] = None
    result["status"] = calculate_lead_status(
        result,
        "details_incomplete",
    )

    return result, {
        "previous_service": previous_name,
        "new_service": selected_name,
        "active_category": selected_identity.get("category"),
        "switched_at": switched_at,
    }


def calculate_next_required_detail(
    lead_data: dict,
) -> str | None:
    """
    Calculate one backend-controlled next step.

    Variant selection always comes before duration, date, time, name, or phone.
    """
    if (
        needs_vitamin_shot_variant(lead_data)
        or needs_ultrasound_variant(lead_data)
        or needs_microneedling_variant(lead_data)
        or needs_facial_variant(lead_data)
        or needs_dermal_filler_variant(lead_data)
    ):
        return "service_variant"

    for field in REQUIRED_BOOKING_FIELDS:
        if lead_data.get(field) in [None, ""]:
            return field

    return None


def normalise_controller_slot_status(
    calendar_result: dict | None,
) -> str:
    status = str(
        (calendar_result or {}).get("status") or "not_checked"
    ).strip()

    if status in {
        "not_requested",
        "not_checked",
        "needs_duration_for_alternatives",
    }:
        return "not_checked"

    if status in {
        "outside_hours",
        "closed_day",
        "past",
        "busy",
        "busy_needs_duration",
    }:
        return "unavailable"

    if status in {
        "free",
        "provisional_free",
    }:
        return "provisional_free"

    return status


def apply_conversation_controller_state(
    lead_data: dict,
    booking_flow_active: bool,
    calendar_result: dict | None = None,
) -> dict:
    """
    Store the current reusable workflow state in public.conversations.
    """
    result = dict(lead_data)
    identity = structured_service_identity(
        result.get("treatment")
    )

    result["conversation_mode"] = (
        "booking_request"
        if booking_flow_active
        else "browsing"
    )
    result["active_category"] = (
        identity.get("category")
        if identity
        else result.get("active_category")
    )
    result["slot_status"] = normalise_controller_slot_status(
        calendar_result
    )
    result["next_required_detail"] = (
        calculate_next_required_detail(result)
        if booking_flow_active
        else None
    )

    return result


def format_service_switch_context(
    service_switch: dict | None,
) -> str:
    if not service_switch:
        return "No service switch detected."

    return (
        "Customer changed treatment. "
        f"Previous service: {service_switch.get('previous_service')}. "
        f"New service: {service_switch.get('new_service')}. "
        "Acknowledge the change briefly and continue from the next missing "
        "detail. Do not ask again for details that remain valid."
    )


def format_duration_options(durations: set[int]) -> str:
    ordered = sorted(durations)

    if len(ordered) == 1:
        return f"{ordered[0]} minutes"

    return ", ".join(str(value) for value in ordered[:-1]) + f", or {ordered[-1]} minutes"


# Services with one fixed duration. These are resolved deterministically
# before the AI decides which booking detail to ask for next.
FIXED_TREATMENT_DETAILS = [
    {
        "name": "Fat-Dissolving Injections (Above Knees)",
        "duration": "45 minutes",
        "aliases": ["fat dissolving above knees", "above knees"],
    },
    {
        "name": "Fat-Dissolving Injections (Double Chin)",
        "duration": "45 minutes",
        "aliases": ["fat dissolving double chin", "double chin"],
    },
    {
        "name": "Fat-Dissolving Injections (Bra Area)",
        "duration": "45 minutes",
        "aliases": ["fat dissolving bra area", "bra area"],
    },
    {
        "name": "Fat-Dissolving Injections (Arms)",
        "duration": "45 minutes",
        "aliases": ["fat dissolving arms", "arms"],
    },
    {
        "name": "Fat-Dissolving Injections (Back Area)",
        "duration": "1 hour 15 minutes",
        "aliases": ["fat dissolving back area", "back area"],
    },
    {
        "name": "Fat-Dissolving Injections (Love Handles)",
        "duration": "45 minutes",
        "aliases": ["fat dissolving love handles", "love handles"],
    },
    {
        "name": "Fat-Dissolving Injections (Thighs)",
        "duration": "1 hour 15 minutes",
        "aliases": ["fat dissolving thighs", "thighs"],
    },
    {
        "name": "Fat-Dissolving Injections (Full Stomach and Love Handles)",
        "duration": "1 hour 30 minutes",
        "aliases": [
            "full stomach and love handles",
            "stomach and love handles",
            "full stomach love handles",
        ],
    },
    {
        "name": "Fat-Dissolving Injections (Full Stomach)",
        "duration": "1 hour 45 minutes",
        "aliases": [
            "fat dissolving full stomach",
            "full stomach",
            "stomach area",
            "stomach",
        ],
    },
    {
        "name": "Fat-Dissolving Injections (Hips, Thighs and Bum)",
        "duration": "1 hour 30 minutes",
        "aliases": [
            "hips thighs and bum",
            "hips thighs bum",
            "hips, thighs and bum",
        ],
    },
]


def normalise_service_match_text(value: str | None) -> str:
    cleaned = normalise_treatment_name(value)
    cleaned = cleaned.replace("&", " and ")

    # Common customer spellings. Keep this list conservative: only normalise
    # words where the intended service name is unambiguous.
    replacements = {
        "hydrofacial": "hydrafacial",
        "hydroface": "hydraface",
    }

    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)

    cleaned = re.sub(r"[^a-z0-9.]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


VARIANT_REPLY_IGNORED_TOKENS = {
    "a",
    "an",
    "and",
    "for",
    "the",
    "to",
    "of",
    "treatment",
    "treatments",
    "session",
    "sessions",
    "minute",
    "minutes",
    "hour",
    "hours",
    "area",
    "one",
}


def meaningful_variant_tokens(
    value: str | None,
    category_tokens: set[str] | None = None,
) -> set[str]:
    """
    Extract words useful for resolving a short answer after the chatbot has
    already asked the customer to choose between options.

    Broad category words are removed so a reply such as "facial" does not
    accidentally select one facial service. More specific replies such as
    "radio frequency", "stretch marks", and "hydrafacial 90 minutes" remain
    useful.
    """
    cleaned = normalise_service_match_text(value)

    if not cleaned:
        return set()

    ignored = set(VARIANT_REPLY_IGNORED_TOKENS)

    if category_tokens:
        ignored.update(category_tokens)

    return {
        token
        for token in cleaned.split()
        if token not in ignored
    }


def find_service_from_short_variant_reply(
    value: str | None,
    services: list[dict],
    category_tokens: set[str] | None = None,
) -> dict | None:
    """
    Resolve concise customer replies only while a category choice is pending.

    Example:
      chatbot: Which facial would you like?
      customer: Radio frequency

    The relaxed matching is intentionally category-aware and is not used
    globally. This prevents vague words such as "single" from accidentally
    selecting an unrelated service in a different workflow.
    """
    reply_tokens = meaningful_variant_tokens(
        value,
        category_tokens,
    )

    if not reply_tokens:
        return None

    matches = []

    for service in services:
        candidates = [
            service.get("service_name"),
            *(service.get("aliases") or []),
        ]
        best_score = 0

        for candidate in candidates:
            candidate_tokens = meaningful_variant_tokens(
                candidate,
                category_tokens,
            )

            if not candidate_tokens:
                continue

            if reply_tokens.issubset(candidate_tokens):
                score = (
                    1000
                    + len(reply_tokens) * 100
                    - len(candidate_tokens)
                )
                best_score = max(best_score, score)

        if best_score:
            matches.append((best_score, service))

    if not matches:
        return None

    matches.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    # Do not guess when two services are equally plausible.
    if (
        len(matches) > 1
        and matches[0][0] == matches[1][0]
    ):
        return None

    return matches[0][1]


def find_fixed_treatment_details(value: str | None) -> dict | None:
    cleaned = normalise_service_match_text(value)

    if not cleaned:
        return None

    padded_cleaned = f" {cleaned} "
    candidates = []

    for details in FIXED_TREATMENT_DETAILS:
        aliases = [details["name"], *details["aliases"]]

        for alias in aliases:
            cleaned_alias = normalise_service_match_text(alias)

            if cleaned_alias and f" {cleaned_alias} " in padded_cleaned:
                candidates.append((len(cleaned_alias), details))

    if not candidates:
        return None

    # Prefer the most specific match. For example, "stomach and love handles"
    # must win over the shorter alias "stomach".
    return max(candidates, key=lambda item: item[0])[1]


def apply_fixed_treatment_details(
    lead_data: dict,
    latest_message: str,
) -> dict:
    result = dict(lead_data)

    # The latest customer message wins when it clearly names a fixed service.
    # Otherwise, use the extracted or previously saved treatment.
    details = (
        find_fixed_treatment_details(latest_message)
        or find_fixed_treatment_details(result.get("treatment"))
    )

    if not details:
        return result

    result["treatment"] = details["name"]
    result["duration"] = details["duration"]
    result["status"] = calculate_lead_status(
        result,
        result.get("status"),
    )

    if result["status"] != "booking_request_complete":
        result["notification_sent_at"] = None

    return result


def is_valid_customer_name(value: str | None) -> bool:
    if value in [None, ""]:
        return False

    cleaned = str(value).strip()
    lower_value = cleaned.lower()

    rejected_phrases = {
        "i don't know",
        "i dont know",
        "dont know",
        "don't know",
        "unknown",
        "not sure",
        "no name",
        "none",
        "n/a",
        "na",
        "test",
    }

    if lower_value in rejected_phrases:
        return False

    if any(char.isdigit() for char in cleaned):
        return False

    words = cleaned.split()

    return (
        1 <= len(words) <= 5
        and all(
            re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ'-]+", word)
            for word in words
        )
    )


def is_valid_phone_number(value: str | None) -> bool:
    if value in [None, ""]:
        return False

    digits = re.sub(r"\D", "", str(value))

    return 10 <= len(digits) <= 15


def customer_message_interrupts_contact_collection(
    latest_message: str,
) -> bool:
    """
    Do not force name or phone validation when the customer is asking a side
    question, changing treatment, or clarifying a treatment variant.
    """
    cleaned = normalise_service_match_text(
        latest_message
    )

    if not cleaned:
        return False

    if "?" in str(latest_message or ""):
        return True

    interruption_phrases = [
        "actually",
        "instead",
        "change",
        "switch",
        "what",
        "which",
        "how much",
        "do you",
        "can i",
        "could i",
        "where",
    ]

    if any(
        phrase in cleaned
        for phrase in interruption_phrases
    ):
        return True

    if resolve_latest_generic_or_partial_service(
        latest_message
    ):
        return True

    return False


def validate_latest_customer_details(
    lead_data: dict,
    latest_message: str,
    previous_assistant_message: str,
    skip_contact_validation: bool = False,
) -> tuple[dict, str | None]:
    """
    Reject obviously invalid booking details before they complete a lead.
    """
    result = dict(lead_data)
    lower_previous = previous_assistant_message.lower()
    latest_duration = parse_duration_minutes(latest_message)
    allowed_durations = allowed_durations_for_treatment(
        result.get("treatment")
    )

    if latest_duration is not None and "long" in lower_previous:
        if allowed_durations and latest_duration not in allowed_durations:
            result["duration"] = None
            result["status"] = "details_incomplete"
            result["notification_sent_at"] = None

            return (
                result,
                "invalid_duration:"
                + format_duration_options(allowed_durations),
            )

    if not skip_contact_validation and "name" in lower_previous:
        if not is_valid_customer_name(result.get("name")):
            result["name"] = None
            result["status"] = "details_incomplete"
            result["notification_sent_at"] = None
            return result, "invalid_name"

    if (
        not skip_contact_validation
        and ("phone" in lower_previous or "number" in lower_previous)
    ):
        if not is_valid_phone_number(result.get("phone")):
            result["phone"] = None
            result["status"] = "details_incomplete"
            result["notification_sent_at"] = None
            return result, "invalid_phone"

    result["status"] = calculate_lead_status(
        result,
        result.get("status"),
    )

    if result["status"] != "booking_request_complete":
        result["notification_sent_at"] = None

    return result, None


def working_hours_for_date(
    requested_date,
) -> tuple[datetime, datetime] | None:
    hours = load_business_hours().get(requested_date.weekday())

    if not hours:
        return None

    open_value, close_value = hours

    if isinstance(open_value, tuple):
        open_hour, open_minute = open_value
    else:
        open_hour, open_minute = open_value, 0

    if isinstance(close_value, tuple):
        close_hour, close_minute = close_value
    else:
        close_hour, close_minute = close_value, 0

    open_time = datetime.combine(
        requested_date,
        datetime.min.time(),
        tzinfo=BUSINESS_TIMEZONE,
    ).replace(
        hour=open_hour,
        minute=open_minute,
    )

    close_time = datetime.combine(
        requested_date,
        datetime.min.time(),
        tzinfo=BUSINESS_TIMEZONE,
    ).replace(
        hour=close_hour,
        minute=close_minute,
    )

    return open_time, close_time


def business_can_accept_more_bookings_today(
    duration_minutes: int | None = None,
) -> bool:
    now = datetime.now(BUSINESS_TIMEZONE)
    opening_window = working_hours_for_date(now.date())

    if not opening_window:
        return False

    _, close_time = opening_window
    required_minutes = duration_minutes or SLOT_STEP_MINUTES

    return now + timedelta(minutes=required_minutes) <= close_time


def clear_expired_same_day_preference(
    lead_data: dict,
) -> tuple[dict, bool]:
    """
    If the customer selects today after the business has effectively closed,
    remove the unusable date and ask for another day instead of asking for a
    time that cannot be fulfilled.
    """
    result = dict(lead_data)
    today = datetime.now(BUSINESS_TIMEZONE).date().isoformat()

    if result.get("preferred_date") != today:
        return result, False

    duration_minutes = parse_duration_minutes(
        result.get("duration")
    )

    if business_can_accept_more_bookings_today(duration_minutes):
        return result, False

    result["preferred_date"] = None
    result["preferred_time"] = None
    result["notification_sent_at"] = None
    result["status"] = "details_incomplete"

    return result, True


def validate_slot_against_working_hours(
    start_time: datetime,
    end_time: datetime,
    duration_minutes: int | None,
) -> dict | None:
    opening_window = working_hours_for_date(start_time.date())

    if not opening_window:
        return {
            "status": "closed_day",
            "clear_date": True,
            "message": (
                f"{format_customer_date(start_time.date().isoformat())} is closed. "
                "Offer the next available alternatives and ask which one suits the customer."
            ),
        }

    open_time, close_time = opening_window

    if start_time < open_time or start_time >= close_time:
        return {
            "status": "outside_hours",
            "message": (
                f"{format_customer_slot(start_time.date().isoformat(), start_time.strftime('%H:%M'))} "
                f"is outside working hours. Opening hours that day are "
                f"{open_time.strftime('%H:%M')}–{close_time.strftime('%H:%M')}. "
                "Ask the customer for another preferred time."
            ),
        }

    if duration_minutes and end_time > close_time:
        return {
            "status": "outside_hours",
            "message": (
                f"That session would finish after closing time. Opening hours that day are "
                f"{open_time.strftime('%H:%M')}–{close_time.strftime('%H:%M')}. "
                "Ask the customer for another preferred time."
            ),
        }

    return None


def preserve_existing_booking_details(
    existing: dict,
    merged: dict,
    latest_message: str,
    previous_assistant_message: str,
) -> dict:
    """
    Keep previously confirmed lead values stable unless the latest customer
    message explicitly changes them. This prevents casual follow-up messages
    such as "Fantastic" from accidentally changing the requested date or time.
    """
    result = dict(merged)
    lower_latest = latest_message.lower()
    lower_previous = previous_assistant_message.lower()

    explicit_date = parse_relative_date_from_text(
        latest_message,
        reference_date=existing.get("preferred_date"),
    )
    explicit_time = parse_time_from_text(latest_message)
    explicit_phone = parse_phone_from_text(latest_message)
    explicit_duration = parse_duration_minutes(latest_message)

    if existing.get("preferred_date") not in [None, ""] and not explicit_date:
        result["preferred_date"] = existing.get("preferred_date")

    if existing.get("preferred_time") not in [None, ""] and not explicit_time:
        result["preferred_time"] = existing.get("preferred_time")

    duration_is_explicit_reply = (
        explicit_duration is not None
        and (
            "long" in lower_previous
            or "duration" in lower_previous
            or re.search(r"\b(?:minute|minutes|min|mins|hour|hours|hr|hrs)\b", lower_latest)
        )
    )

    if existing.get("duration") not in [None, ""] and not duration_is_explicit_reply:
        result["duration"] = existing.get("duration")

    if existing.get("phone") not in [None, ""] and not explicit_phone:
        result["phone"] = existing.get("phone")

    if existing.get("name") not in [None, ""]:
        name_is_explicit_reply = (
            "name" in lower_previous
            and looks_like_customer_name(latest_message)
        )

        if not name_is_explicit_reply:
            result["name"] = existing.get("name")

    existing_treatment = existing.get("treatment")

    if existing_treatment not in [None, ""]:
        treatment_keywords = [
            "massage",
            "facial",
            "b12",
            "vitamin",
            "filler",
            "microneedling",
            "hydrafacial",
            "sculpting",
            "ultrasound",
            "ems",
            "injection",
        ]

        if not any(keyword in lower_latest for keyword in treatment_keywords):
            result["treatment"] = existing_treatment

    result["status"] = calculate_lead_status(
        result,
        result.get("status"),
    )

    if result["status"] != "booking_request_complete":
        result["notification_sent_at"] = None

    return result


def normalise_reply_line_breaks(reply: str) -> str:
    """Store and send real line breaks instead of visible backslash-n text."""
    return str(reply or "").replace("\\r\\n", "\n").replace("\\n", "\n")


def remove_invalid_fixed_duration_questions(
    reply: str,
    lead_data: dict,
) -> str:
    """
    A fixed-duration treatment must never ask the customer to choose a session
    length. Remove any duration-selection question accidentally generated by
    the language model. The deterministic booking flow will then append the
    correct next missing question, such as the customer's name.
    """
    details = find_fixed_treatment_details(
        lead_data.get("treatment")
    )

    if (
        not details
        and not is_dynamic_fixed_duration_treatment(
            lead_data.get("treatment")
        )
        and not is_dynamic_ultrasound_fixed_duration(
            lead_data.get("treatment")
        )
        and not is_dynamic_body_treatment_fixed_duration(
            lead_data.get("treatment")
        )
        and not is_dynamic_microneedling_fixed_duration(
            lead_data.get("treatment")
        )
        and not is_dynamic_facial_fixed_duration(
            lead_data.get("treatment")
        )
        and not is_dynamic_dermal_filler_fixed_duration(
            lead_data.get("treatment")
        )
    ) or lead_data.get("duration") in [None, ""]:
        return reply

    cleaned = str(reply or "")

    patterns = [
        (
            r"(?is)(?:\s*\n\s*)?"
            r"(?:how long|what duration|which duration)"
            r"[^?]*\?"
        ),
        (
            r"(?is)(?:\s*\n\s*)?"
            r"how long would you like the session for"
            r"[^?]*\?"
        ),
        (
            r"(?is)(?:\s*\n\s*)?"
            r"please choose (?:one of )?(?:the )?(?:available )?"
            r"(?:durations?|session lengths?)[^?.]*[?.]"
        ),
    ]

    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)

    return cleaned.strip()


def remove_premature_booking_questions(
    reply: str,
    booking_flow_active: bool,
) -> str:
    """
    Informational enquiries must not immediately turn into a booking form.
    Remove requests for personal details until the customer actually indicates
    that they want to arrange an appointment.
    """
    if booking_flow_active:
        return reply

    cleaned = str(reply or "")

    patterns = [
        r"(?is)(?:\s*\n\s*)?(?:may|can|could|would) i have your name[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:what(?:'s| is)|may i have|can i have|could you provide) your (?:phone|contact|mobile) number[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:which|what) (?:day|date) would you prefer[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:which|what) time(?: today)? would you prefer[^?]*\?",
        r"(?is)(?:\s*\n\s*)?how long would you like the session for[^?]*\?",
    ]

    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)

    return cleaned.strip()


def remove_invalid_today_time_questions(
    reply: str,
    today_is_unavailable: bool,
) -> str:
    if not today_is_unavailable:
        return reply

    cleaned = str(reply or "")

    patterns = [
        r"(?is)(?:\s*\n\s*)?(?:which|what) time[^?]*\btoday\b[^?]*\?",
        r"(?is)(?:\s*\n\s*)?what time would you like today[^?]*\?",
        r"(?is)(?:\s*\n\s*)?what time today would you prefer[^?]*\?",
    ]

    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)

    return cleaned.strip()


def remove_redundant_confirmed_field_questions(
    reply: str,
    lead_data: dict,
) -> str:
    """
    Do not ask for a date or time that the customer already supplied and the
    backend accepted.
    """
    cleaned = str(reply or "")

    if lead_data.get("preferred_time") not in [None, ""]:
        cleaned = re.sub(
            r"(?is)(?:\s*\n\s*)?(?:which|what) time[^?]*\?",
            "",
            cleaned,
        )

    if lead_data.get("preferred_date") not in [None, ""]:
        cleaned = re.sub(
            r"(?is)(?:\s*\n\s*)?(?:which|what) (?:day|date)[^?]*\?",
            "",
            cleaned,
        )

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)

    return cleaned.strip()


def sanitise_booking_claims(reply: str) -> str:
    """Stop the AI from accidentally claiming that a request is confirmed."""
    cleaned = reply

    replacements = [
        (r"(?i)\bi have you booked in for\b", "I've noted your request for"),
        (r"(?i)\bi have you booked for\b", "I've noted your request for"),
        (r"(?i)\\byou(?:'re| are) booked in for\\b", "I've noted your request for"),
        (r"(?i)\\byou(?:'re| are) booked for\\b", "I've noted your request for"),
        (r"(?i)\byour booking is confirmed\b", "Your request has been noted. The therapist still needs to confirm the appointment"),
        (r"(?i)\byour appointment is confirmed\b", "Your request has been noted. The therapist still needs to confirm the appointment"),
    ]

    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned)

    return cleaned


def canonical_missing_question(
    field: str,
    lead_data: dict | None = None,
) -> str:
    if field == "duration":
        allowed_durations = allowed_durations_for_treatment(
            (lead_data or {}).get("treatment")
        )

        if allowed_durations:
            options = format_duration_options(allowed_durations)
            return f"How long would you like the session for: {options}?"

        return "Which treatment option would you like?"

    if field == "preferred_date":
        known_time = (lead_data or {}).get("preferred_time")

        if known_time not in [None, ""]:
            return (
                f"Which day did you have in mind for "
                f"{format_customer_time(known_time)}?"
            )

        if not business_can_accept_more_bookings_today(
            parse_duration_minutes((lead_data or {}).get("duration"))
        ):
            return "Which day would you prefer? We're closed for today."

        return "Which day would you prefer?"

    if field == "preferred_time":
        preferred_date = (lead_data or {}).get("preferred_date")
        today = datetime.now(BUSINESS_TIMEZONE).date().isoformat()

        if (
            preferred_date == today
            and not business_can_accept_more_bookings_today(
                parse_duration_minutes((lead_data or {}).get("duration"))
            )
        ):
            return "We're closed for today. Which other day would you prefer?"

        return "What time would you prefer?"

    questions = {
        "treatment": "Which treatment would you like?",
        "name": "What's your name?",
        "phone": "What's your phone number?",
    }

    return questions.get(field, "Could you provide the next booking detail?")


def reply_already_asks_for_field(reply: str, field: str) -> bool:
    lower_reply = reply.lower()

    markers = {
        "treatment": ["which treatment", "what treatment", "which service"],
        "duration": ["how long", "duration", "how many minutes"],
        "name": ["your name", "what's your name", "what is your name"],
        "phone": ["phone number", "contact number", "mobile number"],
        "preferred_date": ["which day", "what day", "which date", "what date"],
        "preferred_time": ["what time", "which time", "preferred time"],
    }

    return any(marker in lower_reply for marker in markers.get(field, []))


def grouped_missing_question(
    missing_fields: list[str],
    lead_data: dict,
) -> str:
    """
    Ask for up to two closely related details in one natural question.
    This keeps the flow efficient without turning the chat into a long form.
    """
    missing = set(missing_fields)

    if "duration" in missing and "preferred_date" in missing:
        duration_question = canonical_missing_question(
            "duration",
            lead_data,
        ).rstrip("?")

        return f"{duration_question}, and which day would you prefer?"

    if "duration" in missing and "preferred_time" in missing:
        duration_question = canonical_missing_question(
            "duration",
            lead_data,
        ).rstrip("?")

        return f"{duration_question}, and what time would suit you?"

    if (
        "preferred_date" in missing
        and "preferred_time" in missing
        and "duration" not in missing
    ):
        return "Which day and time would you prefer?"

    if "name" in missing and "phone" in missing:
        return "Could I take your name and phone number, please?"

    return canonical_missing_question(
        missing_fields[0],
        lead_data,
    )


def remove_false_handover_confirmation(
    reply: str,
    missing_fields: list[str],
) -> str:
    """
    Never tell the customer that the therapist will confirm shortly while
    required booking details are still missing.
    """
    if not missing_fields:
        return reply

    cleaned = str(reply or "")

    patterns = [
        r"(?is)(?:\s*\n\s*)?thanks[.!]?\s*the therapist will confirm the appointment shortly[.!]?",
        r"(?is)(?:\s*\n\s*)?the therapist will confirm the appointment shortly[.!]?",
        r"(?is)(?:\s*\n\s*)?your request has been noted[.!]?",
        r"(?is)(?:\s*\n\s*)?the therapist still needs to confirm(?: the booking| the appointment)?[.!]?",
        r"(?is)(?:\s*\n\s*)?i(?:'ve| have) noted your request[^.!?]*[.!]?",
    ]

    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)

    return cleaned.strip()


def format_natural_list(items: list[str]) -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]

    if not cleaned:
        return ""

    if len(cleaned) == 1:
        return cleaned[0]

    if len(cleaned) == 2:
        return f"{cleaned[0]} or {cleaned[1]}"

    return ", ".join(cleaned[:-1]) + f", or {cleaned[-1]}"


def build_duration_needed_for_alternatives_reply(
    lead_data: dict,
    calendar_result: dict,
) -> str:
    """
    Ask for duration before listing availability. This prevents vague or
    misleading statements such as "any time before closing could work".
    """
    allowed_durations = allowed_durations_for_treatment(
        lead_data.get("treatment")
    )

    if allowed_durations:
        options = format_duration_options(allowed_durations)
        duration_question = (
            f"How long would you like the session for: {options}?"
        )
    else:
        duration_question = "How long would you like the session for?"

    requested_slot = calendar_result.get("customer_slot")

    if requested_slot:
        opening = f"{requested_slot} is unavailable."
    else:
        opening = "I can check the available slots for you."

    return (
        f"{opening}\n\n"
        f"{duration_question} Once you tell me, I can check the nearest "
        "available times before and after your requested slot."
    )


def build_direct_calendar_suggestions_reply(
    calendar_result: dict,
) -> str:
    """
    Render direct availability suggestions in Python.

    The LLM must not rewrite weekday/date combinations because that can create
    impossible wording such as "Thursday 9 June".
    """
    suggestions = calendar_result.get(
        "suggestions"
    ) or []

    if not suggestions:
        return ""

    options = format_natural_list(
        suggestions
    )

    return (
        f"The next available options are {options}.\n\n"
        "Would any of those times work for you?"
    )


def build_calendar_alternative_reply(
    calendar_result: dict,
) -> str:
    """
    Render alternative availability directly in Python.
    Do not let the LLM continue collecting contact details until the customer
    chooses one of the proposed replacement slots.
    """
    suggestions = calendar_result.get("suggestions") or []

    if not suggestions:
        return ""

    options = format_natural_list(suggestions)
    status = calendar_result.get("status")

    if status == "closed_day":
        opening = "That day is closed."
    elif status == "past":
        opening = "That time has already passed."
    else:
        opening = "That time is unavailable."

    return (
        f"{opening} The next available options are {options}.\n\n"
        "Would any of those times work for you?"
    )


def ensure_calendar_alternative_question(
    reply: str,
    calendar_result: dict,
) -> str:
    """
    If the backend supplied alternative free slots, ask the customer to pick
    one rather than ending the conversation or asking unrelated questions.
    """
    if not calendar_result.get("suggestions"):
        return reply

    lower_reply = str(reply or "").lower()

    if any(
        phrase in lower_reply
        for phrase in [
            "would one of those suit",
            "would any of those suit",
            "would any of these suit",
            "which of those",
            "which option",
        ]
    ):
        return reply

    question = "Would any of those times work for you?"

    if not str(reply or "").strip():
        return question

    return str(reply).rstrip() + "\n\n" + question


def remove_generated_booking_detail_questions(reply: str) -> str:
    """
    Strip any booking-detail question produced by the model. The backend adds
    one clean deterministic grouped question afterwards, preventing repeated
    or out-of-order questions.
    """
    cleaned = str(reply or "")

    patterns = [
        r"(?is)(?:\s*\n\s*)?(?:which|what) treatment[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:how long|what duration|which duration)[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:may|can|could|would) i (?:have|take) your name[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:what(?:'s| is)|may i have|can i have|could you provide|could i take) your (?:phone|contact|mobile) number[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:which|what) (?:day|date)[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:which|what) time[^?]*\?",
    ]

    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)

    return cleaned.strip()


def remove_contact_detail_questions(reply: str) -> str:
    """
    Remove any personal-detail question generated before the schedule has been
    settled. Contact details are collected only after an acceptable date and
    time exist.
    """
    cleaned = str(reply or "")

    patterns = [
        r"(?is)(?:\s*\n\s*)?(?:may|can|could|would) i (?:have|take) your name(?: and (?:phone|contact|mobile) number)?[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:what(?:'s| is)|may i have|can i have|could you provide|could i take) your (?:phone|contact|mobile) number[^?]*\?",
        r"(?is)(?:\s*\n\s*)?could i take your name and (?:phone|contact|mobile) number[^?]*\?",
    ]

    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)

    return cleaned.strip()


def build_schedule_first_question(lead_data: dict) -> str:
    """
    Ask only for the unsettled schedule detail. This is deliberately separate
    from the LLM so the booking order cannot drift.
    """
    duration = lead_data.get("duration")
    preferred_date = lead_data.get("preferred_date")
    preferred_time = lead_data.get("preferred_time")

    if needs_vitamin_shot_variant(lead_data):
        return (
            "B12 and Vitamin C shots are both £20 and take 15 minutes. "
            "Which one would you like?"
        )

    if needs_ultrasound_variant(lead_data):
        return (
            "A single ultrasound session is £50 for 45 minutes. "
            "The package is £220 for 5 sessions, with each session lasting "
            "45 minutes. Which option would you like?"
        )

    if needs_microneedling_variant(lead_data):
        return (
            "Microneedling for stretch marks is £60 and microneedling for "
            "the face and neck is £70. Both take 1 hour 15 minutes. "
            "Which option would you like?"
        )

    if needs_facial_variant(lead_data):
        return (
            "We offer Deep Facial Cleanse (£60, 60 minutes), Radio Frequency "
            "Facial (£60, 60 minutes), Facial Treatment for Young Skin "
            "(£45, 60 minutes), Hydraface Facial Treatment (£60, 60 minutes), "
            "and Hydrafacial (£80, 90 minutes). Which facial would you like?"
        )

    if needs_dermal_filler_variant(lead_data):
        return build_dermal_filler_variant_prompt(
            lead_data
        )

    ultrasound_service = find_ultrasound_service(
        lead_data.get("treatment")
    )

    if (
        ultrasound_service
        and ultrasound_service.get("booking_mode") == "package"
        and preferred_date in [None, ""]
        and preferred_time in [None, ""]
    ):
        return (
            "Which day and time would suit you for the first ultrasound "
            "session?"
        )

    if duration in [None, ""]:
        allowed_durations = allowed_durations_for_treatment(
            lead_data.get("treatment")
        )

        if allowed_durations:
            options = format_duration_options(allowed_durations)

            if preferred_date in [None, ""]:
                return (
                    f"How long would you like the session for: {options}, "
                    "and which day would you prefer?"
                )

            return f"How long would you like the session for: {options}?"

        return "How long would you like the session for?"

    if preferred_date in [None, ""] and preferred_time in [None, ""]:
        return "Which day and time would suit you?"

    if preferred_date in [None, ""]:
        return (
            f"Which day did you have in mind for "
            f"{format_customer_time(preferred_time)}?"
        )

    if preferred_time in [None, ""]:
        return (
            f"What time would suit you on "
            f"{format_customer_date(preferred_date)}?"
        )

    return ""


def enforce_schedule_before_contact(
    reply: str,
    lead_data: dict,
    calendar_result: dict,
) -> str:
    """
    Hard booking-order guardrail:
    - do not ask for name or phone until date and time are settled;
    - do not override a proper alternatives reply;
    - append exactly one schedule question when needed.
    """
    if calendar_result.get("suggestions"):
        return reply

    schedule_question = build_schedule_first_question(
        lead_data
    )

    if not schedule_question:
        return reply

    cleaned = remove_contact_detail_questions(
        reply
    )

    cleaned = re.sub(
        r"(?is)(?:\s*\n\s*)?(?:which|what) (?:day|date|time)[^?]*\?",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()

    if not cleaned:
        return schedule_question

    return cleaned.rstrip() + "\n\n" + schedule_question


def strip_model_booking_questions(reply: str) -> str:
    """
    Remove only workflow questions generated by the LLM.

    Preserve factual sentences such as prices and durations. The backend then
    appends exactly one deterministic follow-up question.
    """
    cleaned = str(reply or "")

    question_patterns = [
        r"(?is)(?:\s*\n\s*)?(?:how long|which duration|what duration)[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:which|what) (?:day|date|time)[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:could|can|may|would) i (?:take|have) your (?:name|phone number|contact number|mobile number)[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:what(?:'s| is)) your (?:name|phone number|contact number|mobile number)[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:which treatment|what treatment|which service|which treatment option)[^?]*\?",
    ]

    for pattern in question_patterns:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"(?im)^\s*(?:and|or|,\s*and|—|-)+\s*$", "", cleaned)
    cleaned = re.sub(r"(?i)(?:,|\s)\s*and\s*$", "", cleaned.strip())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned.strip()


def customer_asks_about_price(message: str) -> bool:
    cleaned = normalise_service_match_text(message)

    return any(
        phrase in cleaned
        for phrase in [
            "how much",
            "price",
            "prices",
            "cost",
            "costs",
            "what are they",
        ]
    )


def format_service_price_sentence(service: dict | None) -> str:
    if not service:
        return ""

    name = str(service.get("service_name") or "").strip()
    price = format_price_pence(service.get("price_pence"))
    duration = format_minutes_as_duration(
        service.get("fixed_duration_minutes")
    )
    mode = service.get("booking_mode")

    if not name or not price:
        return ""

    if mode == "package":
        return (
            f"The ultrasound package is {price} for 5 sessions. "
            "Each session lasts 45 minutes."
        )

    if duration:
        return f"{name} is {price} and takes {duration}."

    return f"{name} is {price}."


def build_structured_price_information(
    latest_message: str,
    lead_data: dict,
) -> str:
    """
    Return a deterministic fixed-price answer when the customer asks about a
    structured vitamin-shot or ultrasound service.
    """
    if not customer_asks_about_price(latest_message):
        return ""

    if needs_ultrasound_variant(lead_data):
        return (
            "A single ultrasound session is £50 for 45 minutes. "
            "The package is £220 for 5 sessions, with each session lasting "
            "45 minutes."
        )

    ultrasound_service = find_ultrasound_service(
        lead_data.get("treatment")
    )

    if ultrasound_service:
        return format_service_price_sentence(
            ultrasound_service
        )

    if needs_vitamin_shot_variant(lead_data):
        return (
            "B12 and Vitamin C shots are both £20 and take 15 minutes."
        )

    vitamin_service = find_vitamin_shot_service(
        lead_data.get("treatment")
    )

    if vitamin_service:
        return format_service_price_sentence(
            vitamin_service
        )

    return ""


def prepend_information_once(reply: str, information: str) -> str:
    information = str(information or "").strip()
    reply = str(reply or "").strip()

    if not information:
        return reply

    if information.lower() in reply.lower():
        return reply

    if not reply:
        return information

    return information + "\n\n" + reply


def latest_message_selects_ultrasound_variant(
    message: str,
) -> bool:
    cleaned = normalise_service_match_text(message)

    if find_ultrasound_service(message):
        return True

    return cleaned in {
        "single",
        "single session",
        "one session",
        "just one",
        "package",
        "the package",
        "package deal",
        "the package deal",
        "course",
    }


def latest_message_selects_dermal_filler_service(
    message: str,
) -> bool:
    return bool(
        find_dermal_filler_service(message)
    )


def latest_message_selects_facial_service(
    message: str,
) -> bool:
    return bool(
        find_facial_service(message)
    )


def latest_message_selects_microneedling_variant(
    message: str,
) -> bool:
    return bool(
        find_microneedling_service(message)
    )


def latest_message_selects_body_treatment(
    message: str,
) -> bool:
    return bool(
        find_body_treatment_service(message)
    )


def latest_message_selects_vitamin_variant(
    message: str,
) -> bool:
    service = find_vitamin_shot_service(message)

    return bool(
        service
        and service.get("booking_mode") == "fixed_duration"
    )


def strip_structured_service_price_sentences(
    reply: str,
) -> str:
    """
    Remove only old LLM-written price sentences for dynamic categories.

    A deterministic price sentence is inserted afterwards. This avoids
    semantically duplicated wording such as:
      "The package is £220 for five 45-minute sessions."
      "The ultrasound package is £220 for 5 sessions."
    """
    segments = re.split(
        r"(?<=[.!?])\s+|\n+",
        str(reply or "").strip(),
    )

    kept = []

    for segment in segments:
        cleaned = segment.strip()

        if not cleaned:
            continue

        lower_segment = cleaned.lower()

        is_dynamic_service_price = (
            "£" in cleaned
            and any(
                marker in lower_segment
                for marker in [
                    "ultrasound",
                    "package",
                    "vitamin",
                    "b12",
                    "shot",
                    "body sculpting",
                    "ems",
                    "muscle stimulation",
                    "microneedling",
                    "stretch marks",
                    "face and neck",
                    "facial",
                    "hydraface",
                    "hydrafacial",
                    "dermal filler",
                    "lip filler",
                    "marionette",
                    "nasolabial",
                ]
            )
        )

        if not is_dynamic_service_price:
            kept.append(cleaned)

    return "\n\n".join(kept).strip()


def finalise_dynamic_service_reply(
    reply: str,
    latest_message: str,
    lead_data: dict,
    missing_fields: list[str],
    calendar_result: dict,
) -> str:
    """
    Use one deterministic answer for migrated fixed-price categories.

    This prevents the LLM and backend from both listing the same information
    in slightly different wording.
    """
    if needs_ultrasound_variant(lead_data):
        return build_schedule_first_question(
            lead_data
        )

    if needs_vitamin_shot_variant(lead_data):
        return build_schedule_first_question(
            lead_data
        )

    if needs_microneedling_variant(lead_data):
        return build_schedule_first_question(
            lead_data
        )

    if needs_facial_variant(lead_data):
        return build_schedule_first_question(
            lead_data
        )

    if needs_dermal_filler_variant(lead_data):
        return build_dermal_filler_variant_prompt(
            lead_data,
            latest_message,
        )

    ultrasound_service = find_ultrasound_service(
        lead_data.get("treatment")
    )
    vitamin_service = find_vitamin_shot_service(
        lead_data.get("treatment")
    )
    body_treatment_service = find_body_treatment_service(
        lead_data.get("treatment")
    )
    microneedling_service = find_microneedling_service(
        lead_data.get("treatment")
    )
    facial_service = find_facial_service(
        lead_data.get("treatment")
    )
    dermal_filler_service = find_dermal_filler_service(
        lead_data.get("treatment")
    )

    service = (
        ultrasound_service
        or vitamin_service
        or body_treatment_service
        or microneedling_service
        or facial_service
        or dermal_filler_service
    )

    if not service:
        return reply

    announce_price = (
        customer_asks_about_price(latest_message)
        or latest_message_selects_ultrasound_variant(
            latest_message
        )
        or latest_message_selects_vitamin_variant(
            latest_message
        )
        or latest_message_selects_body_treatment(
            latest_message
        )
        or latest_message_selects_microneedling_variant(
            latest_message
        )
        or latest_message_selects_facial_service(
            latest_message
        )
        or latest_message_selects_dermal_filler_service(
            latest_message
        )
    )

    if not announce_price:
        return reply

    information = format_service_price_sentence(
        service
    )

    if not information:
        return reply

    cleaned_reply = strip_structured_service_price_sentences(
        reply
    )

    if calendar_result.get("status") in {
        "not_checked",
        "not_requested",
        "provisional_free",
    }:
        question = build_deterministic_next_question(
            lead_data,
            missing_fields,
        )

        if question:
            return information + "\n\n" + question

        return information

    if not cleaned_reply:
        return information

    return information + "\n\n" + cleaned_reply


def format_service_switch_summary(
    lead_data: dict,
) -> str:
    """
    Build one clear customer-facing summary for the newly selected service.

    Use the structured Supabase row so fixed-price services and
    duration-specific services are both handled consistently.
    """
    service_row = load_service_row_for_booking_item(
        lead_data.get("treatment")
    )

    if not service_row:
        return str(
            lead_data.get("treatment") or "the new treatment"
        )

    service_name = str(
        service_row.get("service_name")
        or lead_data.get("treatment")
        or "the new treatment"
    ).strip()

    duration_minutes = booking_item_duration_minutes(
        lead_data,
        service_row,
    )
    price_pence = booking_item_price_pence(
        service_row,
        duration_minutes,
    )

    details = []

    if price_pence is not None:
        details.append(
            format_price_pence(price_pence)
        )

    duration_label = format_minutes_as_duration(
        duration_minutes
    )

    if duration_label:
        details.append(duration_label)

    if not details:
        return service_name

    return service_name + " — " + ", ".join(details)


def build_service_switch_reply(
    lead_data: dict,
    missing_fields: list[str],
    calendar_result: dict,
) -> str:
    """
    Reply deterministically after a treatment switch.

    The customer may change their mind at any point, including when the
    previous chatbot message asked for a name or phone number. Confirm the new
    request clearly, show updated service details, then continue from the next
    genuinely missing field.
    """
    if is_partial_dermal_filler_treatment(
        lead_data.get("treatment")
    ):
        area = dermal_filler_partial_area(
            lead_data.get("treatment")
        )
        label = DERMAL_FILLER_AREA_LABELS.get(
            area,
            "dermal filler",
        )
        opening = (
            "No problem. I've changed the request to "
            f"{label.lower()}."
        )
    else:
        summary = format_service_switch_summary(
            lead_data
        )
        opening = (
            "No problem. I've updated your request to "
            f"{summary}."
        )

    status = str(
        calendar_result.get("status") or "not_checked"
    )

    if status in {
        "busy_needs_duration",
        "needs_duration_for_alternatives",
    }:
        return (
            opening
            + "\n\n"
            + build_duration_needed_for_alternatives_reply(
                lead_data,
                calendar_result,
            )
        )

    if calendar_result.get("suggestions"):
        return (
            opening
            + "\n\n"
            + build_calendar_alternative_reply(
                calendar_result,
            )
        )

    if status == "free":
        slot = format_customer_slot(
            lead_data.get("preferred_date"),
            lead_data.get("preferred_time"),
        )

        if missing_fields:
            question = build_deterministic_next_question(
                lead_data,
                missing_fields,
            )

            return (
                opening
                + f"\n\n{slot} still appears free."
                + (f"\n\n{question}" if question else "")
            )

        return (
            opening
            + f"\n\n{slot} still appears free."
            + "\n\nYour updated request has been recorded. "
            "The therapist will confirm it shortly."
        )

    if status == "unknown":
        if missing_fields:
            question = build_deterministic_next_question(
                lead_data,
                missing_fields,
            )

            return (
                opening
                + "\n\nThe therapist will need to check the availability."
                + (f"\n\n{question}" if question else "")
            )

        return (
            opening
            + "\n\nThe therapist will check the availability and confirm "
            "the updated request shortly."
        )

    question = build_deterministic_next_question(
        lead_data,
        missing_fields,
    )

    if question:
        return opening + "\n\n" + question

    return (
        opening
        + "\n\nYour updated request has been recorded. "
        "The therapist will confirm it shortly."
    )


def build_deterministic_next_question(
    lead_data: dict,
    missing_fields: list[str],
) -> str:
    """
    Booking order is controlled by Python:
      duration -> date -> time -> name and phone.
    """
    schedule_question = build_schedule_first_question(
        lead_data
    )

    if schedule_question:
        return schedule_question

    if not missing_fields:
        return ""

    return grouped_missing_question(
        missing_fields,
        lead_data,
    )


def append_single_deterministic_followup(
    reply: str,
    lead_data: dict,
    missing_fields: list[str],
) -> str:
    """
    Keep any useful factual prefix from the LLM, remove its workflow questions,
    then append exactly one backend-controlled follow-up question.
    """
    question = build_deterministic_next_question(
        lead_data,
        missing_fields,
    )

    cleaned = strip_model_booking_questions(
        reply
    )

    if not question:
        return cleaned

    if not cleaned:
        return question

    return cleaned.rstrip() + "\n\n" + question


def ensure_next_booking_question(
    reply: str,
    missing_fields: list[str],
    lead_data: dict,
) -> str:
    """
    Add exactly one clean follow-up question after the model answers.
    """
    if not missing_fields:
        return reply

    question = grouped_missing_question(
        missing_fields,
        lead_data,
    )

    if not reply.strip():
        return question

    return reply.rstrip() + "\n\n" + question


def load_service_row_for_booking_item(
    treatment: str | None,
) -> dict | None:
    """
    Load the structured public.services row used for a booking item.

    Use the database row when available so the shadow booking item remains
    linked to owner-editable Supabase configuration. If the lookup fails,
    retain a safe minimal fallback from the structured identity.
    """
    identity = structured_service_identity(
        treatment
    )

    if not identity:
        return None

    service_name = identity.get("service_name")

    if not service_name:
        return None

    try:
        response = (
            supabase.table("services")
            .select(
                "id, category, service_name, booking_mode, "
                "fixed_duration_minutes, allowed_durations_minutes, "
                "price_pence, price_by_duration"
            )
            .eq("business_slug", BUSINESS_SLUG)
            .eq("service_name", service_name)
            .limit(1)
            .execute()
        )

        if response.data:
            return response.data[0]

    except Exception as error:
        print(f"Could not load service row for booking item: {error}")

    return {
        "id": None,
        "category": identity.get("category"),
        "service_name": service_name,
        "booking_mode": identity.get("booking_mode"),
        "fixed_duration_minutes": identity.get(
            "fixed_duration_minutes"
        ),
        "allowed_durations_minutes": identity.get(
            "allowed_durations_minutes"
        ),
        "price_pence": None,
        "price_by_duration": None,
    }


def booking_item_duration_minutes(
    lead_data: dict,
    service_row: dict,
) -> int | None:
    """
    Resolve one authoritative duration for the active booking item.

    Fixed-duration services use their structured Supabase value. Services such
    as massage use the customer's selected duration.
    """
    fixed_duration = service_row.get(
        "fixed_duration_minutes"
    )

    if fixed_duration not in [None, ""]:
        try:
            return int(fixed_duration)
        except (TypeError, ValueError):
            pass

    return parse_duration_minutes(
        lead_data.get("duration")
    )


def booking_item_price_pence(
    service_row: dict,
    duration_minutes: int | None,
) -> int | None:
    """
    Resolve the configured fixed price or duration-specific price.
    """
    fixed_price = service_row.get("price_pence")

    if fixed_price not in [None, ""]:
        try:
            return int(fixed_price)
        except (TypeError, ValueError):
            return None

    prices = service_row.get("price_by_duration") or {}

    if duration_minutes is None:
        return None

    value = prices.get(
        str(duration_minutes)
    )

    if value in [None, ""]:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_active_draft_booking_request(
    session_id: str,
) -> dict | None:
    try:
        response = (
            supabase.table("booking_requests")
            .select("*")
            .eq("session_id", session_id)
            .eq("request_status", "draft")
            .limit(1)
            .execute()
        )

        if response.data:
            return response.data[0]

    except Exception as error:
        print(f"Could not load active draft booking request: {error}")

    return None


def next_booking_request_number(
    session_id: str,
) -> int:
    try:
        response = (
            supabase.table("booking_requests")
            .select("request_number")
            .eq("session_id", session_id)
            .order("request_number", desc=True)
            .limit(1)
            .execute()
        )

        if response.data:
            return int(
                response.data[0].get("request_number") or 0
            ) + 1

    except Exception as error:
        print(f"Could not calculate next booking request number: {error}")

    return 1


def update_conversation_active_booking_request(
    session_id: str,
    booking_request_id: int,
):
    try:
        (
            supabase.table("conversations")
            .update({
                "active_booking_request_id": booking_request_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            .eq("session_id", session_id)
            .execute()
        )

    except Exception as error:
        print(f"Could not save active booking request pointer: {error}")


def load_booking_items(
    booking_request_id: int,
) -> list[dict]:
    try:
        response = (
            supabase.table("booking_items")
            .select("*")
            .eq("booking_request_id", booking_request_id)
            .order("item_order")
            .execute()
        )

        return list(response.data or [])

    except Exception as error:
        print(f"Could not load booking items: {error}")
        return []


def latest_message_explicitly_adds_same_visit(
    latest_message: str,
) -> bool:
    cleaned = normalise_service_match_text(
        latest_message
    )

    same_visit_phrases = [
        "same visit",
        "same appointment",
        "back to back",
        "one after another",
        "straight after",
        "after that",
        "add it to",
        "add this to",
    ]

    return any(
        phrase in cleaned
        for phrase in same_visit_phrases
    )


def latest_message_suggests_additional_treatment(
    latest_message: str,
) -> bool:
    cleaned = normalise_service_match_text(
        latest_message
    )

    addition_phrases = [
        "also",
        "add",
        "another treatment",
        "as well",
        "too",
    ]

    replacement_phrases = [
        "instead",
        "change",
        "switch",
        "replace",
        "rather than",
    ]

    return (
        any(phrase in cleaned for phrase in addition_phrases)
        and not any(
            phrase in cleaned
            for phrase in replacement_phrases
        )
    )


def load_pending_booking_actions(
    session_id: str,
) -> list[dict]:
    try:
        response = (
            supabase.table("pending_booking_actions")
            .select("*")
            .eq("session_id", session_id)
            .eq("action_status", "pending")
            .order("created_at")
            .execute()
        )

        return list(response.data or [])

    except Exception as error:
        print(f"Could not load pending booking actions: {error}")
        return []


def save_pending_booking_action(
    session_id: str,
    selected_identity: dict,
    source_message: str,
) -> dict | None:
    """
    Save one unresolved cart question safely.

    Do not use PostgREST upsert here. The database uniqueness rule is a
    partial unique index that applies only while action_status='pending'.
    PostgreSQL cannot infer that partial index from the generic
    ON CONFLICT(session_id, service_name, action_type) clause generated by
    Supabase upsert. Check for an existing pending row first, then update or
    insert explicitly.
    """
    service_name = str(
        selected_identity.get("service_name") or ""
    ).strip()

    if not service_name:
        return None

    service_row = load_service_row_for_booking_item(
        service_name
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    payload = {
        "session_id": session_id,
        "service_id": (
            service_row.get("id")
            if service_row
            else None
        ),
        "service_name": service_name,
        "category": (
            selected_identity.get("category")
            or (
                service_row.get("category")
                if service_row
                else None
            )
        ),
        "action_type": "choose_appointment_structure",
        "action_status": "pending",
        "source_message": source_message,
        "updated_at": now_iso,
    }

    try:
        existing_response = (
            supabase.table("pending_booking_actions")
            .select("id")
            .eq("session_id", session_id)
            .eq("service_name", service_name)
            .eq(
                "action_type",
                "choose_appointment_structure",
            )
            .eq("action_status", "pending")
            .limit(1)
            .execute()
        )

        if existing_response.data:
            action_id = int(
                existing_response.data[0]["id"]
            )

            response = (
                supabase.table("pending_booking_actions")
                .update(payload)
                .eq("id", action_id)
                .execute()
            )
        else:
            response = (
                supabase.table("pending_booking_actions")
                .insert(payload)
                .execute()
            )

        if response.data:
            return response.data[0]

    except Exception as error:
        print(f"Could not save pending booking action: {error}")

    return None


def update_pending_booking_action_status(
    action_id: int,
    action_status: str,
):
    now_iso = datetime.now(timezone.utc).isoformat()
    payload = {
        "action_status": action_status,
        "updated_at": now_iso,
    }

    if action_status in {
        "resolved",
        "cancelled",
        "superseded",
    }:
        payload["resolved_at"] = now_iso

    try:
        (
            supabase.table("pending_booking_actions")
            .update(payload)
            .eq("id", action_id)
            .execute()
        )

    except Exception as error:
        print(f"Could not update pending booking action: {error}")



def resolve_matching_pending_booking_actions(
    session_id: str,
    service_name: str | None,
):
    """
    Resolve any unanswered appointment-structure question for a service that
    has now been explicitly added to the same visit.
    """
    cleaned_name = str(service_name or "").strip()

    if not cleaned_name:
        return

    try:
        response = (
            supabase.table("pending_booking_actions")
            .select("id")
            .eq("session_id", session_id)
            .eq("service_name", cleaned_name)
            .eq(
                "action_type",
                "choose_appointment_structure",
            )
            .eq("action_status", "pending")
            .execute()
        )

        for row in response.data or []:
            update_pending_booking_action_status(
                int(row["id"]),
                "resolved",
            )

    except Exception as error:
        print(
            "Could not resolve matching pending booking action: "
            f"{error}"
        )

def latest_message_confirms_same_visit(
    latest_message: str,
) -> bool:
    cleaned = normalise_service_match_text(
        latest_message
    )

    phrases = [
        "same visit",
        "same appointment",
        "back to back",
        "one after another",
        "together",
        "add it",
        "add that",
        "yes same",
    ]

    return any(
        phrase in cleaned
        for phrase in phrases
    )


def latest_message_requests_separate_appointment(
    latest_message: str,
) -> bool:
    cleaned = normalise_service_match_text(
        latest_message
    )

    phrases = [
        "separate",
        "separately",
        "different appointment",
        "another appointment",
        "another day",
    ]

    return any(
        phrase in cleaned
        for phrase in phrases
    )


def latest_message_cancels_pending_action(
    latest_message: str,
) -> bool:
    cleaned = normalise_service_match_text(
        latest_message
    )

    phrases = [
        "cancel that",
        "forget that",
        "dont add",
        "do not add",
        "leave it",
        "never mind",
        "nevermind",
    ]

    return any(
        phrase in cleaned
        for phrase in phrases
    )


def selected_identity_from_pending_action(
    action: dict,
) -> dict | None:
    service_name = action.get("service_name")
    identity = structured_service_identity(
        service_name
    )

    if identity:
        return identity

    service_row = load_service_row_for_booking_item(
        service_name
    )

    if not service_row:
        return None

    return {
        "category": (
            service_row.get("category")
            or action.get("category")
        ),
        "service_name": service_row.get(
            "service_name"
        ),
        "booking_mode": service_row.get(
            "booking_mode"
        ),
        "fixed_duration_minutes": service_row.get(
            "fixed_duration_minutes"
        ),
        "allowed_durations_minutes": service_row.get(
            "allowed_durations_minutes"
        ),
    }


def detect_pending_cart_action_resolution(
    session_id: str,
    latest_message: str,
) -> dict | None:
    """
    Resolve the oldest unanswered appointment-structure question.

    The same-visit branch is enabled now. Separate-request creation is kept as
    a clearly stored pending decision until Stage 4C creates a second
    booking_request safely.
    """
    actions = load_pending_booking_actions(
        session_id
    )

    if not actions:
        return None

    action = actions[0]

    if latest_message_confirms_same_visit(
        latest_message
    ):
        selected_identity = (
            selected_identity_from_pending_action(
                action
            )
        )

        if not selected_identity:
            return None

        draft_request = load_active_draft_booking_request(
            session_id
        )

        if not draft_request:
            return None

        return {
            "action": "add_same_visit",
            "draft_request": draft_request,
            "selected_identity": selected_identity,
            "pending_action_id": action.get("id"),
            "resolved_from_pending": True,
        }

    if latest_message_requests_separate_appointment(
        latest_message
    ):
        return {
            "action": "create_separate_request",
            "pending_action": action,
        }

    if latest_message_cancels_pending_action(
        latest_message
    ):
        return {
            "action": "cancel_pending_action",
            "pending_action": action,
        }

    return None


def format_pending_action_reminder(
    session_id: str,
) -> str:
    actions = load_pending_booking_actions(
        session_id
    )

    if not actions:
        return ""

    names = [
        str(action.get("service_name") or "").strip()
        for action in actions
        if str(action.get("service_name") or "").strip()
    ]

    if not names:
        return ""

    if len(names) == 1:
        return (
            f"{names[0]} is still waiting for your decision. "
            "Would you like to add it to this same visit, or make it "
            "a separate appointment request?"
        )

    return (
        "These treatments are still waiting for your decision: "
        + "; ".join(names)
        + ". Would you like to add them to this same visit, or handle "
        "them as separate appointment requests?"
    )


def has_pending_booking_actions(
    session_id: str,
) -> bool:
    return bool(
        load_pending_booking_actions(session_id)
    )


def suppress_contact_collection_while_pending(
    reply: str,
    session_id: str,
) -> str:
    """
    Pending cart decisions take priority over contact collection.

    A customer should finish choosing treatments before the bot asks for a
    name or phone number. This also prevents premature handover wording while
    an unresolved add / separate-appointment decision still exists.
    """
    reminder = format_pending_action_reminder(
        session_id
    )

    if not reminder:
        return str(reply or "").strip()

    cleaned = remove_contact_detail_questions(
        str(reply or "")
    )
    cleaned = remove_false_handover_confirmation(
        cleaned,
        ["pending_cart_action"],
    )
    cleaned = cleaned.strip()

    if not cleaned:
        return reminder

    return cleaned.rstrip() + "\n\n" + reminder


def append_pending_action_reminder(
    reply: str,
    session_id: str,
) -> str:
    return suppress_contact_collection_while_pending(
        reply,
        session_id,
    )


def detect_treatment_cart_intent(
    session_id: str,
    latest_message: str,
    existing_lead: dict,
) -> dict | None:
    """
    Detect whether a newly mentioned service should be added to the current
    appointment instead of replacing the existing treatment.

    We add automatically only when the customer explicitly says the treatments
    belong to the same visit. Ambiguous "also" messages trigger a clarification
    question rather than guessing.
    """
    existing_treatment = existing_lead.get("treatment")

    if not existing_treatment:
        return None

    draft_request = load_active_draft_booking_request(
        session_id
    )

    if not draft_request:
        return None

    selected_identity = resolve_latest_structured_service(
        latest_message,
        existing_treatment=existing_treatment,
    )

    if not selected_identity:
        return None

    selected_name = selected_identity.get("service_name")
    previous_identity = structured_service_identity(
        existing_treatment
    )
    previous_name = (
        previous_identity.get("service_name")
        if previous_identity
        else existing_treatment
    )

    if (
        not selected_name
        or normalise_treatment_name(selected_name)
        == normalise_treatment_name(previous_name)
    ):
        return None

    if latest_message_explicitly_adds_same_visit(
        latest_message
    ):
        return {
            "action": "add_same_visit",
            "draft_request": draft_request,
            "selected_identity": selected_identity,
        }

    if latest_message_suggests_additional_treatment(
        latest_message
    ):
        return {
            "action": "clarify_visit_structure",
            "draft_request": draft_request,
            "selected_identity": selected_identity,
        }

    return None


def restore_existing_primary_treatment(
    existing_lead: dict,
    lead_data: dict,
) -> dict:
    """
    During cart additions, keep the legacy overview pointed at its original
    primary treatment. The new treatment is stored as an additional item.
    """
    result = dict(lead_data)

    for field in [
        "treatment",
        "duration",
        "active_category",
    ]:
        if existing_lead.get(field) not in [None, ""]:
            result[field] = existing_lead.get(field)

    return result


def cart_item_from_selected_identity(
    selected_identity: dict,
) -> dict | None:
    service_row = load_service_row_for_booking_item(
        selected_identity.get("service_name")
    )

    if not service_row:
        return None

    fixed_duration = service_row.get(
        "fixed_duration_minutes"
    )

    if fixed_duration in [None, ""]:
        # Same-visit additions are enabled first for services with a resolved
        # duration. A selectable-duration massage is handled after the client
        # chooses its length.
        return None

    try:
        duration_minutes = int(fixed_duration)
    except (TypeError, ValueError):
        return None

    return {
        "service_row": service_row,
        "duration_minutes": duration_minutes,
        "price_pence": booking_item_price_pence(
            service_row,
            duration_minutes,
        ),
    }


def combined_duration_for_cart_action(
    cart_intent: dict | None,
) -> int | None:
    if (
        not cart_intent
        or cart_intent.get("action") != "add_same_visit"
    ):
        return None

    draft_request = cart_intent.get("draft_request") or {}
    selected_identity = cart_intent.get("selected_identity") or {}
    new_item = cart_item_from_selected_identity(
        selected_identity
    )

    if not new_item:
        return None

    current_total = int(
        draft_request.get("total_duration_minutes") or 0
    )

    return current_total + int(
        new_item["duration_minutes"]
    )


def calendar_lead_for_cart_action(
    lead_data: dict,
    cart_intent: dict | None,
) -> dict:
    """
    Calendar checks must use the full back-to-back duration while the legacy
    lead overview remains unchanged.
    """
    result = dict(lead_data)
    combined_minutes = combined_duration_for_cart_action(
        cart_intent
    )

    if combined_minutes:
        result["duration"] = format_minutes_as_duration(
            combined_minutes
        )

    return result


def active_lead_for_separate_request(
    existing_lead: dict,
    service_row: dict,
    duration_minutes: int,
) -> dict:
    """
    Build the conversations overview for the newly active separate draft.

    Contact details remain attached to the conversation, but date and time are
    cleared because they belong to the original appointment request.
    """
    result = dict(existing_lead)
    now_iso = datetime.now(timezone.utc).isoformat()

    result["treatment"] = service_row.get(
        "service_name"
    )
    result["duration"] = format_minutes_as_duration(
        duration_minutes
    )
    result["preferred_date"] = None
    result["preferred_time"] = None
    result["status"] = "details_incomplete"
    result["notification_sent_at"] = None
    result["conversation_mode"] = "booking_request"
    result["active_category"] = service_row.get(
        "category"
    )
    result["slot_status"] = "not_checked"
    result["next_required_detail"] = "preferred_date"
    result["last_service_switch_at"] = now_iso

    return result


def restore_parked_request_after_failure(
    session_id: str,
    original_request_id: int,
):
    """
    Best-effort rollback so a failed second-request creation cannot strand the
    customer's original appointment as parked.
    """
    try:
        (
            supabase.table("booking_requests")
            .update({
                "request_status": "draft",
                "parked_at": None,
                "updated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            })
            .eq("id", original_request_id)
            .execute()
        )

        update_conversation_active_booking_request(
            session_id,
            original_request_id,
        )

    except Exception as error:
        print(
            "Could not restore original draft after separate-request "
            f"failure: {error}"
        )


def create_separate_draft_booking_request(
    session_id: str,
    pending_action: dict,
    existing_lead: dict,
) -> dict | None:
    """
    Park the current draft and create one new independent appointment request.

    Only one request remains actively editable as request_status='draft'.
    The original appointment remains stored as parked_draft and is never
    overwritten by the new request.
    """
    original_request = load_active_draft_booking_request(
        session_id
    )

    if not original_request:
        return {
            "status": "missing_original_draft",
        }

    selected_identity = selected_identity_from_pending_action(
        pending_action
    )

    if not selected_identity:
        return {
            "status": "missing_service",
        }

    new_item = cart_item_from_selected_identity(
        selected_identity
    )

    if not new_item:
        return {
            "status": "needs_duration",
            "selected_service_name": selected_identity.get(
                "service_name"
            ),
        }

    service_row = new_item["service_row"]
    duration_minutes = int(
        new_item["duration_minutes"]
    )
    price_pence = new_item.get("price_pence")
    now_iso = datetime.now(timezone.utc).isoformat()
    original_request_id = int(
        original_request["id"]
    )
    new_request_id = None

    try:
        # Park first: the database intentionally permits only one active
        # request_status='draft' row for a session.
        (
            supabase.table("booking_requests")
            .update({
                "request_status": "parked_draft",
                "parked_at": now_iso,
                "updated_at": now_iso,
            })
            .eq("id", original_request_id)
            .execute()
        )

        response = (
            supabase.table("booking_requests")
            .insert({
                "session_id": session_id,
                "request_number": next_booking_request_number(
                    session_id
                ),
                "request_status": "draft",
                "appointment_structure": "single_service",
                "preferred_date": None,
                "preferred_time": None,
                "slot_status": "not_checked",
                "item_count": 1,
                "total_duration_minutes": duration_minutes,
                "total_price_pence": price_pence or 0,
                "missing_detail": "preferred_date",
                "updated_at": now_iso,
            })
            .execute()
        )

        if not response.data:
            raise RuntimeError(
                "Supabase returned no new booking request row."
            )

        new_request_id = int(
            response.data[0]["id"]
        )

        (
            supabase.table("booking_items")
            .insert({
                "booking_request_id": new_request_id,
                "service_id": service_row.get("id"),
                "service_name": service_row.get(
                    "service_name"
                ),
                "category": service_row.get("category"),
                "duration_minutes": duration_minutes,
                "price_pence": price_pence,
                "item_order": 1,
                "updated_at": now_iso,
            })
            .execute()
        )

        update_conversation_active_booking_request(
            session_id,
            new_request_id,
        )

        active_lead = active_lead_for_separate_request(
            existing_lead,
            service_row,
            duration_minutes,
        )

        save_conversation_overview(
            session_id=session_id,
            lead_data=active_lead,
            source=existing_lead.get("source") or "website",
        )

        if pending_action.get("id"):
            update_pending_booking_action_status(
                int(pending_action["id"]),
                "resolved",
            )

        return {
            "status": "created",
            "original_request": original_request,
            "new_request_id": new_request_id,
            "service_row": service_row,
            "duration_minutes": duration_minutes,
            "price_pence": price_pence,
        }

    except Exception as error:
        print(
            "Could not create separate appointment request: "
            f"{error}"
        )

        if new_request_id is not None:
            try:
                (
                    supabase.table("booking_requests")
                    .delete()
                    .eq("id", new_request_id)
                    .execute()
                )
            except Exception as cleanup_error:
                print(
                    "Could not remove failed second appointment draft: "
                    f"{cleanup_error}"
                )

        restore_parked_request_after_failure(
            session_id,
            original_request_id,
        )

        return {
            "status": "failed",
        }


def format_original_request_summary(
    original_request: dict,
) -> str:
    items = load_booking_items(
        int(original_request["id"])
    )
    names = [
        str(item.get("service_name") or "Treatment")
        for item in items
    ]

    if names:
        treatment_text = "; ".join(names)
    else:
        treatment_text = "your original appointment"

    preferred_date = original_request.get(
        "preferred_date"
    )
    preferred_time = original_request.get(
        "preferred_time"
    )

    if preferred_date and preferred_time:
        return (
            f"{treatment_text} for "
            f"{format_customer_slot(preferred_date, preferred_time)}"
        )

    return treatment_text


def build_separate_request_started_reply(
    separate_result: dict,
    session_id: str,
) -> str:
    original_summary = format_original_request_summary(
        separate_result["original_request"]
    )
    service_row = separate_result["service_row"]
    service_name = str(
        service_row.get("service_name")
        or "the new treatment"
    )
    price = format_price_pence(
        separate_result.get("price_pence")
    )
    duration = format_minutes_as_duration(
        separate_result.get("duration_minutes")
    )
    details = [
        value
        for value in [price, duration]
        if value
    ]
    new_summary = service_name

    if details:
        new_summary += " — " + ", ".join(details)

    reply = (
        f"No problem. Your original request for {original_summary} "
        "is still saved unchanged. "
        f"I've started a separate appointment request for {new_summary}.\n\n"
        "Which day and time would suit you for the separate appointment?"
    )

    return append_pending_action_reminder(
        reply,
        session_id,
    )


def sync_same_visit_treatment_item(
    session_id: str,
    lead_data: dict,
    cart_intent: dict,
) -> dict | None:
    """
    Append one treatment to the current request and recalculate cart totals.
    Existing items are never overwritten.
    """
    draft_request = cart_intent.get("draft_request") or {}
    selected_identity = cart_intent.get("selected_identity") or {}
    booking_request_id = draft_request.get("id")

    if not booking_request_id:
        return None

    new_item = cart_item_from_selected_identity(
        selected_identity
    )

    if not new_item:
        return {
            "status": "needs_duration",
            "selected_service_name": selected_identity.get(
                "service_name"
            ),
        }

    service_row = new_item["service_row"]
    duration_minutes = int(
        new_item["duration_minutes"]
    )
    price_pence = new_item.get("price_pence")
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        existing_items = load_booking_items(
            int(booking_request_id)
        )

        duplicate = next(
            (
                item
                for item in existing_items
                if normalise_treatment_name(
                    item.get("service_name")
                )
                == normalise_treatment_name(
                    service_row.get("service_name")
                )
            ),
            None,
        )

        if not duplicate:
            next_order = (
                max(
                    [
                        int(item.get("item_order") or 0)
                        for item in existing_items
                    ]
                    or [0]
                )
                + 1
            )

            (
                supabase.table("booking_items")
                .insert({
                    "booking_request_id": int(
                        booking_request_id
                    ),
                    "service_id": service_row.get("id"),
                    "service_name": service_row.get(
                        "service_name"
                    ),
                    "category": service_row.get("category"),
                    "duration_minutes": duration_minutes,
                    "price_pence": price_pence,
                    "item_order": next_order,
                    "updated_at": now_iso,
                })
                .execute()
            )

        items = load_booking_items(
            int(booking_request_id)
        )
        total_duration = sum(
            int(item.get("duration_minutes") or 0)
            for item in items
        )
        total_price = sum(
            int(item.get("price_pence") or 0)
            for item in items
        )

        (
            supabase.table("booking_requests")
            .update({
                "appointment_structure": (
                    "back_to_back"
                    if len(items) > 1
                    else "single_service"
                ),
                "preferred_date": lead_data.get(
                    "preferred_date"
                ),
                "preferred_time": lead_data.get(
                    "preferred_time"
                ),
                "slot_status": lead_data.get(
                    "slot_status"
                ) or "not_checked",
                "item_count": len(items),
                "total_duration_minutes": total_duration,
                "total_price_pence": total_price,
                "missing_detail": lead_data.get(
                    "next_required_detail"
                ),
                "updated_at": now_iso,
            })
            .eq("id", int(booking_request_id))
            .execute()
        )

        update_conversation_active_booking_request(
            session_id,
            int(booking_request_id),
        )

        return {
            "status": (
                "already_present"
                if duplicate
                else "added"
            ),
            "booking_request_id": int(
                booking_request_id
            ),
            "items": items,
            "item_count": len(items),
            "total_duration_minutes": total_duration,
            "total_price_pence": total_price,
        }

    except Exception as error:
        print(f"Could not add same-visit treatment item: {error}")
        return None


def format_cart_item_summary(
    item: dict,
) -> str:
    name = str(
        item.get("service_name") or "Treatment"
    )
    duration = format_minutes_as_duration(
        item.get("duration_minutes")
    )
    price = format_price_pence(
        item.get("price_pence")
    )

    details = [
        value
        for value in [price, duration]
        if value
    ]

    if not details:
        return name

    return name + " — " + ", ".join(details)


def build_same_visit_added_reply(
    cart_result: dict,
    lead_data: dict,
    calendar_result: dict,
    missing_fields: list[str],
) -> str:
    items = cart_result.get("items") or []
    summaries = [
        format_cart_item_summary(item)
        for item in items
    ]
    item_list = "; ".join(summaries)
    total_duration = format_minutes_as_duration(
        cart_result.get("total_duration_minutes")
    )
    total_price = format_price_pence(
        cart_result.get("total_price_pence")
    )

    opening = (
        "I've added that to the same visit. "
        f"Your request now includes: {item_list}. "
        f"Total: {total_price}, {total_duration}."
    )

    if calendar_result.get("suggestions"):
        return (
            opening
            + "\n\n"
            + build_calendar_alternative_reply(
                calendar_result
            )
        )

    status = str(
        calendar_result.get("status") or "not_checked"
    )

    if status == "free":
        slot = format_customer_slot(
            lead_data.get("preferred_date"),
            lead_data.get("preferred_time"),
        )
        opening += (
            f"\n\n{slot} still appears free for the "
            "combined appointment."
        )

    question = build_deterministic_next_question(
        lead_data,
        missing_fields,
    )

    if question:
        return opening + "\n\n" + question

    return (
        opening
        + "\n\nYour updated request has been recorded. "
        "The therapist will confirm it shortly."
    )


def build_visit_structure_clarification_reply(
    selected_identity: dict,
) -> str:
    selected_name = str(
        selected_identity.get("service_name")
        or "that treatment"
    )

    return (
        f"Would you like to add {selected_name} to the same visit, "
        "or make it a separate appointment request?"
    )


def sync_single_treatment_draft_request(
    session_id: str,
    lead_data: dict,
    booking_flow_active: bool,
):
    """
    Shadow-write the active single-treatment booking flow into the new tables.

    This does not alter replies or notifications yet. It gives us a safe way to
    verify that:
      conversation -> draft booking request -> booking item
    is populated correctly before treatment-cart behaviour is enabled.
    """
    if not booking_flow_active:
        return

    service_row = load_service_row_for_booking_item(
        lead_data.get("treatment")
    )

    if not service_row:
        return

    duration_minutes = booking_item_duration_minutes(
        lead_data,
        service_row,
    )

    # booking_items.duration_minutes is intentionally required. For selectable
    # services such as massage, wait until the client has chosen a duration.
    if duration_minutes is None or duration_minutes <= 0:
        return

    price_pence = booking_item_price_pence(
        service_row,
        duration_minutes,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    request_row = load_active_draft_booking_request(
        session_id
    )

    request_payload = {
        "session_id": session_id,
        "request_status": "draft",
        "appointment_structure": "single_service",
        "preferred_date": lead_data.get("preferred_date"),
        "preferred_time": lead_data.get("preferred_time"),
        "slot_status": lead_data.get("slot_status") or "not_checked",
        "item_count": 1,
        "total_duration_minutes": duration_minutes,
        "total_price_pence": price_pence or 0,
        "missing_detail": lead_data.get("next_required_detail"),
        "updated_at": now_iso,
    }

    try:
        if request_row:
            booking_request_id = int(
                request_row["id"]
            )

            (
                supabase.table("booking_requests")
                .update(request_payload)
                .eq("id", booking_request_id)
                .execute()
            )

        else:
            request_payload["request_number"] = (
                next_booking_request_number(session_id)
            )

            response = (
                supabase.table("booking_requests")
                .insert(request_payload)
                .execute()
            )

            if not response.data:
                return

            booking_request_id = int(
                response.data[0]["id"]
            )

        item_payload = {
            "booking_request_id": booking_request_id,
            "service_id": service_row.get("id"),
            "service_name": service_row.get("service_name"),
            "category": service_row.get("category"),
            "duration_minutes": duration_minutes,
            "price_pence": price_pence,
            "item_order": 1,
            "updated_at": now_iso,
        }

        (
            supabase.table("booking_items")
            .upsert(
                item_payload,
                on_conflict="booking_request_id,item_order",
            )
            .execute()
        )

        update_conversation_active_booking_request(
            session_id,
            booking_request_id,
        )

    except Exception as error:
        # Shadow persistence must never break the working customer chat.
        print(f"Could not sync draft booking request shadow data: {error}")


def save_conversation_overview(
    session_id: str,
    lead_data: dict,
    source: str = "website",
):
    try:
        payload = {
            "session_id": session_id,
            "source": normalise_source(source),
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
            model=LEAD_EXTRACTION_MODEL,
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

    raw_value = str(value).strip()

    for time_format in [
        "%H:%M",
        "%H:%M:%S",
        "%H:%M:%S.%f",
    ]:
        try:
            parsed_time = datetime.strptime(
                raw_value,
                time_format,
            )
            return parsed_time.strftime("%H:%M")

        except ValueError:
            continue

    return raw_value


def format_customer_slot(
    preferred_date: str | None,
    preferred_time: str | None,
) -> str:
    return (
        f"{format_customer_date(preferred_date)} "
        f"at {format_customer_time(preferred_time)}"
    )


def format_datetime_slot(
    slot_start: datetime,
) -> str:
    """
    Render one appointment slot from the real datetime object.

    Never let a generated weekday label drift away from the actual ISO date.
    Example: 2026-06-11 always renders as Thursday 11 June, never Thursday
    9 June.
    """
    local_start = slot_start.astimezone(
        BUSINESS_TIMEZONE
    )

    return format_customer_slot(
        local_start.date().isoformat(),
        local_start.strftime("%H:%M"),
    )


def suggested_slot_is_within_live_hours(
    slot_start: datetime,
    duration_minutes: int,
) -> bool:
    """
    Defensive display-time validation.

    Suggestions are generated from live hours already, but recheck them before
    displaying anything to a customer so a closed Tuesday, Wednesday, or
    Sunday can never appear in the final list.
    """
    local_start = slot_start.astimezone(
        BUSINESS_TIMEZONE
    )
    local_end = local_start + timedelta(
        minutes=duration_minutes
    )
    opening_window = working_hours_for_date(
        local_start.date()
    )

    if not opening_window:
        return False

    open_time, close_time = opening_window

    return (
        local_start >= open_time
        and local_start < close_time
        and local_end <= close_time
    )


def previous_reply_offered_calendar_alternatives(
    previous_assistant_message: str,
) -> bool:
    lower_message = str(previous_assistant_message or "").lower()

    return (
        "the next available options are" in lower_message
        or "would any of those times work" in lower_message
        or "would one of those times work" in lower_message
    )


def is_negative_reply(message: str) -> bool:
    cleaned = re.sub(
        r"[^a-z]+",
        " ",
        str(message or "").lower(),
    ).strip()

    return cleaned in {
        "no",
        "nope",
        "nah",
        "no thanks",
        "none",
        "none of those",
        "not really",
        "they dont work",
        "they do not work",
        "those dont work",
        "those do not work",
    }


def customer_rejected_calendar_alternatives(
    latest_message: str,
    previous_assistant_message: str,
) -> bool:
    """
    A plain rejection of proposed slots means no schedule has been agreed.
    Clear the old date and time before collecting any personal details.
    """
    return (
        previous_reply_offered_calendar_alternatives(
            previous_assistant_message
        )
        and is_negative_reply(latest_message)
        and not parse_relative_date_from_text(latest_message)
        and not parse_time_from_text(latest_message)
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
        "fantastic",
        "brilliant",
        "lovely",
        "fine",
        "sounds good",
        "sure",
        "alright",
        "yes",
        "yes please",
    }

    return cleaned in acknowledgements


def build_next_question_hint(
    missing_fields: list[str],
    lead_data: dict,
) -> str:
    if not missing_fields:
        return (
            "No booking details are missing. Do not ask another question. "
            "If the customer is only acknowledging the previous message, "
            "reply briefly: Thanks. The therapist will confirm the appointment shortly."
        )

    question = grouped_missing_question(
        missing_fields,
        lead_data,
    )

    return (
        "Ask this concise follow-up question after answering any customer "
        f"question: {question}"
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
    if (
        advanced_multi_booking_enabled()
        and has_pending_booking_actions(session_id)
    ):
        print(
            "Email notification skipped because a pending cart action "
            "still needs the customer's decision."
        )
        return

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
        "personality_notes": "",
        "enable_multi_booking": False,
    }

    try:
        response = (
            supabase.table("chatbot_settings")
            .select(
                "tone, reply_length, emoji_style, personality_notes, "
                "enable_multi_booking"
            )
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
        enable_multi_booking = bool(
            saved.get("enable_multi_booking")
        )

        if tone in ["friendly", "professional", "relaxed", "luxury"]:
            defaults["tone"] = tone

        if reply_length in ["short", "normal"]:
            defaults["reply_length"] = reply_length

        if emoji_style in ["none", "light", "friendly"]:
            defaults["emoji_style"] = emoji_style

        defaults["personality_notes"] = personality_notes[:600]
        defaults["enable_multi_booking"] = enable_multi_booking

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


def load_business_hours() -> dict:
    """
    Read editable working hours from Supabase.

    The dashboard table is authoritative. Missing rows and closed rows are
    treated as unavailable. Do not silently fall back to hardcoded hours,
    because that could offer a slot on a day the owner deliberately closed.
    """
    hours = {day: None for day in range(7)}

    try:
        response = (
            supabase.table("business_hours")
            .select("day_of_week, open_time, close_time, is_closed")
            .eq("business_slug", BUSINESS_SLUG)
            .order("day_of_week")
            .execute()
        )

        for row in response.data or []:
            day = row.get("day_of_week")

            if day not in range(7):
                continue

            if row.get("is_closed"):
                hours[day] = None
                continue

            open_time = row.get("open_time")
            close_time = row.get("close_time")

            if not open_time or not close_time:
                continue

            try:
                open_hour, open_minute, *_ = [
                    int(part)
                    for part in open_time.split(":")
                ]

                close_hour, close_minute, *_ = [
                    int(part)
                    for part in close_time.split(":")
                ]

                hours[day] = (
                    (open_hour, open_minute),
                    (close_hour, close_minute),
                )

            except Exception:
                continue

    except Exception as error:
        print(f"Could not load live business hours: {error}")

    return hours


def format_live_business_hours() -> str:
    labels = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    hours = load_business_hours()
    lines = []

    for day_number, label in enumerate(labels):
        value = hours.get(day_number)

        if not value:
            lines.append(f"- {label}: Closed")
            continue

        open_value, close_value = value

        if isinstance(open_value, tuple):
            open_hour, open_minute = open_value
        else:
            open_hour, open_minute = open_value, 0

        if isinstance(close_value, tuple):
            close_hour, close_minute = close_value
        else:
            close_hour, close_minute = close_value, 0

        lines.append(
            f"- {label}: {open_hour:02d}:{open_minute:02d}–"
            f"{close_hour:02d}:{close_minute:02d}"
        )

    return "\n".join(lines)


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
    business_hours = load_business_hours()

    for day_offset in range(NEXT_SLOT_SEARCH_DAYS + 1):
        candidate_date = (search_start + timedelta(days=day_offset)).date()
        hours = business_hours.get(candidate_date.weekday())

        if not hours:
            continue

        open_value, close_value = hours

        if isinstance(open_value, tuple):
            open_hour, open_minute = open_value
        else:
            open_hour, open_minute = open_value, 0

        if isinstance(close_value, tuple):
            close_hour, close_minute = close_value
        else:
            close_hour, close_minute = close_value, 0

        open_time = datetime.combine(
            candidate_date,
            datetime.min.time(),
            tzinfo=BUSINESS_TIMEZONE,
        ).replace(
            hour=open_hour,
            minute=open_minute,
        )

        close_time = datetime.combine(
            candidate_date,
            datetime.min.time(),
            tzinfo=BUSINESS_TIMEZONE,
        ).replace(
            hour=close_hour,
            minute=close_minute,
        )

        candidate = round_up_to_slot_step(max(open_time, search_start))

        while candidate + timedelta(minutes=duration_minutes) <= close_time:
            candidate_end = candidate + timedelta(minutes=duration_minutes)

            if (
                suggested_slot_is_within_live_hours(
                    candidate,
                    duration_minutes,
                )
                and not slot_overlaps_busy_period(
                    candidate,
                    candidate_end,
                    busy_periods,
                )
            ):
                suggestions.append(
                    format_datetime_slot(candidate)
                )

                if len(suggestions) >= NEXT_SLOT_LIMIT:
                    return suggestions

            candidate += timedelta(minutes=SLOT_STEP_MINUTES)

    return suggestions


def find_nearby_available_slots(
    duration_minutes: int,
    requested_start: datetime,
) -> list[str] | None:
    """
    Find the nearest valid slots around the customer's requested time on the
    same day first. If fewer than NEXT_SLOT_LIMIT are available, continue
    searching forward using the normal calendar search.
    """
    now = datetime.now(BUSINESS_TIMEZONE)
    opening_window = working_hours_for_date(requested_start.date())

    if not opening_window:
        return find_next_available_slots(
            duration_minutes,
            search_start=max(requested_start, now),
        )

    open_time, close_time = opening_window
    same_day_start = max(open_time, now) if requested_start.date() == now.date() else open_time

    busy_periods = query_google_freebusy(
        same_day_start,
        close_time,
    )

    if busy_periods is None:
        return None

    candidates = []
    candidate = round_up_to_slot_step(same_day_start)

    while candidate + timedelta(minutes=duration_minutes) <= close_time:
        candidate_end = candidate + timedelta(minutes=duration_minutes)

        if (
            suggested_slot_is_within_live_hours(
                candidate,
                duration_minutes,
            )
            and not slot_overlaps_busy_period(
                candidate,
                candidate_end,
                busy_periods,
            )
        ):
            candidates.append(candidate)

        candidate += timedelta(minutes=SLOT_STEP_MINUTES)

    candidates.sort(
        key=lambda value: (
            abs((value - requested_start).total_seconds()),
            value,
        )
    )

    suggestions = [
        format_datetime_slot(value)
        for value in candidates[:NEXT_SLOT_LIMIT]
    ]

    if len(suggestions) >= NEXT_SLOT_LIMIT:
        return suggestions

    future_search_start = max(close_time, now)
    future_suggestions = find_next_available_slots(
        duration_minutes,
        search_start=future_search_start,
    )

    if future_suggestions is None:
        return suggestions or None

    for option in future_suggestions:
        if option not in suggestions:
            suggestions.append(option)

        if len(suggestions) >= NEXT_SLOT_LIMIT:
            break

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


def add_calendar_alternatives(
    result: dict,
    duration_minutes: int | None,
    search_start: datetime,
) -> dict:
    """
    Attach the next genuinely available dashboard-and-calendar slots after an
    unavailable request. These suggestions are safe to show to the customer.
    """
    if not duration_minutes:
        return result

    suggestions = find_next_available_slots(
        duration_minutes,
        search_start=max(search_start, datetime.now(BUSINESS_TIMEZONE)),
    )

    if suggestions:
        result["suggestions"] = suggestions
        result["message"] = (
            result.get("message", "").rstrip()
            + " Suggested alternatives: "
            + "; ".join(suggestions)
            + ". Ask whether one of these works."
        )

    return result


def check_requested_calendar_slot(
    lead_data: dict,
    latest_message: str,
) -> dict:
    duration_minutes = parse_duration_minutes(
        lead_data.get("duration")
    )

    if (
        asks_for_next_available_time(latest_message)
        and not duration_minutes
    ):
        return {
            "status": "needs_duration_for_alternatives",
            "message": (
                "Ask for the treatment duration before listing slots. "
                "Do not guess availability."
            ),
        }

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
        return add_calendar_alternatives(
            {
                "status": "past",
                "message": (
                    "The requested time has already passed. "
                    "Offer the next available alternatives."
                ),
            },
            duration_minutes,
            datetime.now(BUSINESS_TIMEZONE),
        )

    working_hours_error = validate_slot_against_working_hours(
        start_time,
        end_time,
        duration_minutes,
    )

    if working_hours_error:
        return add_calendar_alternatives(
            working_hours_error,
            duration_minutes,
            start_time,
        )

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
        if not duration_minutes:
            return {
                "status": "busy_needs_duration",
                "customer_slot": customer_slot,
                "message": (
                    f"{customer_slot} is unavailable. Ask for the duration "
                    "before listing alternative slots."
                ),
            }

        suggestions = find_nearby_available_slots(
            duration_minutes,
            requested_start=start_time,
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
            "suggestions": suggestions or [],
            "message": (
                f"{customer_slot} is unavailable. "
                "Say this once, briefly, and offer the nearest available "
                "alternatives before and after the requested slot when present."
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
                "suggestions": suggestions,
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


def instagram_is_configured() -> bool:
    return all([
        INSTAGRAM_ACCESS_TOKEN,
        INSTAGRAM_ACCOUNT_ID,
        INSTAGRAM_APP_SECRET,
        INSTAGRAM_VERIFY_TOKEN,
    ])


def verify_instagram_signature(
    raw_body: bytes,
    signature_header: str | None,
) -> bool:
    """Verify Meta's signed webhook payload before processing it."""
    if not INSTAGRAM_APP_SECRET or not signature_header:
        return False

    if not signature_header.startswith("sha256="):
        return False

    received_signature = signature_header.split("=", 1)[1]

    expected_signature = hmac.new(
        INSTAGRAM_APP_SECRET.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(received_signature, expected_signature)


def load_recent_messages(
    session_id: str,
    limit: int = 20,
) -> list[ChatMessage]:
    try:
        response = (
            supabase.table("messages")
            .select("role, content, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        rows = list(reversed(response.data or []))

        return [
            ChatMessage(
                role=row.get("role", "user"),
                content=row.get("content", ""),
            )
            for row in rows
            if row.get("content")
        ]

    except Exception as error:
        print(f"Could not load recent messages: {error}")
        return []


def send_instagram_text_message(
    recipient_id: str,
    text: str,
) -> bool:
    if not INSTAGRAM_ACCESS_TOKEN:
        print("Instagram reply skipped: missing access token.")
        return False

    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text[:1000]},
    }

    endpoint = (
        f"https://graph.instagram.com/"
        f"{INSTAGRAM_GRAPH_VERSION}/me/messages?"
        + urllib_parse.urlencode({
            "access_token": INSTAGRAM_ACCESS_TOKEN,
        })
    )

    api_request = urllib_request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "receptionist-chatbot/1.0",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(api_request, timeout=15) as response:
            response.read()

        print(f"Instagram reply sent to {recipient_id}")
        return True

    except urllib_error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        print(f"Instagram reply failed: HTTP {error.code} {body}")

    except Exception as error:
        print(f"Instagram reply failed: {error}")

    return False


_processed_instagram_message_ids: set[str] = set()
_instagram_rate_limit_notice_sent_at: dict[str, datetime] = {}


def new_instagram_session_id(sender_id: str) -> str:
    return f"instagram:{sender_id}:{secrets.token_urlsafe(8)}"


def get_latest_instagram_conversation(sender_id: str) -> dict:
    try:
        response = (
            supabase.table("conversations")
            .select("*")
            .eq("source", "instagram")
            .like("session_id", f"instagram:{sender_id}%")
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )

        if response.data:
            return response.data[0]

    except Exception as error:
        print(f"Could not load latest Instagram conversation: {error}")

    return {}


def instagram_session_is_stale(conversation: dict) -> bool:
    updated_at = conversation.get("updated_at")

    if not updated_at:
        return True

    try:
        updated = datetime.fromisoformat(
            str(updated_at).replace("Z", "+00:00")
        )

        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)

        return (
            datetime.now(timezone.utc) - updated.astimezone(timezone.utc)
            > timedelta(hours=INSTAGRAM_SESSION_IDLE_HOURS)
        )

    except ValueError:
        return True


def resolve_instagram_session(
    sender_id: str,
    latest_message: str,
) -> tuple[str, dict | None]:
    """
    Continue a recent incomplete Instagram lead. Start fresh after a completed
    request or long inactivity so old booking details do not leak into a new
    enquiry.
    """
    latest = get_latest_instagram_conversation(sender_id)

    if not latest:
        return new_instagram_session_id(sender_id), None

    if (
        latest.get("status") == "booking_request_complete"
        and not is_simple_acknowledgement(latest_message)
    ):
        return new_instagram_session_id(sender_id), latest

    if instagram_session_is_stale(latest):
        return new_instagram_session_id(sender_id), latest

    return str(latest.get("session_id")), None


def format_previous_instagram_request_summary(conversation: dict) -> str:
    treatment = conversation.get("treatment") or "your previous treatment request"
    duration = conversation.get("duration")
    preferred_date = format_customer_date(conversation.get("preferred_date"))
    preferred_time = format_customer_time(conversation.get("preferred_time"))

    details = [str(treatment)]

    if duration:
        details.append(str(duration))

    if conversation.get("preferred_date") and conversation.get("preferred_time"):
        details.append(f"{preferred_date} at {preferred_time}")

    return (
        "Your previous request for "
        + ", ".join(details)
        + " has been noted, and the therapist still needs to confirm it."
    )


def count_recent_instagram_customer_messages(
    sender_id: str,
    since: datetime,
    limit: int,
) -> int:
    try:
        response = (
            supabase.table("messages")
            .select("session_id")
            .eq("source", "instagram")
            .eq("role", "user")
            .like("session_id", f"instagram:{sender_id}%")
            .gte("created_at", since.astimezone(timezone.utc).isoformat())
            .limit(limit + 1)
            .execute()
        )

        return len(response.data or [])

    except Exception as error:
        print(f"Could not count recent Instagram messages: {error}")
        return 0


def instagram_rate_limit_message(sender_id: str) -> str | None:
    now = datetime.now(timezone.utc)

    ten_minute_count = count_recent_instagram_customer_messages(
        sender_id,
        now - timedelta(minutes=10),
        INSTAGRAM_MAX_MESSAGES_PER_10_MINUTES,
    )

    day_count = count_recent_instagram_customer_messages(
        sender_id,
        now - timedelta(days=1),
        INSTAGRAM_MAX_MESSAGES_PER_DAY,
    )

    if (
        ten_minute_count < INSTAGRAM_MAX_MESSAGES_PER_10_MINUTES
        and day_count < INSTAGRAM_MAX_MESSAGES_PER_DAY
    ):
        return None

    last_notice = _instagram_rate_limit_notice_sent_at.get(sender_id)

    if (
        last_notice
        and now - last_notice
        < timedelta(minutes=INSTAGRAM_RATE_LIMIT_NOTICE_COOLDOWN_MINUTES)
    ):
        return ""

    _instagram_rate_limit_notice_sent_at[sender_id] = now

    return (
        "You've sent quite a few messages. Please wait a little while before "
        "sending more, or call 07943319617 if your enquiry is urgent."
    )


def mark_instagram_message_seen(message_id: str) -> bool:
    """Small duplicate guard for Meta webhook retries during MVP testing."""
    if not message_id:
        return True

    if message_id in _processed_instagram_message_ids:
        return False

    _processed_instagram_message_ids.add(message_id)

    if len(_processed_instagram_message_ids) > 2000:
        _processed_instagram_message_ids.clear()
        _processed_instagram_message_ids.add(message_id)

    return True


def process_instagram_webhook_payload(payload: dict):
    """Process incoming Instagram text DMs after Meta already received HTTP 200."""
    if payload.get("object") != "instagram":
        return

    for entry in payload.get("entry", []):
        for event in entry.get("messaging", []):
            sender_id = str(event.get("sender", {}).get("id", ""))
            recipient_id = str(event.get("recipient", {}).get("id", ""))
            message_data = event.get("message") or {}
            message_id = str(message_data.get("mid", ""))
            message_text = message_data.get("text")

            if not sender_id or not message_text:
                continue

            if sender_id == str(INSTAGRAM_ACCOUNT_ID):
                continue

            if recipient_id and recipient_id != str(INSTAGRAM_ACCOUNT_ID):
                continue

            if message_data.get("is_echo"):
                continue

            if not mark_instagram_message_seen(message_id):
                continue

            rate_limit_reply = instagram_rate_limit_message(sender_id)

            if rate_limit_reply is not None:
                if rate_limit_reply:
                    send_instagram_text_message(
                        recipient_id=sender_id,
                        text=rate_limit_reply,
                    )

                continue

            session_id, previous_conversation = resolve_instagram_session(
                sender_id,
                message_text,
            )

            history = load_recent_messages(session_id)

            request = ChatRequest(
                session_id=session_id,
                message=message_text,
                history=history,
                source="instagram",
            )

            try:
                # If a prior request is already complete and the returning
                # customer only says hello, provide a concise summary and ask
                # whether they need anything else instead of reusing old lead
                # details or restarting an unnecessary form.
                if (
                    previous_conversation
                    and re.sub(r"[^a-z]+", " ", message_text.lower()).strip()
                    in {"hi", "hello", "hey", "hiya", "good morning", "good afternoon"}
                ):
                    reply = (
                        format_previous_instagram_request_summary(
                            previous_conversation
                        )
                        + " Is there anything else I can help with, such as "
                        "booking another treatment?"
                    )

                    save_message(
                        session_id=session_id,
                        role="user",
                        content=message_text,
                        source="instagram",
                    )

                    save_message(
                        session_id=session_id,
                        role="assistant",
                        content=reply,
                        source="instagram",
                    )

                else:
                    response = asyncio.run(
                        chat(request, BackgroundTasks())
                    )

                    reply = response.get("reply")

                    if previous_conversation and reply:
                        reply = (
                            format_previous_instagram_request_summary(
                                previous_conversation
                            )
                            + "\n\n"
                            + reply
                        )

                if reply:
                    send_instagram_text_message(
                        recipient_id=sender_id,
                        text=reply,
                    )

                # The BackgroundTasks object above is not executed by FastAPI,
                # because this call is internal. Trigger the alert directly.
                send_booking_notification(session_id)

            except Exception as error:
                print(f"Could not process Instagram message: {error}")


@app.get("/instagram/status")
async def instagram_status():
    return {
        "configured": instagram_is_configured(),
        "account_id_present": bool(INSTAGRAM_ACCOUNT_ID),
        "access_token_present": bool(INSTAGRAM_ACCESS_TOKEN),
        "app_secret_present": bool(INSTAGRAM_APP_SECRET),
        "verify_token_present": bool(INSTAGRAM_VERIFY_TOKEN),
    }


@app.get("/instagram/webhook")
async def verify_instagram_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    verify_token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if (
        mode == "subscribe"
        and verify_token
        and verify_token == INSTAGRAM_VERIFY_TOKEN
        and challenge
    ):
        return PlainTextResponse(challenge)

    return PlainTextResponse(
        "Instagram webhook verification failed.",
        status_code=403,
    )


@app.post("/instagram/webhook")
async def receive_instagram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_instagram_signature(raw_body, signature):
        return PlainTextResponse(
            "Invalid Instagram webhook signature.",
            status_code=403,
        )

    try:
        payload = json.loads(raw_body.decode("utf-8"))

    except json.JSONDecodeError:
        return PlainTextResponse(
            "Invalid JSON payload.",
            status_code=400,
        )

    background_tasks.add_task(
        process_instagram_webhook_payload,
        payload,
    )

    return {"status": "received"}


@app.get("/ai/status")
async def ai_status():
    return {
        "live_chat_model": LIVE_CHAT_MODEL,
        "live_chat_reasoning_effort": LIVE_CHAT_REASONING_EFFORT,
        "live_chat_max_completion_tokens": LIVE_CHAT_MAX_COMPLETION_TOKENS,
        "lead_extraction_model": LEAD_EXTRACTION_MODEL,
    }


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


def customer_asks_for_booking_summary(
    latest_message: str,
) -> bool:
    """
    Detect a clear request to list or summarise the customer's appointment
    requests. Keep this conservative so normal treatment questions do not
    trigger a booking summary unexpectedly.
    """
    cleaned = normalise_service_match_text(
        latest_message
    )

    exact_phrases = {
        "what have i requested",
        "what have i booked",
        "what bookings do i have",
        "what appointments do i have",
        "show my bookings",
        "show my booking requests",
        "show my appointments",
        "summarise my bookings",
        "summarize my bookings",
        "summarise my appointments",
        "summarize my appointments",
        "list my bookings",
        "list my appointments",
        "what is currently booked",
        "what is currently requested",
    }

    if cleaned in exact_phrases:
        return True

    has_summary_word = any(
        word in cleaned
        for word in [
            "summarise",
            "summarize",
            "summary",
            "list",
            "show",
            "what",
        ]
    )
    has_booking_word = any(
        phrase in cleaned
        for phrase in [
            "my booking",
            "my bookings",
            "my appointment",
            "my appointments",
            "my request",
            "my requests",
            "have i requested",
            "have i booked",
        ]
    )

    return has_summary_word and has_booking_word


def load_booking_requests_for_summary(
    session_id: str,
) -> list[dict]:
    try:
        response = (
            supabase.table("booking_requests")
            .select("*")
            .eq("session_id", session_id)
            .order("request_number")
            .execute()
        )

        return list(response.data or [])

    except Exception as error:
        print(f"Could not load booking requests for summary: {error}")
        return []


def booking_request_status_for_customer(
    request_status: str | None,
) -> str:
    status = str(
        request_status or "draft"
    ).strip()

    labels = {
        "draft": "draft request — details may still be needed",
        "parked_draft": "saved request — not confirmed yet",
        "ready_for_review": "ready for therapist review — not confirmed yet",
        "awaiting_owner_confirmation": "awaiting therapist confirmation",
        "confirmed": "confirmed",
        "declined": "declined",
        "cancelled_by_customer": "cancelled",
        "abandoned": "unfinished request",
        "expired": "expired",
    }

    return labels.get(
        status,
        status.replace("_", " "),
    )


def booking_request_items_summary(
    booking_request_id: int,
) -> list[str]:
    items = load_booking_items(
        booking_request_id
    )
    summaries = []

    for item in items:
        summaries.append(
            format_cart_item_summary(item)
        )

    return summaries


def booking_request_schedule_summary(
    booking_request: dict,
) -> str:
    preferred_date = booking_request.get(
        "preferred_date"
    )
    preferred_time = booking_request.get(
        "preferred_time"
    )

    if preferred_date and preferred_time:
        return format_customer_slot(
            preferred_date,
            preferred_time,
        )

    if preferred_date:
        return (
            f"{format_customer_date(preferred_date)} "
            "— time still needed"
        )

    if preferred_time:
        return (
            f"time requested: "
            f"{format_customer_time(preferred_time)} "
            "— day still needed"
        )

    return "day and time still needed"


def format_single_booking_request_summary(
    booking_request: dict,
) -> str:
    request_number = int(
        booking_request.get("request_number") or 0
    )
    status_label = booking_request_status_for_customer(
        booking_request.get("request_status")
    )
    schedule_label = booking_request_schedule_summary(
        booking_request
    )
    item_summaries = booking_request_items_summary(
        int(booking_request["id"])
    )

    lines = [
        f"{request_number}. {status_label}",
        f"   When: {schedule_label}",
    ]

    if item_summaries:
        lines.append(
            "   Treatments: "
            + "; ".join(item_summaries)
        )
    else:
        lines.append(
            "   Treatments: details still needed"
        )

    item_count = int(
        booking_request.get("item_count") or 0
    )

    if item_count > 1:
        total_price = format_price_pence(
            booking_request.get("total_price_pence")
        )
        total_duration = format_minutes_as_duration(
            booking_request.get("total_duration_minutes")
        )
        total_details = [
            value
            for value in [
                total_price,
                total_duration,
            ]
            if value
        ]

        if total_details:
            lines.append(
                "   Combined total: "
                + ", ".join(total_details)
            )

    return "\n".join(lines)


def build_customer_booking_summary(
    session_id: str,
) -> str:
    """
    Build a clear summary from relational booking data.

    Active requests are shown first. Cancelled, declined, abandoned, and
    expired requests are shown separately so the customer can distinguish
    what remains active from what is no longer proceeding.
    """
    booking_requests = load_booking_requests_for_summary(
        session_id
    )
    pending_actions = load_pending_booking_actions(
        session_id
    )

    if not booking_requests and not pending_actions:
        return (
            "You do not currently have any appointment requests recorded "
            "in this chat. Which treatment would you like to arrange?"
        )

    inactive_statuses = {
        "cancelled_by_customer",
        "declined",
        "abandoned",
        "expired",
    }
    active_requests = [
        request
        for request in booking_requests
        if request.get("request_status")
        not in inactive_statuses
    ]
    inactive_requests = [
        request
        for request in booking_requests
        if request.get("request_status")
        in inactive_statuses
    ]

    sections = []

    if active_requests:
        active_lines = [
            "You currently have:"
        ]

        for request in active_requests:
            active_lines.append(
                format_single_booking_request_summary(
                    request
                )
            )

        if not any(
            request.get("request_status") == "confirmed"
            for request in active_requests
        ):
            active_lines.append(
                "These are appointment requests, not confirmed bookings. "
                "The therapist still needs to confirm them."
            )

        sections.append(
            "\n\n".join(active_lines)
        )
    else:
        sections.append(
            "You do not currently have any active appointment requests."
        )

    if pending_actions:
        names = [
            str(
                action.get("service_name")
                or "an additional treatment"
            )
            for action in pending_actions
        ]

        sections.append(
            "Still waiting for your decision: "
            + "; ".join(names)
            + ". Please tell me whether you want "
            + (
                "it"
                if len(names) == 1
                else "them"
            )
            + " added to an existing visit or arranged as "
            "a separate appointment request."
        )

    if inactive_requests:
        inactive_lines = [
            "No longer active:"
        ]

        for request in inactive_requests:
            inactive_lines.append(
                format_single_booking_request_summary(
                    request
                )
            )

        sections.append(
            "\n\n".join(inactive_lines)
        )

    return "\n\n".join(sections)


def identity_key(
    identity: dict,
) -> tuple[str, str]:
    return (
        str(identity.get("category") or "").strip(),
        normalise_treatment_name(
            identity.get("service_name")
        ),
    )


def dedupe_service_identities(
    identities: list[dict],
) -> list[dict]:
    seen = set()
    result = []

    for identity in identities:
        key = identity_key(identity)

        if not key[1] or key in seen:
            continue

        seen.add(key)
        result.append(identity)

    return result


def generic_massage_identity() -> dict:
    return {
        "category": "massage",
        "service_name": "Massage",
        "booking_mode": "choose_variant",
        "fixed_duration_minutes": None,
        "allowed_durations_minutes": None,
    }


def message_explicitly_requests_same_visit(
    latest_message: str,
) -> bool:
    cleaned = normalise_service_match_text(
        latest_message
    )

    phrases = [
        "together",
        "same visit",
        "same appointment",
        "back to back",
        "one after another",
        "straight after",
        "during the same visit",
    ]

    return any(
        phrase in cleaned
        for phrase in phrases
    )


def split_multi_service_message(
    latest_message: str,
) -> list[str]:
    """
    Split only on clear service-list separators.

    The full message is also scanned separately, so this helper does not need
    to understand every sentence structure.
    """
    cleaned = str(latest_message or "")

    pieces = re.split(
        r"(?i)\s*(?:,|&|\+|\band\b|\bplus\b|\bwith\b|\btogether\b)\s*",
        cleaned,
    )

    return [
        piece.strip()
        for piece in pieces
        if piece.strip()
    ]


def detect_all_structured_services(
    latest_message: str,
) -> list[dict]:
    """
    Resolve every service explicitly mentioned in the newest customer message.

    Existing single-service resolvers return only one result. This controller
    scans each list segment and each structured category independently so
    same-category combinations such as Body Sculpting + EMS are preserved.
    """
    identities = []

    direct_lookups = [
        ("massage", find_massage_service),
        ("vitamin_shots", find_vitamin_shot_service),
        ("ultrasound", find_ultrasound_service),
        ("body_treatments", find_body_treatment_service),
        ("microneedling", find_microneedling_service),
        ("facials", find_facial_service),
        ("dermal_fillers", find_dermal_filler_service),
    ]

    segments = split_multi_service_message(
        latest_message
    )

    for segment in segments:
        resolved = resolve_latest_structured_service(
            segment
        )

        if resolved:
            identities.append(resolved)
            continue

        cleaned_segment = normalise_service_match_text(
            segment
        )

        if re.search(r"\bmassages?\b", cleaned_segment):
            identities.append(
                generic_massage_identity()
            )

    # Scan category finders against the full message as a second pass.
    # This improves recall when the customer's grammar is less tidy.
    for category, finder in direct_lookups:
        service = finder(latest_message)

        if service:
            identities.append({
                "category": category,
                "service_name": service.get("service_name"),
                "booking_mode": service.get("booking_mode"),
                "fixed_duration_minutes": service.get(
                    "fixed_duration_minutes"
                ),
                "allowed_durations_minutes": service.get(
                    "allowed_durations_minutes"
                ),
            })

    cleaned = normalise_service_match_text(
        latest_message
    )

    if (
        re.search(r"\bmassages?\b", cleaned)
        and not any(
            identity.get("category") == "massage"
            for identity in identities
        )
    ):
        identities.append(
            generic_massage_identity()
        )

    # Preserve partial filler choices such as "lip filler" when amount is
    # still missing.
    generic_or_partial = resolve_latest_generic_or_partial_service(
        latest_message
    )

    if generic_or_partial and not any(
        identity.get("category")
        == generic_or_partial.get("category")
        for identity in identities
    ):
        identities.append(
            generic_or_partial
        )

    return dedupe_service_identities(
        identities
    )


def load_pending_service_detail_actions(
    session_id: str,
) -> list[dict]:
    try:
        response = (
            supabase.table("pending_booking_actions")
            .select("*")
            .eq("session_id", session_id)
            .eq("action_status", "pending")
            .in_(
                "action_type",
                ["choose_variant", "choose_duration"],
            )
            .order("created_at")
            .execute()
        )

        return list(response.data or [])

    except Exception as error:
        print(
            "Could not load pending service-detail actions: "
            f"{error}"
        )
        return []


def save_pending_service_detail_action(
    session_id: str,
    identity: dict,
    action_type: str,
    source_message: str,
) -> dict | None:
    service_name = str(
        identity.get("service_name") or ""
    ).strip()

    if (
        not service_name
        or action_type not in {
            "choose_variant",
            "choose_duration",
        }
    ):
        return None

    service_row = load_service_row_for_booking_item(
        service_name
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    payload = {
        "session_id": session_id,
        "service_id": (
            service_row.get("id")
            if service_row
            else None
        ),
        "service_name": service_name,
        "category": (
            identity.get("category")
            or (
                service_row.get("category")
                if service_row
                else None
            )
        ),
        "action_type": action_type,
        "action_status": "pending",
        "source_message": source_message,
        "updated_at": now_iso,
    }

    try:
        existing_response = (
            supabase.table("pending_booking_actions")
            .select("id")
            .eq("session_id", session_id)
            .eq("service_name", service_name)
            .eq("action_type", action_type)
            .eq("action_status", "pending")
            .limit(1)
            .execute()
        )

        if existing_response.data:
            action_id = int(
                existing_response.data[0]["id"]
            )

            response = (
                supabase.table("pending_booking_actions")
                .update(payload)
                .eq("id", action_id)
                .execute()
            )
        else:
            response = (
                supabase.table("pending_booking_actions")
                .insert(payload)
                .execute()
            )

        if response.data:
            return response.data[0]

    except Exception as error:
        print(
            "Could not save pending service-detail action: "
            f"{error}"
        )

    return None


def update_pending_service_detail_action(
    action_id: int,
    identity: dict,
    action_type: str,
):
    service_name = str(
        identity.get("service_name") or ""
    ).strip()
    service_row = load_service_row_for_booking_item(
        service_name
    )

    try:
        (
            supabase.table("pending_booking_actions")
            .update({
                "service_id": (
                    service_row.get("id")
                    if service_row
                    else None
                ),
                "service_name": service_name,
                "category": (
                    identity.get("category")
                    or (
                        service_row.get("category")
                        if service_row
                        else None
                    )
                ),
                "action_type": action_type,
                "updated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            })
            .eq("id", action_id)
            .execute()
        )

    except Exception as error:
        print(
            "Could not update pending service-detail action: "
            f"{error}"
        )


def pending_action_needs_variant(
    identity: dict,
) -> bool:
    return (
        identity.get("booking_mode") == "choose_variant"
        or identity.get("service_name") in {
            "Massage",
            "Vitamin Shot",
            "Ultrasound",
            "Microneedling",
            "Facial",
            "Dermal Filler",
        }
        or is_partial_dermal_filler_treatment(
            identity.get("service_name")
        )
    )


def pending_action_needs_duration(
    identity: dict,
) -> bool:
    allowed = identity.get(
        "allowed_durations_minutes"
    ) or []

    return bool(
        allowed
        and not identity.get(
            "fixed_duration_minutes"
        )
    )


def build_pending_service_detail_question(
    action: dict,
) -> str:
    service_name = str(
        action.get("service_name") or ""
    )
    category = str(
        action.get("category") or ""
    )
    action_type = str(
        action.get("action_type") or ""
    )

    if action_type == "choose_duration":
        service_row = load_service_row_for_booking_item(
            service_name
        ) or {}
        durations = service_row.get(
            "allowed_durations_minutes"
        ) or []
        options = format_duration_options(
            {int(value) for value in durations}
        ) if durations else ""

        if options:
            return (
                f"How long would you like the {service_name} for: "
                f"{options}?"
            )

        return (
            f"How long would you like the {service_name} for?"
        )

    if category == "massage":
        return (
            "Which massage would you like: Relaxing Massage, Swedish "
            "Massage, Deep Tissue Massage, or Hot Stone Massage?"
        )

    if category == "dermal_fillers":
        return build_dermal_filler_variant_prompt({
            "treatment": service_name,
        })

    if category == "vitamin_shots":
        return (
            "Would you like a B12 Vitamin Shot or a Vitamin C Shot? "
            "Both cost £20 and take 15 minutes."
        )

    if category == "ultrasound":
        return (
            "Would you like a single ultrasound session (£50, 45 minutes) "
            "or the package of 5 sessions (£220 total, 45 minutes each)?"
        )

    if category == "microneedling":
        return (
            "Would you like microneedling for stretch marks (£60, "
            "1 hour 15 minutes) or for the face and neck (£70, "
            "1 hour 15 minutes)?"
        )

    if category == "facials":
        return (
            "Which facial would you like: Deep Facial Cleanse (£60, "
            "60 minutes), Radio Frequency Facial (£60, 60 minutes), "
            "Facial Treatment for Young Skin (£45, 60 minutes), "
            "Hydraface Facial Treatment (£60, 60 minutes), or "
            "Hydrafacial (£80, 90 minutes)?"
        )

    return (
        f"Which {service_name or 'treatment'} option would you like?"
    )


def append_booking_item_to_request(
    booking_request_id: int,
    service_row: dict,
    duration_minutes: int,
) -> dict | None:
    price_pence = booking_item_price_pence(
        service_row,
        duration_minutes,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        items = load_booking_items(
            booking_request_id
        )

        duplicate = next(
            (
                item
                for item in items
                if normalise_treatment_name(
                    item.get("service_name")
                )
                == normalise_treatment_name(
                    service_row.get("service_name")
                )
            ),
            None,
        )

        if not duplicate:
            next_order = (
                max(
                    [
                        int(item.get("item_order") or 0)
                        for item in items
                    ]
                    or [0]
                )
                + 1
            )

            (
                supabase.table("booking_items")
                .insert({
                    "booking_request_id": booking_request_id,
                    "service_id": service_row.get("id"),
                    "service_name": service_row.get(
                        "service_name"
                    ),
                    "category": service_row.get("category"),
                    "duration_minutes": duration_minutes,
                    "price_pence": price_pence,
                    "item_order": next_order,
                    "updated_at": now_iso,
                })
                .execute()
            )

        items = load_booking_items(
            booking_request_id
        )
        total_duration = sum(
            int(item.get("duration_minutes") or 0)
            for item in items
        )
        total_price = sum(
            int(item.get("price_pence") or 0)
            for item in items
        )

        (
            supabase.table("booking_requests")
            .update({
                "appointment_structure": (
                    "back_to_back"
                    if len(items) > 1
                    else "single_service"
                ),
                "item_count": len(items),
                "total_duration_minutes": total_duration,
                "total_price_pence": total_price,
                "updated_at": now_iso,
            })
            .eq("id", booking_request_id)
            .execute()
        )

        return {
            "items": items,
            "item_count": len(items),
            "total_duration_minutes": total_duration,
            "total_price_pence": total_price,
        }

    except Exception as error:
        print(
            "Could not append booking item to request: "
            f"{error}"
        )
        return None


def create_initial_multi_service_request(
    session_id: str,
    identities: list[dict],
    source_message: str,
    source: str,
) -> dict | None:
    now_iso = datetime.now(timezone.utc).isoformat()
    request_id = None

    try:
        response = (
            supabase.table("booking_requests")
            .insert({
                "session_id": session_id,
                "request_number": next_booking_request_number(
                    session_id
                ),
                "request_status": "draft",
                "appointment_structure": "back_to_back",
                "preferred_date": parse_relative_date_from_text(
                    source_message
                ),
                "preferred_time": parse_time_from_text(
                    source_message
                ),
                "slot_status": "not_checked",
                "item_count": 0,
                "total_duration_minutes": 0,
                "total_price_pence": 0,
                "missing_detail": "service_variant",
                "updated_at": now_iso,
            })
            .execute()
        )

        if not response.data:
            return None

        request_id = int(
            response.data[0]["id"]
        )

        update_conversation_active_booking_request(
            session_id,
            request_id,
        )

        first_resolved_service = None

        for identity in identities:
            if pending_action_needs_variant(
                identity
            ):
                save_pending_service_detail_action(
                    session_id,
                    identity,
                    "choose_variant",
                    source_message,
                )
                continue

            service_row = load_service_row_for_booking_item(
                identity.get("service_name")
            )

            if not service_row:
                continue

            fixed_duration = service_row.get(
                "fixed_duration_minutes"
            )

            if fixed_duration not in [None, ""]:
                duration_minutes = int(
                    fixed_duration
                )
                append_booking_item_to_request(
                    request_id,
                    service_row,
                    duration_minutes,
                )

                if not first_resolved_service:
                    first_resolved_service = service_row

                continue

            if pending_action_needs_duration(
                identity
            ):
                save_pending_service_detail_action(
                    session_id,
                    identity,
                    "choose_duration",
                    source_message,
                )

        items = load_booking_items(
            request_id
        )
        pending_actions = load_pending_service_detail_actions(
            session_id
        )
        request_row = load_active_draft_booking_request(
            session_id
        ) or {}

        primary_item = (
            items[0]
            if items
            else None
        )
        primary_name = (
            primary_item.get("service_name")
            if primary_item
            else (
                identities[0].get("service_name")
                if identities
                else "Multiple treatments"
            )
        )
        primary_duration = (
            format_minutes_as_duration(
                primary_item.get("duration_minutes")
            )
            if primary_item
            else None
        )

        lead_data = {
            "treatment": primary_name,
            "duration": primary_duration,
            "preferred_date": request_row.get(
                "preferred_date"
            ),
            "preferred_time": request_row.get(
                "preferred_time"
            ),
            "status": "details_incomplete",
            "summary": (
                "Customer requested multiple treatments in one visit."
            ),
            "conversation_mode": "booking_request",
            "active_category": (
                primary_item.get("category")
                if primary_item
                else identities[0].get("category")
            ),
            "slot_status": "not_checked",
            "next_required_detail": (
                "service_variant"
                if pending_actions
                else (
                    "preferred_date"
                    if not request_row.get("preferred_date")
                    else (
                        "preferred_time"
                        if not request_row.get("preferred_time")
                        else "name"
                    )
                )
            ),
            "notification_sent_at": None,
        }

        save_conversation_overview(
            session_id=session_id,
            lead_data=lead_data,
            source=source,
        )

        return {
            "request_id": request_id,
            "items": items,
            "pending_actions": pending_actions,
            "request": request_row,
        }

    except Exception as error:
        print(
            "Could not create initial multi-service request: "
            f"{error}"
        )

        if request_id is not None:
            try:
                (
                    supabase.table("booking_requests")
                    .delete()
                    .eq("id", request_id)
                    .execute()
                )
            except Exception as cleanup_error:
                print(
                    "Could not clean failed initial multi-service request: "
                    f"{cleanup_error}"
                )

        return None


def format_multi_service_cart_summary(
    request_id: int,
) -> str:
    items = load_booking_items(
        request_id
    )

    if not items:
        return ""

    item_text = "; ".join(
        format_cart_item_summary(item)
        for item in items
    )
    total_duration = sum(
        int(item.get("duration_minutes") or 0)
        for item in items
    )
    total_price = sum(
        int(item.get("price_pence") or 0)
        for item in items
    )

    return (
        f"Your same-visit request currently includes: {item_text}. "
        f"Total so far: {format_price_pence(total_price)}, "
        f"{format_minutes_as_duration(total_duration)}."
    )


def next_multi_service_question(
    session_id: str,
    request_id: int,
) -> str:
    pending_actions = load_pending_service_detail_actions(
        session_id
    )

    if pending_actions:
        return build_pending_service_detail_question(
            pending_actions[0]
        )

    request_row = load_active_draft_booking_request(
        session_id
    ) or {}
    preferred_date = request_row.get(
        "preferred_date"
    )
    preferred_time = request_row.get(
        "preferred_time"
    )

    if not preferred_date and not preferred_time:
        return "Which day and time would suit you?"

    if not preferred_date:
        return (
            f"Which day did you have in mind for "
            f"{format_customer_time(preferred_time)}?"
        )

    if not preferred_time:
        return (
            f"What time would suit you on "
            f"{format_customer_date(preferred_date)}?"
        )

    return (
        "Could I take your name and phone number, please?"
    )


def build_initial_multi_service_reply(
    session_id: str,
    request_id: int,
) -> str:
    summary = format_multi_service_cart_summary(
        request_id
    )
    question = next_multi_service_question(
        session_id,
        request_id,
    )

    if summary:
        return summary + "\n\n" + question

    return (
        "I can help arrange those treatments together.\n\n"
        + question
    )


def resolve_pending_variant_identity(
    action: dict,
    latest_message: str,
) -> dict | None:
    category = str(
        action.get("category") or ""
    )
    service_name = str(
        action.get("service_name") or ""
    )

    if category == "massage":
        service = find_massage_service(
            latest_message
        )

        if service:
            return {
                "category": "massage",
                "service_name": service.get("service_name"),
                "booking_mode": service.get("booking_mode"),
                "fixed_duration_minutes": service.get(
                    "fixed_duration_minutes"
                ),
                "allowed_durations_minutes": service.get(
                    "allowed_durations_minutes"
                ),
            }

    if category == "dermal_fillers":
        service = (
            find_dermal_filler_service(latest_message)
            or find_dermal_filler_service_for_partial_reply(
                service_name,
                latest_message,
            )
        )

        if service:
            return {
                "category": "dermal_fillers",
                "service_name": service.get("service_name"),
                "booking_mode": service.get("booking_mode"),
                "fixed_duration_minutes": service.get(
                    "fixed_duration_minutes"
                ),
                "allowed_durations_minutes": service.get(
                    "allowed_durations_minutes"
                ),
            }

    category_configs = {
        "vitamin_shots": (
            load_vitamin_shot_services,
            {"vitamin", "shot", "shots"},
        ),
        "ultrasound": (
            load_ultrasound_services,
            {"ultrasound"},
        ),
        "microneedling": (
            load_microneedling_services,
            {"microneedling"},
        ),
        "facials": (
            load_facial_services,
            {"facial", "facials"},
        ),
    }

    if category in category_configs:
        loader, tokens = category_configs[category]
        service = find_service_from_short_variant_reply(
            latest_message,
            loader()[0],
            category_tokens=tokens,
        )

        if service:
            return {
                "category": category,
                "service_name": service.get("service_name"),
                "booking_mode": service.get("booking_mode"),
                "fixed_duration_minutes": service.get(
                    "fixed_duration_minutes"
                ),
                "allowed_durations_minutes": service.get(
                    "allowed_durations_minutes"
                ),
            }

    return None


def handle_pending_multi_service_detail_reply(
    session_id: str,
    latest_message: str,
) -> str | None:
    actions = load_pending_service_detail_actions(
        session_id
    )

    if not actions:
        return None

    action = actions[0]
    request_row = load_active_draft_booking_request(
        session_id
    )

    if not request_row:
        return None

    request_id = int(
        request_row["id"]
    )
    action_type = action.get(
        "action_type"
    )

    if action_type == "choose_variant":
        identity = resolve_pending_variant_identity(
            action,
            latest_message,
        )

        if not identity:
            return None

        if pending_action_needs_duration(
            identity
        ):
            update_pending_service_detail_action(
                int(action["id"]),
                identity,
                "choose_duration",
            )

            summary = format_multi_service_cart_summary(
                request_id
            )
            question = build_pending_service_detail_question({
                **action,
                "service_name": identity.get("service_name"),
                "category": identity.get("category"),
                "action_type": "choose_duration",
            })

            return (
                (summary + "\n\n" if summary else "")
                + question
            )

        service_row = load_service_row_for_booking_item(
            identity.get("service_name")
        )

        if not service_row:
            return None

        duration_minutes = service_row.get(
            "fixed_duration_minutes"
        )

        if duration_minutes in [None, ""]:
            return None

        append_booking_item_to_request(
            request_id,
            service_row,
            int(duration_minutes),
        )
        update_pending_booking_action_status(
            int(action["id"]),
            "resolved",
        )

        return build_initial_multi_service_reply(
            session_id,
            request_id,
        )

    if action_type == "choose_duration":
        service_row = load_service_row_for_booking_item(
            action.get("service_name")
        )

        if not service_row:
            return None

        chosen_duration = parse_duration_minutes(
            latest_message
        )
        allowed = {
            int(value)
            for value in (
                service_row.get(
                    "allowed_durations_minutes"
                ) or []
            )
        }

        if (
            not chosen_duration
            or (
                allowed
                and chosen_duration not in allowed
            )
        ):
            options = format_duration_options(
                allowed
            ) if allowed else ""

            if options:
                return (
                    f"Please choose one of the available durations "
                    f"for {service_row.get('service_name')}: "
                    f"{options}."
                )

            return (
                f"Please tell me how long you would like "
                f"{service_row.get('service_name')} for."
            )

        append_booking_item_to_request(
            request_id,
            service_row,
            chosen_duration,
        )
        update_pending_booking_action_status(
            int(action["id"]),
            "resolved",
        )

        return build_initial_multi_service_reply(
            session_id,
            request_id,
        )

    return None


def handle_initial_multi_service_intake(
    session_id: str,
    latest_message: str,
    source: str,
) -> str | None:
    """
    Intercept explicit same-visit multi-service requests before the old
    single-treatment extractor can discard one of the treatments.
    """
    if load_active_draft_booking_request(
        session_id
    ):
        return None

    identities = detect_all_structured_services(
        latest_message
    )

    if len(identities) < 2:
        return None

    if not message_explicitly_requests_same_visit(
        latest_message
    ):
        names = "; ".join(
            identity.get("service_name") or "treatment"
            for identity in identities
        )

        return (
            f"I can help with {names}. Would you like those treatments "
            "during the same visit, or as separate appointment requests?"
        )

    result = create_initial_multi_service_request(
        session_id=session_id,
        identities=identities,
        source_message=latest_message,
        source=source,
    )

    if not result:
        return (
            "I could not start that combined treatment request just now. "
            "Please try again shortly or call 07943319617."
        )

    return build_initial_multi_service_reply(
        session_id,
        int(result["request_id"]),
    )


def advanced_multi_booking_enabled() -> bool:
    """
    Business-level feature flag.

    Advanced same-visit carts and separate appointment drafts remain in the
    codebase, but they are disabled by default for the receptionist MVP.
    """
    return bool(
        load_chatbot_settings().get(
            "enable_multi_booking",
            False,
        )
    )


def format_manual_multi_treatment_names(
    identities: list[dict],
) -> str:
    names = []

    for identity in identities:
        name = str(
            identity.get("service_name")
            or "treatment"
        ).strip()

        if name and name not in names:
            names.append(name)

    if not names:
        return "multiple treatments"

    if len(names) == 1:
        return names[0]

    if len(names) == 2:
        return f"{names[0]} and {names[1]}"

    return (
        ", ".join(names[:-1])
        + f", and {names[-1]}"
    )


def append_manual_multi_treatment_note(
    existing_notes: str | None,
    treatment_names: str,
) -> str:
    note = (
        "Customer mentioned multiple treatments: "
        f"{treatment_names}. "
        "Exact arrangement should be confirmed manually by Veronika."
    )
    current = str(
        existing_notes or ""
    ).strip()

    if not current:
        return note

    if note.lower() in current.lower():
        return current

    return current + " " + note


def handle_simple_multi_treatment_handoff(
    session_id: str,
    latest_message: str,
    source: str,
) -> str | None:
    """
    Default receptionist-mode behaviour for rare multi-treatment enquiries.

    Do not start the advanced cart controller. Record the customer's request
    as a lead note and hand the exact arrangement to Veronika for manual
    confirmation.
    """
    identities = detect_all_structured_services(
        latest_message
    )

    if len(identities) < 2:
        return None

    treatment_names = format_manual_multi_treatment_names(
        identities
    )
    existing = get_existing_conversation(
        session_id
    )
    existing_summary = str(
        existing.get("summary") or ""
    ).strip()
    summary = (
        f"Customer is interested in {treatment_names}. "
        "Exact multi-treatment arrangement requires manual follow-up."
    )

    if existing_summary and summary.lower() not in existing_summary.lower():
        summary = existing_summary + " " + summary

    lead_data = {
        "treatment": (
            existing.get("treatment")
            or identities[0].get("service_name")
            or "Multiple treatments"
        ),
        "duration": existing.get("duration"),
        "preferred_date": (
            parse_relative_date_from_text(
                latest_message,
                reference_date=existing.get(
                    "preferred_date"
                ),
            )
            or existing.get("preferred_date")
        ),
        "preferred_time": (
            parse_time_from_text(latest_message)
            or existing.get("preferred_time")
        ),
        "name": existing.get("name"),
        "phone": existing.get("phone"),
        "notes": append_manual_multi_treatment_note(
            existing.get("notes"),
            treatment_names,
        ),
        "summary": summary,
        "status": "details_incomplete",
        "conversation_mode": "booking_request",
        "slot_status": "not_checked",
        "next_required_detail": "preferred_date",
        "notification_sent_at": None,
    }

    save_conversation_overview(
        session_id=session_id,
        lead_data=lead_data,
        source=source,
    )

    return (
        f"Of course. I have noted that you are interested in "
        f"{treatment_names}. Veronika will confirm the exact arrangement "
        "with you manually. Which day and time would suit you, and could "
        "I take your name and phone number?"
    )


@app.post("/chat")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    request_source = normalise_source(request.source)
    multi_booking_enabled = (
        advanced_multi_booking_enabled()
    )

    save_message(
        session_id=request.session_id,
        role="user",
        content=request.message,
        source=request_source,
    )

    existing_before_processing = get_existing_conversation(
        request.session_id
    )

    if (
        multi_booking_enabled
        and customer_asks_for_booking_summary(
            request.message
        )
    ):
        reply = build_customer_booking_summary(
            request.session_id
        )

        save_message(
            session_id=request.session_id,
            role="assistant",
            content=reply,
            source=request_source,
        )

        return {"reply": reply}

    if (
        booking_request_is_complete(existing_before_processing)
        and (
            not multi_booking_enabled
            or not has_pending_booking_actions(
                request.session_id
            )
        )
        and is_simple_acknowledgement(request.message)
    ):
        reply = "Thanks. The therapist will confirm the appointment shortly."

        save_message(
            session_id=request.session_id,
            role="assistant",
            content=reply,
            source=request_source,
        )

        return {"reply": reply}

    response = supabase.table("documents").select("content").execute()

    live_massage_context, massage_from_supabase = (
        format_live_massage_services_context()
    )
    live_vitamin_context, vitamin_shots_from_supabase = (
        format_live_vitamin_shot_services_context()
    )
    live_ultrasound_context, ultrasound_from_supabase = (
        format_live_ultrasound_services_context()
    )
    live_body_treatment_context, body_treatments_from_supabase = (
        format_live_body_treatment_services_context()
    )
    live_microneedling_context, microneedling_from_supabase = (
        format_live_microneedling_services_context()
    )
    live_facial_context, facials_from_supabase = (
        format_live_facial_services_context()
    )
    live_dermal_filler_context, dermal_fillers_from_supabase = (
        format_live_dermal_filler_services_context()
    )

    context_rows = []

    for item in response.data or []:
        content = str(item.get("content") or "")

        # Dashboard hours are authoritative. Ignore old free-text opening-hours
        # rows so they cannot override live owner settings.
        if re.match(r"(?i)^\s*opening\s+hours\s*:", content):
            continue

        # The structured public.services table is authoritative for massage.
        # Prevent older free-text massage rows from conflicting with live rows.
        if (
            massage_from_supabase
            and re.match(
                r"(?i)^\s*(?:relaxing|swedish|hot\s+stone|deep\s+tissue)\s+massage\s*:",
                content,
            )
        ):
            continue

        if (
            vitamin_shots_from_supabase
            and re.match(
                r"(?i)^\s*(?:b12\s+vitamin\s+shot|vitamin\s+c\s+shot)\s*:",
                content,
            )
        ):
            continue

        if (
            ultrasound_from_supabase
            and re.match(
                r"(?i)^\s*ultrasound\s*:",
                content,
            )
        ):
            continue

        if (
            body_treatments_from_supabase
            and re.match(
                r"(?i)^\s*(?:body\s+sculpting\s+treatment|"
                r"electronic\s+muscle\s+stimulation\s*\(ems\))\s*:",
                content,
            )
        ):
            continue

        if (
            microneedling_from_supabase
            and re.match(
                r"(?i)^\s*microneedling\s+for\s+1\s+area",
                content,
            )
        ):
            continue

        if (
            facials_from_supabase
            and re.match(
                r"(?i)^\s*(?:deep\s+facial\s+cleanse|"
                r"radio\s+frequency\s+facial|"
                r"facial\s+treatment\s+for\s+young\s+skin|"
                r"hydraface\s+facial\s+treatment|"
                r"hydrafacial(?:\s+for\s+90\s+minutes)?)\s*:",
                content,
            )
        ):
            continue

        if (
            dermal_fillers_from_supabase
            and re.match(
                r"(?i)^\s*dermal\s+filler",
                content,
            )
        ):
            continue

        context_rows.append(content)

    context_rows.append(live_massage_context)
    context_rows.append(live_vitamin_context)
    context_rows.append(live_ultrasound_context)
    context_rows.append(live_body_treatment_context)
    context_rows.append(live_microneedling_context)
    context_rows.append(live_facial_context)
    context_rows.append(live_dermal_filler_context)

    context = "\n".join(context_rows) if context_rows else "No information available."

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

    if multi_booking_enabled:
        pending_multi_service_reply = (
            handle_pending_multi_service_detail_reply(
                session_id=request.session_id,
                latest_message=request.message,
            )
        )

        if pending_multi_service_reply:
            save_message(
                session_id=request.session_id,
                role="assistant",
                content=pending_multi_service_reply,
                source=request_source,
            )

            return {"reply": pending_multi_service_reply}

        initial_multi_service_reply = (
            handle_initial_multi_service_intake(
                session_id=request.session_id,
                latest_message=request.message,
                source=request_source,
            )
        )

        if initial_multi_service_reply:
            save_message(
                session_id=request.session_id,
                role="assistant",
                content=initial_multi_service_reply,
                source=request_source,
            )

            return {"reply": initial_multi_service_reply}

    else:
        manual_multi_treatment_reply = (
            handle_simple_multi_treatment_handoff(
                session_id=request.session_id,
                latest_message=request.message,
                source=request_source,
            )
        )

        if manual_multi_treatment_reply:
            save_message(
                session_id=request.session_id,
                role="assistant",
                content=manual_multi_treatment_reply,
                source=request_source,
            )

            return {"reply": manual_multi_treatment_reply}

    extracted_lead = extract_lead_data(conversation_history)
    existing_lead = get_existing_conversation(request.session_id)
    explicit_cart_intent = (
        detect_treatment_cart_intent(
            session_id=request.session_id,
            latest_message=request.message,
            existing_lead=existing_lead,
        )
        if multi_booking_enabled
        else None
    )
    pending_cart_resolution = (
        (
            None
            if explicit_cart_intent
            else detect_pending_cart_action_resolution(
                session_id=request.session_id,
                latest_message=request.message,
            )
        )
        if multi_booking_enabled
        else None
    )
    cart_intent = (
        explicit_cart_intent
        or pending_cart_resolution
    )
    merged_lead = merge_lead_data(existing_lead, extracted_lead)

    previous_assistant_message = next(
        (
            msg.content
            for msg in reversed(request.history)
            if msg.role == "assistant"
        ),
        "",
    )

    if customer_rejected_calendar_alternatives(
        request.message,
        previous_assistant_message,
    ):
        merged_lead["preferred_date"] = None
        merged_lead["preferred_time"] = None
        merged_lead["status"] = "details_incomplete"
        merged_lead["notification_sent_at"] = None

        save_conversation_overview(
            session_id=request.session_id,
            lead_data=merged_lead,
            source=request_source,
        )

        reply = (
            "No problem. Which day and time would suit you instead? "
            "I can check the available slots."
        )

        save_message(
            session_id=request.session_id,
            role="assistant",
            content=reply,
            source=request_source,
        )

        return {"reply": reply}

    merged_lead = apply_latest_message_fallbacks(
        merged_lead,
        request.message,
        previous_assistant_message,
    )

    merged_lead = preserve_existing_booking_details(
        existing_lead,
        merged_lead,
        request.message,
        previous_assistant_message,
    )

    merged_lead = apply_fixed_treatment_details(
        merged_lead,
        request.message,
    )

    merged_lead = apply_dynamic_massage_details(
        merged_lead,
        request.message,
    )

    merged_lead = apply_dynamic_vitamin_shot_details(
        merged_lead,
        request.message,
    )

    merged_lead = apply_dynamic_ultrasound_details(
        merged_lead,
        request.message,
    )

    merged_lead = apply_dynamic_body_treatment_details(
        merged_lead,
        request.message,
    )

    merged_lead = apply_dynamic_microneedling_details(
        merged_lead,
        request.message,
    )

    merged_lead = apply_dynamic_facial_details(
        merged_lead,
        request.message,
    )

    merged_lead = apply_dynamic_dermal_filler_details(
        merged_lead,
        request.message,
    )

    if cart_intent:
        merged_lead = restore_existing_primary_treatment(
            existing_lead,
            merged_lead,
        )
        service_switch = None
    else:
        merged_lead, service_switch = apply_generic_service_switch_controller(
            existing_lead=existing_lead,
            lead_data=merged_lead,
            latest_message=request.message,
        )

    merged_lead = normalise_misplaced_preferred_fields(
        merged_lead,
        request.message,
    )

    merged_lead = apply_authoritative_schedule_fields(
        merged_lead,
        existing_lead,
        request.message,
    )

    booking_flow_active = conversation_has_booking_intent(
        conversation_history,
        request.message,
        previous_assistant_message,
        merged_lead,
    )

    merged_lead, today_is_unavailable = clear_expired_same_day_preference(
        merged_lead,
    )

    merged_lead = align_status_with_booking_intent(
        merged_lead,
        booking_flow_active,
    )

    merged_lead = apply_conversation_controller_state(
        merged_lead,
        booking_flow_active=booking_flow_active,
        calendar_result={"status": "not_checked"},
    )

    merged_lead, validation_issue = validate_latest_customer_details(
        merged_lead,
        request.message,
        previous_assistant_message,
        skip_contact_validation=(
            bool(service_switch)
            or bool(cart_intent)
            or customer_message_interrupts_contact_collection(
                request.message
            )
        ),
    )

    save_conversation_overview(
        session_id=request.session_id,
        lead_data=merged_lead,
        source=request_source,
    )

    if validation_issue:
        if validation_issue.startswith("invalid_duration:"):
            options = validation_issue.split(":", 1)[1]
            reply = f"Please choose one of the available durations: {options}."
        elif validation_issue == "invalid_name":
            reply = "Please enter your name."
        else:
            reply = "Please enter a valid phone number, including the area code."

        save_message(
            session_id=request.session_id,
            role="assistant",
            content=reply,
            source=request_source,
        )

        return {"reply": reply}

    if booking_flow_active:
        calendar_lead = calendar_lead_for_cart_action(
            merged_lead,
            cart_intent,
        )
        calendar_result = check_requested_calendar_slot(
            calendar_lead,
            request.message,
        )

        calendar_should_be_visible = should_surface_calendar_result(
            request.message,
            existing_lead,
            merged_lead,
        )
    else:
        calendar_result = {
            "status": "not_requested",
            "message": (
                "The customer is asking for information only. "
                "Do not start collecting booking details yet."
            ),
        }
        calendar_should_be_visible = False

    calendar_context = format_calendar_result(
        calendar_result,
        calendar_should_be_visible,
    )

    if calendar_result.get("status") in {
        "outside_hours",
        "closed_day",
        "past",
        "busy",
    }:
        merged_lead["preferred_time"] = None

        if calendar_result.get("clear_date"):
            merged_lead["preferred_date"] = None

        merged_lead["status"] = "details_incomplete"
        merged_lead["notification_sent_at"] = None

        save_conversation_overview(
            session_id=request.session_id,
            lead_data=merged_lead,
            source=request_source,
        )

    merged_lead = apply_conversation_controller_state(
        merged_lead,
        booking_flow_active=booking_flow_active,
        calendar_result=calendar_result,
    )

    save_conversation_overview(
        session_id=request.session_id,
        lead_data=merged_lead,
        source=request_source,
    )

    cart_result = None

    if (
        cart_intent
        and cart_intent.get("action") == "add_same_visit"
    ):
        cart_result = sync_same_visit_treatment_item(
            session_id=request.session_id,
            lead_data=merged_lead,
            cart_intent=cart_intent,
        )

        if (
            cart_result
            and cart_result.get("status") in {
                "added",
                "already_present",
            }
        ):
            if cart_intent.get("pending_action_id"):
                update_pending_booking_action_status(
                    int(cart_intent["pending_action_id"]),
                    "resolved",
                )

            selected_identity = (
                cart_intent.get("selected_identity")
                or {}
            )
            resolve_matching_pending_booking_actions(
                request.session_id,
                selected_identity.get("service_name"),
            )
    elif not cart_intent:
        sync_single_treatment_draft_request(
            session_id=request.session_id,
            lead_data=merged_lead,
            booking_flow_active=booking_flow_active,
        )

    missing_fields = (
        [
            field
            for field in REQUIRED_BOOKING_FIELDS
            if merged_lead.get(field) in [None, ""]
        ]
        if booking_flow_active
        else []
    )

    confirmed_details = format_confirmed_details(merged_lead)
    missing_details = (
        ", ".join(field.replace("_", " ") for field in missing_fields)
        if missing_fields
        else "None"
    )

    next_question_hint = build_next_question_hint(
        missing_fields,
        merged_lead,
    )
    customer_date_label = format_customer_date(
        merged_lead.get("preferred_date")
    )
    live_business_hours = format_live_business_hours()
    service_switch_context = format_service_switch_context(
        service_switch
    )

    if (
        cart_intent
        and cart_intent.get("action")
        == "clarify_visit_structure"
    ):
        save_pending_booking_action(
            session_id=request.session_id,
            selected_identity=(
                cart_intent.get("selected_identity")
                or {}
            ),
            source_message=request.message,
        )

        reply = build_visit_structure_clarification_reply(
            cart_intent.get("selected_identity") or {}
        )

        save_message(
            session_id=request.session_id,
            role="assistant",
            content=reply,
            source=request_source,
        )

        return {"reply": reply}

    if (
        cart_intent
        and cart_intent.get("action")
        == "cancel_pending_action"
    ):
        pending_action = (
            cart_intent.get("pending_action") or {}
        )

        if pending_action.get("id"):
            update_pending_booking_action_status(
                int(pending_action["id"]),
                "cancelled",
            )

        reply = (
            "No problem. I have removed that unfinished treatment "
            "request. Your existing appointment request is unchanged."
        )

        save_message(
            session_id=request.session_id,
            role="assistant",
            content=reply,
            source=request_source,
        )

        return {"reply": reply}

    if (
        cart_intent
        and cart_intent.get("action")
        == "create_separate_request"
    ):
        separate_result = create_separate_draft_booking_request(
            session_id=request.session_id,
            pending_action=(
                cart_intent.get("pending_action")
                or {}
            ),
            existing_lead=existing_lead,
        )

        if (
            separate_result
            and separate_result.get("status") == "created"
        ):
            reply = build_separate_request_started_reply(
                separate_result,
                request.session_id,
            )
        elif (
            separate_result
            and separate_result.get("status") == "needs_duration"
        ):
            service_name = str(
                separate_result.get("selected_service_name")
                or "the additional treatment"
            )
            reply = (
                f"Before I start a separate request for {service_name}, "
                "how long would you like the session for?"
            )
        else:
            reply = (
                "I could not start the separate appointment request just "
                "now. Your original request is still saved unchanged. "
                "Please try again shortly or call 07943319617."
            )

        save_message(
            session_id=request.session_id,
            role="assistant",
            content=reply,
            source=request_source,
        )

        return {"reply": reply}

    if cart_result:
        if cart_result.get("status") == "needs_duration":
            reply = (
                "How long would you like the additional treatment for? "
                "Once you tell me, I can add it to the same visit and "
                "recheck the full appointment time."
            )
        else:
            reply = build_same_visit_added_reply(
                cart_result=cart_result,
                lead_data=merged_lead,
                calendar_result=calendar_result,
                missing_fields=missing_fields,
            )
            reply = append_pending_action_reminder(
                reply,
                request.session_id,
            )

        save_message(
            session_id=request.session_id,
            role="assistant",
            content=reply,
            source=request_source,
        )

        return {"reply": reply}

    if (
        is_simple_acknowledgement(request.message)
        and booking_request_is_complete(merged_lead)
        and (
            not multi_booking_enabled
            or not has_pending_booking_actions(
                request.session_id
            )
        )
    ):
        reply = "Thanks. The therapist will confirm the appointment shortly."

        save_message(
            session_id=request.session_id,
            role="assistant",
            content=reply,
            source=request_source,
        )

        background_tasks.add_task(
            send_booking_notification,
            request.session_id
        )

        return {"reply": reply}

    if (
        not booking_flow_active
        and is_simple_acknowledgement(request.message)
    ):
        reply = "You're welcome."

        save_message(
            session_id=request.session_id,
            role="assistant",
            content=reply,
            source=request_source,
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
- Answer the customer's question first.
- If BOOKING FLOW ACTIVE says No, treat the message as an informational enquiry. Do not ask for the customer's name, phone number, preferred date, preferred time, or duration. You may briefly ask whether they would like to book only when it feels natural.
- If BOOKING FLOW ACTIVE says Yes, follow NEXT QUESTION TARGET. It may combine up to two closely related missing details in one concise question when sensible.
- Follow NEXT QUESTION TARGET closely.
- If SERVICE SWITCH CONTEXT says the customer changed treatment, acknowledge the change briefly, use the new treatment only, and do not reuse an old duration that no longer applies.
- Do not repeat a treatment name immediately after the customer has just chosen it unless clarification is genuinely needed.
- Do not repeat prices, dates, times, durations, or availability unless the customer asks or the information has changed.
- Do not list duration options unless duration is the next missing detail.
- If duration is the next missing detail, use NEXT QUESTION TARGET. Never offer massage durations for a non-massage treatment.
- If a treatment has one fixed duration listed under CONFIRMED CUSTOMER DETAILS, do not ask the customer to choose a session length. Continue directly to the next missing booking detail.
- For a generic vitamin-shot enquiry, ask whether the customer wants B12 or Vitamin C. Both are fixed 15-minute treatments. Do not ask the customer to choose a duration.
- Do not repeat hello twice.
- Avoid filler such as "I've noted that down", "just to confirm", "we offer", "to proceed", and "the requested time currently looks available" unless genuinely needed.
- Use the conversation history to understand short replies.
- Do not ask the customer to repeat information they already gave.
- Never ask again for any detail listed under CONFIRMED CUSTOMER DETAILS.
- Ask naturally for only the missing details. Asking for two closely related details together is encouraged when NEXT QUESTION TARGET combines them.
- Follow the CALENDAR AVAILABILITY RESULT exactly.
- LIVE DASHBOARD WORKING HOURS are the only authoritative opening hours. Ignore any older opening-hours text elsewhere in the business information.
- The "Massage services (authoritative live catalogue)" block is the only authoritative source for massage names, durations, and prices. Ignore older conflicting massage text elsewhere.
- The "Vitamin-shot services (authoritative live catalogue)" block is the only authoritative source for vitamin-shot variants, durations, and prices. Ignore older conflicting vitamin-shot text elsewhere.
- The "Ultrasound services (authoritative live catalogue)" block is the only authoritative source for ultrasound options, durations, and prices. For the package, check availability for the first 45-minute session only; the therapist arranges the remaining sessions manually.
- The "Body-treatment services (authoritative live catalogue)" block is the only authoritative source for Body Sculpting and EMS durations and prices. Ignore older conflicting body-treatment text elsewhere.
- The "Microneedling services (authoritative live catalogue)" block is the only authoritative source for microneedling variants, durations, and prices. Ignore older conflicting microneedling text elsewhere.
- The "Facial services (authoritative live catalogue)" block is the only authoritative source for facial names, durations, and prices. Ignore older conflicting facial text elsewhere.
- The "Dermal-filler services (authoritative live catalogue)" block is the only authoritative source for dermal-filler variants, durations, and prices. Ignore older conflicting dermal-filler text elsewhere.
- When listing fixed-price service options or when the customer asks how much a structured fixed-price service costs, state the price clearly before continuing the booking flow.
- If the requested day is closed or the requested slot is unavailable, do not collect contact details yet. If duration is missing, ask for duration first so the backend can calculate valid alternatives. If alternatives are supplied, ask the customer to choose one first.
- If a valid date and time are already confirmed, never ask "What time would you prefer?" again.
- If CALENDAR AVAILABILITY RESULT says Visibility: silent, do not mention calendar availability.
- When speaking to the customer, use CUSTOMER-FACING DATE LABEL. Say "today" or "tomorrow" where appropriate. Never show an ISO date such as 2026-06-01 to the customer unless they explicitly ask for the exact date.
- If a requested slot is busy, outside working hours, or in the past, say it once briefly and ask for another preferred time. Never treat that time as accepted.
- If a requested slot appears free, say it once briefly and clearly state that the therapist still needs to confirm.
- Never accept an unsupported session duration. Massage durations must match the allowed options.
- Never suggest, infer, default, or invent a specific date or time unless it comes from the customer's message or from calendar alternatives. If the customer has not supplied a date, ask for it.
- Treat explicit customer dates such as "14th of June", "14 June", "June 14th", and "14/06" as valid dates. Do not ask for the day again after one of these has been supplied.
- If the calendar cannot be checked, do not guess. Say the therapist will check and confirm.
- Never reveal private calendar event details.
- Never say a booking is confirmed, reserved, or completed.
- Never say the therapist will confirm shortly while any required booking detail is still missing.
- Settle an acceptable preferred date and time before asking for the customer's name or phone number. This order is mandatory.
- If the customer rejects proposed alternative slots, ask for a different preferred day and time. Do not collect contact details yet.
- A complete booking request requires: treatment, duration, name, phone number, preferred date, and preferred time.
- If asked "am I booked?", answer clearly: "Not yet. The therapist still needs to confirm the appointment."
- If the customer only says "ok", "thanks", "fantastic", or a similar acknowledgement after giving all details, do not reinterpret it as booking information and do not alter the date or time. Reply briefly: "Thanks. The therapist will confirm the appointment shortly."
- When the customer asks a side question while also providing a booking detail, answer the side question briefly and then ask for the next missing booking detail.
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
- If unsure, suggest calling 07943319617.
- Authoritative contact details: phone 07943319617; address 25 Albion Place, Leeds, LS1 6JS.
- If the business information below contains a conflicting phone number or address, ignore the older conflicting value and use the authoritative contact details above.

CONFIRMED CUSTOMER DETAILS:
{confirmed_details}

MISSING REQUIRED DETAILS:
{missing_details}

NEXT QUESTION TARGET:
{next_question_hint}

BOOKING FLOW ACTIVE:
{"Yes" if booking_flow_active else "No"}

SERVICE SWITCH CONTEXT:
{service_switch_context}

TODAY BOOKING CONTEXT:
{"The business cannot accept any more appointments today. Never suggest today and never ask what time today." if today_is_unavailable or not business_can_accept_more_bookings_today(parse_duration_minutes(merged_lead.get("duration"))) else "The business may still have time remaining today, but never assume the customer wants today unless they explicitly say so."}

CUSTOMER-FACING DATE LABEL:
{customer_date_label}

CALENDAR AVAILABILITY RESULT:
{calendar_context}

LIVE DASHBOARD WORKING HOURS:
{live_business_hours}

Business information:
{context}

Conversation history:
{conversation_history}

Reply as Receptionist:
"""

    completion = groq_client.chat.completions.create(
        model=LIVE_CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        reasoning_effort=LIVE_CHAT_REASONING_EFFORT,
        include_reasoning=False,
        temperature=0.5,
        max_completion_tokens=LIVE_CHAT_MAX_COMPLETION_TOKENS,
    )

    reply = completion.choices[0].message.content
    reply = normalise_reply_line_breaks(reply)
    reply = sanitise_booking_claims(reply)
    reply = remove_invalid_fixed_duration_questions(
        reply,
        merged_lead,
    )
    reply = remove_premature_booking_questions(
        reply,
        booking_flow_active,
    )
    reply = remove_invalid_today_time_questions(
        reply,
        today_is_unavailable,
    )
    reply = remove_redundant_confirmed_field_questions(
        reply,
        merged_lead,
    )

    if booking_flow_active:
        reply = remove_false_handover_confirmation(
            reply,
            missing_fields,
        )

        if calendar_result.get("status") in {
            "busy_needs_duration",
            "needs_duration_for_alternatives",
        }:
            reply = build_duration_needed_for_alternatives_reply(
                merged_lead,
                calendar_result,
            )
        elif calendar_result.get("suggestions"):
            if calendar_result.get("status") == "suggestions":
                reply = build_direct_calendar_suggestions_reply(
                    calendar_result,
                )
            else:
                reply = build_calendar_alternative_reply(
                    calendar_result,
                )
        else:
            reply = append_single_deterministic_followup(
                reply,
                merged_lead,
                missing_fields,
            )

    reply = finalise_dynamic_service_reply(
        reply=reply,
        latest_message=request.message,
        lead_data=merged_lead,
        missing_fields=missing_fields,
        calendar_result=calendar_result,
    )

    if service_switch:
        reply = build_service_switch_reply(
            lead_data=merged_lead,
            missing_fields=missing_fields,
            calendar_result=calendar_result,
        )

    if multi_booking_enabled:
        reply = suppress_contact_collection_while_pending(
            reply,
            request.session_id,
        )

    save_message(
        session_id=request.session_id,
        role="assistant",
        content=reply,
        source=request_source,
    )

    background_tasks.add_task(
        send_booking_notification,
        request.session_id
    )

    return {"reply": reply}
