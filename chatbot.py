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


def format_duration_options(durations: set[int]) -> str:
    ordered = sorted(durations)

    if len(ordered) == 1:
        return f"{ordered[0]} minutes"

    return ", ".join(str(value) for value in ordered[:-1]) + f", or {ordered[-1]} minutes"


# Services with one fixed duration. These are resolved deterministically
# before the AI decides which booking detail to ask for next.
FIXED_TREATMENT_DETAILS = [
    {
        "name": "Body Sculpting Treatment",
        "duration": "30 minutes",
        "aliases": ["body sculpting", "body sculpting treatment"],
    },
    {
        "name": "Electronic Muscle Stimulation (EMS)",
        "duration": "1 hour",
        "aliases": ["electronic muscle stimulation", "ems"],
    },
    {
        "name": "Microneedling for 1 Area (Stretch Marks)",
        "duration": "1 hour 15 minutes",
        "aliases": ["microneedling stretch marks", "microneedling for stretch marks"],
    },
    {
        "name": "Microneedling for 1 Area (Face and Neck)",
        "duration": "1 hour 15 minutes",
        "aliases": ["microneedling face and neck", "microneedling for face and neck"],
    },
    {
        "name": "Deep Facial Cleanse",
        "duration": "1 hour",
        "aliases": ["deep facial cleanse"],
    },
    {
        "name": "Radio Frequency Facial",
        "duration": "1 hour",
        "aliases": ["radio frequency facial"],
    },
    {
        "name": "Facial Treatment for Young Skin",
        "duration": "1 hour",
        "aliases": ["facial treatment for young skin", "young skin facial"],
    },
    {
        "name": "Hydraface Facial Treatment",
        "duration": "1 hour",
        "aliases": ["hydraface facial treatment", "hydraface facial", "hydraface"],
    },
    {
        "name": "Hydrafacial for 90 Minutes",
        "duration": "1 hour 30 minutes",
        "aliases": ["90 minute hydrafacial", "hydrafacial 90 minutes", "hydrafacial for 90 minutes"],
    },
    {
        "name": "Dermal Filler (Lip Filler, 0.5 ml)",
        "duration": "45 minutes",
        "aliases": ["0.5 ml lip filler", "lip filler 0.5 ml", "0.5ml lip filler", "lip filler 0.5ml"],
    },
    {
        "name": "Dermal Filler (Marionette Lines, 0.5 ml)",
        "duration": "45 minutes",
        "aliases": ["0.5 ml marionette lines", "marionette lines 0.5 ml", "0.5ml marionette lines"],
    },
    {
        "name": "Dermal Filler (Nasolabial Folds, 0.5 ml)",
        "duration": "45 minutes",
        "aliases": ["0.5 ml nasolabial folds", "nasolabial folds 0.5 ml", "0.5ml nasolabial folds"],
    },
    {
        "name": "Dermal Filler (Lip Filler, 1 ml)",
        "duration": "1 hour",
        "aliases": ["1 ml lip filler", "lip filler 1 ml", "1ml lip filler", "lip filler 1ml"],
    },
    {
        "name": "Dermal Filler (Marionette Lines, 1 ml)",
        "duration": "1 hour",
        "aliases": ["1 ml marionette lines", "marionette lines 1 ml", "1ml marionette lines"],
    },
    {
        "name": "Dermal Filler (Nasolabial Folds, 1 ml)",
        "duration": "1 hour",
        "aliases": ["1 ml nasolabial folds", "nasolabial folds 1 ml", "1ml nasolabial folds"],
    },
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
    cleaned = re.sub(r"[^a-z0-9.]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


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


def validate_latest_customer_details(
    lead_data: dict,
    latest_message: str,
    previous_assistant_message: str,
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

    if "name" in lower_previous:
        if not is_valid_customer_name(result.get("name")):
            result["name"] = None
            result["status"] = "details_incomplete"
            result["notification_sent_at"] = None
            return result, "invalid_name"

    if "phone" in lower_previous or "number" in lower_previous:
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
        return "Would you like a B12 vitamin shot or a Vitamin C shot?"

    if needs_ultrasound_variant(lead_data):
        return (
            "Would you like a single ultrasound session or the package "
            "of 5 sessions?"
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
    Remove booking-form questions generated by the LLM.

    The backend appends exactly one deterministic follow-up afterwards.
    """
    cleaned = str(reply or "")

    question_patterns = [
        r"(?is)(?:\s*\n\s*)?[^?]*(?:how long|which duration|what duration)[^?]*\?",
        r"(?is)(?:\s*\n\s*)?[^?]*(?:which|what) (?:day|date|time)[^?]*\?",
        r"(?is)(?:\s*\n\s*)?[^?]*(?:preferred day|preferred date|preferred time)[^?]*\?",
        r"(?is)(?:\s*\n\s*)?[^?]*(?:your name|phone number|contact number|mobile number)[^?]*\?",
        r"(?is)(?:\s*\n\s*)?[^?]*(?:which treatment|what treatment|treatment option|which service)[^?]*\?",
    ]

    for pattern in question_patterns:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"(?im)^\s*(?:and|or|,\s*and|—|-)+\s*$", "", cleaned)
    cleaned = re.sub(r"(?i)(?:,|\s)\s*and\s*$", "", cleaned.strip())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned.strip()


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

        if not slot_overlaps_busy_period(
            candidate,
            candidate_end,
            busy_periods,
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
        format_customer_slot(
            value.date().isoformat(),
            value.strftime("%H:%M"),
        )
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


@app.post("/chat")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    request_source = normalise_source(request.source)

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
        booking_request_is_complete(existing_before_processing)
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

        context_rows.append(content)

    context_rows.append(live_massage_context)
    context_rows.append(live_vitamin_context)
    context_rows.append(live_ultrasound_context)

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

    extracted_lead = extract_lead_data(conversation_history)
    existing_lead = get_existing_conversation(request.session_id)
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

    merged_lead, validation_issue = validate_latest_customer_details(
        merged_lead,
        request.message,
        previous_assistant_message,
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
        calendar_result = check_requested_calendar_slot(
            merged_lead,
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

    if (
        is_simple_acknowledgement(request.message)
        and booking_request_is_complete(merged_lead)
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
            reply = build_calendar_alternative_reply(
                calendar_result,
            )
        else:
            reply = append_single_deterministic_followup(
                reply,
                merged_lead,
                missing_fields,
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
