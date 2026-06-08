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


def read_positive_int_environment_variable(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip())
    except (AttributeError, ValueError):
        return default

    return value if value > 0 else default


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

    # Common mobile typing variations. Keep this intentionally conservative:
    # only normalise phrases that clearly mean "tomorrow".
    lower_message = re.sub(
        r"\btom\s+more\b|\btomorow\b|\btommorow\b|\btomorroww+\b",
        "tomorrow",
        lower_message,
    )

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

    descriptive_time_match = re.search(
        r"\b(?:at\s*|around\s*)?"
        r"(1[0-2]|0?[1-9])"
        r"(?:[:.]([0-5]\d))?\s*"
        r"(?:o['’]?clock\s*)?"
        r"(?:in\s+the\s+)?"
        r"(morning|afternoon|evening)\b",
        lower_message,
    )

    if descriptive_time_match:
        hour = int(descriptive_time_match.group(1))
        minute = descriptive_time_match.group(2) or "00"
        period = descriptive_time_match.group(3)

        if period in {"afternoon", "evening"} and hour != 12:
            hour += 12
        elif period == "morning" and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute}"

    oclock_match = re.search(
        r"\b(?:at\s*|around\s*)?"
        r"(1[0-2]|0?[1-9])\s*"
        r"o['’]?clock\b",
        lower_message,
    )

    if oclock_match:
        hour = int(oclock_match.group(1))

        # Salon appointments are daytime appointments. With no am/pm marker,
        # interpret 1-7 o'clock as afternoon.
        if 1 <= hour <= 7:
            hour += 12

        return f"{hour:02d}:00"

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

    bare_hour_match = re.search(
        r"\b(?:at|around)\s+(1[0-2]|0?[1-9])\b",
        lower_message,
    )

    if bare_hour_match:
        hour = int(bare_hour_match.group(1))

        if 1 <= hour <= 7:
            hour += 12

        return f"{hour:02d}:00"

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


def recover_schedule_from_customer_history(
    lead_data: dict,
    conversation_history: str,
    latest_message: str,
) -> dict:
    """
    Recover an explicitly supplied date or time from recent customer messages.

    This is needed when a customer gives the schedule before completing a
    service variant, for example:
      customer: "lip filler tomorrow at 9 am"
      assistant: "0.5 ml or 1 ml?"
      customer: "0.5"

    The newest explicit customer value wins. Assistant suggestions are never
    treated as confirmed details.
    """
    result = dict(lead_data)
    customer_messages = []

    for line in str(conversation_history or "").splitlines():
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith(("customer:", "user:")):
            customer_messages.append(
                stripped.split(":", 1)[1].strip()
            )

    if (
        not customer_messages
        or customer_messages[-1].strip()
        != str(latest_message or "").strip()
    ):
        customer_messages.append(
            str(latest_message or "").strip()
        )

    recovered_date = None
    recovered_time = None

    for message in reversed(customer_messages):
        if recovered_date is None:
            recovered_date = parse_relative_date_from_text(
                message
            )

        if recovered_time is None:
            recovered_time = parse_time_from_text(
                message
            )

        if recovered_date and recovered_time:
            break

    if result.get("preferred_date") in [None, ""] and recovered_date:
        result["preferred_date"] = recovered_date

    if result.get("preferred_time") in [None, ""] and recovered_time:
        result["preferred_time"] = recovered_time

    return result


def explicit_duration_minutes_from_reply(
    latest_message: str,
    previous_assistant_message: str = "",
    treatment: str | None = None,
) -> int | None:
    """
    Extract a duration deterministically from the newest customer message.

    Normal phrases such as "120 minutes" are parsed directly. Compact replies
    such as "60 tomorrow at 12:00" are accepted only when the previous bot
    message asked how long the treatment should be and the number matches one
    of the configured durations for that treatment.
    """
    direct = parse_duration_minutes(
        latest_message
    )

    if direct is not None:
        return direct

    lower_previous = str(
        previous_assistant_message or ""
    ).lower()

    if (
        "how long" not in lower_previous
        and "duration" not in lower_previous
    ):
        return None

    allowed = allowed_durations_for_treatment(
        treatment
    )

    if not allowed:
        allowed = {30, 45, 60, 75, 90, 105, 120}

    for match in re.finditer(
        r"(?<!\d)(\d{1,3})(?!\d)",
        str(latest_message or ""),
    ):
        value = int(match.group(1))

        if value in allowed:
            return value

    return None


def message_looks_like_schedule_input(
    message: str,
) -> bool:
    """
    Detect customer replies that are probably schedule details or corrections,
    not names.
    """
    cleaned = normalise_service_match_text(
        message
    )

    return bool(
        parse_relative_date_from_text(message)
        or parse_time_from_text(message)
        or re.search(
            r"\b(?:today|tomorrow|tom\s+more|morning|afternoon|evening|"
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            cleaned,
        )
    )


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


def needs_massage_variant(
    lead_data: dict,
) -> bool:
    """
    A generic massage enquiry must resolve the massage type before duration.
    """
    return (
        normalise_treatment_name(
            lead_data.get("treatment")
        )
        == "massage"
    )


def build_massage_variant_prompt() -> str:
    services, _ = load_massage_services()
    names = [
        str(service.get("service_name") or "").strip()
        for service in services
        if str(service.get("service_name") or "").strip()
    ]

    if not names:
        names = [
            "Relaxing Massage",
            "Swedish Massage",
            "Deep Tissue Massage",
            "Hot Stone Massage",
        ]

    return (
        "Which massage would you like: "
        + format_natural_list(names)
        + "?"
    )


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
    "massage": "massage",
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
        ("massage", "Massage", ["massage", "massages"]),
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


def fixed_treatment_metadata(
    treatment: str | None,
) -> dict | None:
    """
    Temporary fallback for services that have not yet been migrated into
    public.services.

    New businesses should store these values in public.services. This fallback
    remains only so older configured treatments continue to work during the
    migration.
    """
    cleaned = normalise_service_match_text(
        treatment
    )

    if not cleaned:
        return None

    for item in FIXED_TREATMENT_DETAILS:
        candidates = [
            item.get("name"),
            *(item.get("aliases") or []),
        ]

        for candidate in candidates:
            candidate_cleaned = normalise_service_match_text(
                candidate
            )

            if (
                candidate_cleaned
                and (
                    cleaned == candidate_cleaned
                    or candidate_cleaned in cleaned
                    or cleaned in candidate_cleaned
                )
            ):
                minutes = parse_duration_minutes(
                    item.get("duration")
                )

                return {
                    "service_id": None,
                    "service_name": item.get("name"),
                    "category": "legacy_fixed_treatments",
                    "booking_mode": "fixed_duration",
                    "fixed_duration_minutes": minutes,
                    "allowed_durations_minutes": (
                        [minutes] if minutes else []
                    ),
                    "price_pence": item.get("price_pence"),
                    "price_by_duration": None,
                    "source": "legacy_fallback",
                }

    return None


def load_service_metadata_from_table(
    treatment: str | None,
) -> dict | None:
    """
    Generic public.services lookup used by the reusable controller.

    This is deliberately category-agnostic so another business can add a
    fixed-duration service in Supabase without requiring another Python patch.
    """
    cleaned = normalise_service_match_text(
        treatment
    )

    if not cleaned:
        return None

    try:
        response = (
            supabase.table("services")
            .select(
                "id, category, service_name, aliases, booking_mode, "
                "fixed_duration_minutes, allowed_durations_minutes, "
                "price_pence, price_by_duration, is_active"
            )
            .eq("business_slug", BUSINESS_SLUG)
            .eq("is_active", True)
            .execute()
        )

        matches = []

        for row in response.data or []:
            candidates = [
                row.get("service_name"),
                *(row.get("aliases") or []),
            ]
            best_score = 0

            for candidate in candidates:
                candidate_cleaned = normalise_service_match_text(
                    candidate
                )

                if not candidate_cleaned:
                    continue

                if cleaned == candidate_cleaned:
                    best_score = max(best_score, 10000)
                elif candidate_cleaned in cleaned:
                    best_score = max(
                        best_score,
                        5000 + len(candidate_cleaned),
                    )
                elif cleaned in candidate_cleaned:
                    best_score = max(
                        best_score,
                        1000 + len(cleaned),
                    )

            if best_score:
                matches.append((best_score, row))

        if not matches:
            return None

        matches.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        if (
            len(matches) > 1
            and matches[0][0] == matches[1][0]
        ):
            return None

        row = matches[0][1]
        fixed_minutes = row.get(
            "fixed_duration_minutes"
        )

        if fixed_minutes not in [None, ""]:
            fixed_minutes = int(fixed_minutes)

        allowed = [
            int(value)
            for value in (
                row.get("allowed_durations_minutes")
                or []
            )
        ]

        return {
            "service_id": row.get("id"),
            "service_name": row.get("service_name"),
            "category": row.get("category"),
            "booking_mode": row.get("booking_mode"),
            "fixed_duration_minutes": fixed_minutes,
            "allowed_durations_minutes": allowed,
            "price_pence": row.get("price_pence"),
            "price_by_duration": row.get("price_by_duration"),
            "source": "services_table",
        }

    except Exception as error:
        print(
            "Could not load generic service metadata: "
            f"{error}"
        )

    return None


def authoritative_service_metadata(
    treatment: str | None,
) -> dict | None:
    """
    Return the authoritative service configuration for the active treatment.

    Prefer public.services. Fall back to older hardcoded fixed-treatment
    metadata only for services that have not yet been migrated.
    """
    return (
        load_service_metadata_from_table(
            treatment
        )
        or fixed_treatment_metadata(
            treatment
        )
    )


def metadata_match_tokens(
    value: str | None,
) -> set[str]:
    """
    Tokenise a service phrase for reusable alias matching.

    Broad filler words are ignored so a natural reply such as
    "I'm looking for my chin" can resolve the unique alias "double chin"
    without hard-coding the exact sentence.
    """
    ignored = {
        "a", "an", "and", "area", "for", "i", "id", "im", "in",
        "injection", "injections", "interested", "looking", "my", "of",
        "please", "session", "the", "to", "treatment", "want", "would",
    }

    return {
        token
        for token in normalise_service_match_text(value).split()
        if token and token not in ignored
    }


def service_metadata_match_score(
    value: str | None,
    service_name: str | None,
    aliases: list[str] | None,
) -> int:
    cleaned = normalise_service_match_text(
        value
    )

    if not cleaned:
        return 0

    value_tokens = metadata_match_tokens(
        value
    )
    best_score = 0

    for candidate in [
        service_name,
        *(aliases or []),
    ]:
        candidate_cleaned = normalise_service_match_text(
            candidate
        )

        if not candidate_cleaned:
            continue

        candidate_tokens = metadata_match_tokens(
            candidate
        )

        if cleaned == candidate_cleaned:
            best_score = max(best_score, 10000)
            continue

        if candidate_cleaned in cleaned:
            best_score = max(
                best_score,
                7000 + len(candidate_cleaned),
            )
            continue

        if (
            value_tokens
            and candidate_tokens
            and value_tokens.issubset(candidate_tokens)
        ):
            best_score = max(
                best_score,
                3000 + len(value_tokens) * 100 - len(candidate_tokens),
            )
            continue

        overlap = value_tokens.intersection(
            candidate_tokens
        )

        if overlap:
            best_score = max(
                best_score,
                1000 + len(overlap) * 100 - len(candidate_tokens),
            )

    return best_score


def resolve_service_metadata_from_message(
    message: str | None,
) -> dict | None:
    """
    Resolve one configured service from a natural customer message.

    This generic matcher is reusable across businesses:
    - public.services rows are preferred;
    - aliases stored in Supabase are honoured;
    - equal best matches are rejected rather than guessed;
    - legacy fixed entries are only a migration fallback.
    """
    if not normalise_service_match_text(
        message
    ):
        return None

    matches = []

    try:
        response = (
            supabase.table("services")
            .select(
                "id, category, service_name, aliases, booking_mode, "
                "fixed_duration_minutes, allowed_durations_minutes, "
                "price_pence, price_by_duration, is_active"
            )
            .eq("business_slug", BUSINESS_SLUG)
            .eq("is_active", True)
            .execute()
        )

        for row in response.data or []:
            score = service_metadata_match_score(
                message,
                row.get("service_name"),
                list(row.get("aliases") or []),
            )

            if score:
                matches.append((
                    score,
                    {
                        "service_id": row.get("id"),
                        "service_name": row.get("service_name"),
                        "category": row.get("category"),
                        "booking_mode": row.get("booking_mode"),
                        "fixed_duration_minutes": row.get(
                            "fixed_duration_minutes"
                        ),
                        "allowed_durations_minutes": [
                            int(value)
                            for value in (
                                row.get(
                                    "allowed_durations_minutes"
                                )
                                or []
                            )
                        ],
                        "price_pence": row.get("price_pence"),
                        "price_by_duration": row.get(
                            "price_by_duration"
                        ),
                        "source": "services_table",
                    },
                ))

    except Exception as error:
        print(
            "Could not resolve service aliases from public.services: "
            f"{error}"
        )

    for item in FIXED_TREATMENT_DETAILS:
        score = service_metadata_match_score(
            message,
            item.get("name"),
            list(item.get("aliases") or []),
        )

        if not score:
            continue

        minutes = parse_duration_minutes(
            item.get("duration")
        )

        matches.append((
            score,
            {
                "service_id": None,
                "service_name": item.get("name"),
                "category": "legacy_fixed_treatments",
                "booking_mode": "fixed_duration",
                "fixed_duration_minutes": minutes,
                "allowed_durations_minutes": (
                    [minutes] if minutes else []
                ),
                "price_pence": item.get("price_pence"),
                "price_by_duration": None,
                "source": "legacy_fallback",
            },
        ))

    if not matches:
        return None

    matches.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    if (
        len(matches) > 1
        and matches[0][0] == matches[1][0]
        and matches[0][1].get("service_name")
        != matches[1][1].get("service_name")
    ):
        return None

    return matches[0][1]


def hydrate_configured_service_defaults(
    state: dict,
) -> dict:
    """
    Apply deterministic service defaults before deciding what to ask next.

    A configured service with one fixed duration must never ask the customer
    to choose a duration. The extractor and responder do not control this.
    """
    result = dict(state)
    metadata = authoritative_service_metadata(
        result.get("treatment")
    )

    if not metadata:
        return result

    service_name = metadata.get(
        "service_name"
    )

    if service_name:
        result["treatment"] = service_name

    fixed_minutes = metadata.get(
        "fixed_duration_minutes"
    )

    if fixed_minutes not in [None, ""]:
        result["duration"] = format_minutes_as_duration(
            int(fixed_minutes)
        )

    if metadata.get("category"):
        result["active_category"] = metadata.get(
            "category"
        )

    return result


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

    if message_looks_like_schedule_input(cleaned):
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


def normalise_reply_line_breaks(reply: str) -> str:
    """Store and send real line breaks instead of visible backslash-n text."""
    return str(reply or "").replace("\\r\\n", "\n").replace("\\n", "\n")


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


def format_natural_list(items: list[str]) -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]

    if not cleaned:
        return ""

    if len(cleaned) == 1:
        return cleaned[0]

    if len(cleaned) == 2:
        return f"{cleaned[0]} or {cleaned[1]}"

    return ", ".join(cleaned[:-1]) + f", or {cleaned[-1]}"


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


def build_schedule_first_question(lead_data: dict) -> str:
    """
    Ask only for the unsettled schedule detail. This is deliberately separate
    from the LLM so the booking order cannot drift.
    """
    treatment = lead_data.get("treatment")
    duration = lead_data.get("duration")
    preferred_date = lead_data.get("preferred_date")
    preferred_time = lead_data.get("preferred_time")

    if treatment in [None, ""]:
        return ""

    if needs_massage_variant(lead_data):
        return build_massage_variant_prompt()

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
        r"(?is)(?:\s*\n\s*)?(?:could|can|may|would) i (?:have|take) your preferred (?:day|date|time)[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:could|can|may|would) i (?:take|have) your (?:name|phone number|contact number|mobile number)[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:what(?:'s| is)) your (?:name|phone number|contact number|mobile number)[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:which treatment|what treatment|which service|which treatment option)[^?]*\?",
        r"(?is)(?:\s*\n\s*)?(?:which|what) massage(?: option)?(?: would you like)?[^?]*\?",
    ]

    for pattern in question_patterns:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"(?im)^\s*(?:and|or|,\s*and|—|-)+\s*$", "", cleaned)
    cleaned = re.sub(r"(?i)(?:,|\s)\s*and\s*$", "", cleaned.strip())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned.strip()


def normalise_reply_for_comparison(value: str) -> str:
    cleaned = normalise_service_match_text(
        value
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def collapse_repeated_reply_content(
    reply: str,
) -> str:
    """
    Remove accidental repetition without rewriting useful information.

    This is intentionally conservative:
    - exact duplicate paragraphs are collapsed;
    - repeated therapist-confirmation sentences are collapsed;
    - factual availability sentences remain intact.
    """
    cleaned = normalise_reply_line_breaks(
        reply
    ).strip()

    if not cleaned:
        return cleaned

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", cleaned)
        if paragraph.strip()
    ]
    unique_paragraphs = []
    seen_paragraphs = set()

    for paragraph in paragraphs:
        key = normalise_reply_for_comparison(
            paragraph
        )

        if key and key in seen_paragraphs:
            continue

        seen_paragraphs.add(key)
        unique_paragraphs.append(paragraph)

    cleaned = "\n\n".join(unique_paragraphs)

    # Collapse repeated handover wording even when the second sentence begins
    # with "Thanks." or uses slightly different punctuation.
    handover_pattern = re.compile(
        r"(?is)(?:\s*\n\s*)?"
        r"(?:thanks[.!]?\s*)?"
        r"(?:the therapist will confirm(?: the appointment)? shortly[.!]?|"
        r"the therapist will confirm(?: it| the request| the appointment)? shortly[.!]?|"
        r"the therapist still needs to confirm(?: the appointment| the request)?[.!]?)"
    )

    matches = list(
        handover_pattern.finditer(cleaned)
    )

    if len(matches) > 1:
        first = matches[0]
        before = cleaned[:first.end()]
        after = cleaned[first.end():]

        after = handover_pattern.sub(
            "",
            after,
        )
        cleaned = before + after

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)

    return cleaned.strip()


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


def has_pending_booking_actions(
    session_id: str,
) -> bool:
    return bool(
        load_pending_booking_actions(session_id)
    )


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


# ============================================================================
# STATE-FIRST TWO-PASS RECEPTIONIST CONTROLLER
#
# Pass 1: AI extracts a hidden JSON patch from the newest customer message.
# Python validates and merges that patch into Supabase-backed canonical state.
# Python then checks services, opening hours, and Google Calendar.
# Pass 2: AI writes a natural reply using the refreshed verified state.
# The customer never sees the hidden extractor JSON.
# ============================================================================

CORE_STATE_FIELDS = [
    "name",
    "phone",
    "treatment",
    "duration",
    "preferred_date",
    "preferred_time",
    "notes",
    "summary",
]

PATCHABLE_FIELDS = {
    "name",
    "phone",
    "treatment",
    "duration",
    "preferred_date_expression",
    "preferred_time",
    "notes_append",
}

BOOKING_INTENTS = {
    "booking_request",
    "change_request",
    "booking_detail",
    "availability_request",
}


def serialisable_saved_state(state: dict) -> dict:
    return {
        field: state.get(field)
        for field in [
            *CORE_STATE_FIELDS,
            "status",
            "conversation_mode",
            "active_category",
            "slot_status",
            "next_required_detail",
        ]
    }


def build_authoritative_services_context() -> str:
    blocks = [
        format_live_massage_services_context()[0],
        format_live_vitamin_shot_services_context()[0],
        format_live_ultrasound_services_context()[0],
        format_live_body_treatment_services_context()[0],
        format_live_microneedling_services_context()[0],
        format_live_facial_services_context()[0],
        format_live_dermal_filler_services_context()[0],
    ]

    return "\n\n".join(
        block for block in blocks if block
    )


def load_business_documents_context() -> str:
    try:
        response = (
            supabase.table("documents")
            .select("content")
            .execute()
        )

        rows = []

        for item in response.data or []:
            content = str(item.get("content") or "").strip()

            if not content:
                continue

            # Structured services and live dashboard hours are authoritative.
            # Ignore old rows that could conflict with current configuration.
            if re.match(r"(?i)^\s*opening\s+hours\s*:", content):
                continue

            rows.append(content)

        return "\n".join(rows)

    except Exception as error:
        print(f"Could not load business documents: {error}")
        return ""


def compact_recent_history(history: list[ChatMessage], limit: int = 12) -> str:
    lines = []

    for message in history[-limit:]:
        role = "Customer" if message.role == "user" else "Receptionist"
        lines.append(f"{role}: {message.content}")

    return "\n".join(lines)


def empty_extractor_result() -> dict:
    return {
        "intent": "unclear",
        "state_patch": {},
        "mentioned_treatments": [],
        "customer_question": None,
        "is_correction": False,
    }


def normalise_extractor_result(raw: dict | None) -> dict:
    result = empty_extractor_result()

    if not isinstance(raw, dict):
        return result

    intent = str(raw.get("intent") or "unclear").strip().lower()
    result["intent"] = intent

    patch = raw.get("state_patch") or {}

    if isinstance(patch, dict):
        result["state_patch"] = {
            key: value
            for key, value in patch.items()
            if key in PATCHABLE_FIELDS
            and value not in [None, ""]
        }

    mentioned = raw.get("mentioned_treatments") or []

    if isinstance(mentioned, list):
        result["mentioned_treatments"] = [
            str(value).strip()
            for value in mentioned
            if str(value).strip()
        ][:6]

    question = raw.get("customer_question")

    if isinstance(question, dict):
        result["customer_question"] = {
            "topic": str(question.get("topic") or "").strip() or None,
            "type": str(question.get("type") or "").strip() or None,
        }
    elif isinstance(question, str) and question.strip():
        result["customer_question"] = {
            "topic": question.strip(),
            "type": "general",
        }

    result["is_correction"] = bool(
        raw.get("is_correction")
    )

    return result


def extract_hidden_state_patch(
    current_state: dict,
    latest_message: str,
    history: list[ChatMessage],
    services_context: str,
) -> dict:
    """
    Pass 1: return hidden structured data only.

    The extractor suggests a patch. It never writes to Supabase and never
    decides calendar availability. Python validates every accepted value.
    """
    prompt = f"""
You are the hidden structured-data extractor for a salon receptionist chatbot.
The customer never sees your output.

Read the CURRENT SAVED STATE, recent conversation, and newest customer message.
Return JSON only.

Your job:
- extract only NEW or CHANGED facts supplied by the customer;
- preserve previously saved facts by omitting unchanged fields;
- identify corrections and treatment switches;
- identify customer questions separately from booking-state changes;
- never invent a date, time, treatment, name, phone number, price, duration, or availability;
- never claim that a slot is free or unavailable;
- preserve relative date language such as "today", "tomorrow", or "next Thursday" in preferred_date_expression;
- use a 24-hour HH:MM value for preferred_time only when the customer clearly provided a time;
- use the configured treatment names where possible;
- when a customer answers a pending variant naturally, resolve it where possible. Example: active Lip Filler + "the 0.5 will be great" means Lip Filler 0.5 ml.

Allowed intent values:
information_question, service_menu, booking_request, booking_detail,
change_request, acknowledgement, unclear.

Return exactly this shape:
{{
  "intent": "unclear",
  "state_patch": {{
    "treatment": null,
    "duration": null,
    "preferred_date_expression": null,
    "preferred_time": null,
    "name": null,
    "phone": null,
    "notes_append": null
  }},
  "mentioned_treatments": [],
  "customer_question": {{"topic": null, "type": null}},
  "is_correction": false
}}

CURRENT SAVED STATE:
{json.dumps(serialisable_saved_state(current_state), ensure_ascii=False, indent=2)}

AUTHORITATIVE STRUCTURED SERVICES:
{services_context}

RECENT CONVERSATION:
{compact_recent_history(history)}

NEWEST CUSTOMER MESSAGE:
{latest_message}
"""

    try:
        completion = groq_client.chat.completions.create(
            model=LEAD_EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        raw = json.loads(
            completion.choices[0].message.content
        )

        return normalise_extractor_result(raw)

    except Exception as error:
        print(f"Hidden state extraction failed: {error}")
        return empty_extractor_result()


def merge_notes(existing_notes: str | None, addition: str | None) -> str | None:
    current = str(existing_notes or "").strip()
    new_note = str(addition or "").strip()

    if not new_note:
        return current or None

    if new_note.lower() in current.lower():
        return current or new_note

    return (current + " " + new_note).strip()


def canonical_treatment_from_text(
    candidate: str | None,
    latest_message: str,
    existing_treatment: str | None,
) -> str | None:
    """Resolve a real configured service or a safe unresolved category."""
    texts = [candidate, latest_message]

    for text_value in texts:
        if not text_value:
            continue

        metadata = resolve_service_metadata_from_message(
            str(text_value)
        )

        if metadata and metadata.get("service_name"):
            return str(
                metadata["service_name"]
            )

        identity = resolve_latest_structured_service(
            str(text_value),
            existing_treatment=existing_treatment,
        )

        if identity and identity.get("service_name"):
            return str(identity["service_name"])

    for text_value in texts:
        if not text_value:
            continue

        partial = resolve_latest_generic_or_partial_service(
            str(text_value)
        )

        if partial and partial.get("service_name"):
            return str(partial["service_name"])

    return None


def apply_structured_service_resolution(
    state: dict,
    latest_message: str,
) -> dict:
    """
    Resolve configured variants and defaults in one controlled stage.
    Existing category-specific helpers are reused only as canonicalisers.
    """
    result = dict(state)

    result = apply_dynamic_massage_details(result, latest_message)
    result = apply_dynamic_vitamin_shot_details(result, latest_message)
    result = apply_dynamic_ultrasound_details(result, latest_message)
    result = apply_dynamic_body_treatment_details(result, latest_message)
    result = apply_dynamic_microneedling_details(result, latest_message)
    result = apply_dynamic_facial_details(result, latest_message)
    result = apply_dynamic_dermal_filler_details(result, latest_message)

    metadata = resolve_service_metadata_from_message(
        latest_message
    )

    if metadata and metadata.get("service_name"):
        result["treatment"] = metadata.get(
            "service_name"
        )

        fixed_minutes = metadata.get(
            "fixed_duration_minutes"
        )

        if fixed_minutes not in [None, ""]:
            result["duration"] = format_minutes_as_duration(
                int(fixed_minutes)
            )

        if metadata.get("category"):
            result["active_category"] = metadata.get(
                "category"
            )

    return result


def duration_is_valid_for_treatment(
    treatment: str | None,
    duration_value: str | None,
) -> bool:
    minutes = parse_duration_minutes(duration_value)

    if minutes is None:
        return True

    allowed = allowed_durations_for_treatment(
        treatment
    )

    return not allowed or minutes in allowed


def apply_validated_state_patch(
    existing_state: dict,
    extractor_result: dict,
    latest_message: str,
    history: list[ChatMessage],
) -> dict:
    """
    Validate and merge the hidden extractor patch into canonical state.

    Explicit newest facts win. Existing facts remain unless a validated
    customer correction replaces them.
    """
    result = {
        field: existing_state.get(field)
        for field in CORE_STATE_FIELDS
    }
    result.update({
        "status": existing_state.get("status"),
        "conversation_mode": existing_state.get("conversation_mode"),
        "active_category": existing_state.get("active_category"),
        "slot_status": existing_state.get("slot_status"),
        "next_required_detail": existing_state.get("next_required_detail"),
        "notification_sent_at": existing_state.get("notification_sent_at"),
    })

    patch = extractor_result.get("state_patch") or {}

    # Deterministic parsers are a backup and validator for the extractor.
    date_expression = (
        patch.get("preferred_date_expression")
        or latest_message
    )
    parsed_date = parse_relative_date_from_text(
        str(date_expression),
        reference_date=result.get("preferred_date"),
    )
    parsed_time = parse_time_from_text(
        str(patch.get("preferred_time") or latest_message)
    )
    parsed_phone = parse_phone_from_text(
        str(patch.get("phone") or latest_message)
    )
    parsed_duration = explicit_duration_minutes_from_reply(
        str(patch.get("duration") or latest_message),
        treatment=result.get("treatment"),
    )

    if parsed_date:
        result["preferred_date"] = parsed_date

    if parsed_time:
        result["preferred_time"] = parsed_time

    if parsed_phone and is_valid_phone_number(parsed_phone):
        result["phone"] = parsed_phone

    name_candidate = str(patch.get("name") or "").strip()

    if (
        name_candidate
        and is_valid_customer_name(name_candidate)
        and not message_looks_like_schedule_input(name_candidate)
    ):
        result["name"] = name_candidate

    candidate_treatment = canonical_treatment_from_text(
        patch.get("treatment"),
        latest_message,
        result.get("treatment"),
    )

    previous_treatment = result.get("treatment")

    if candidate_treatment:
        result["treatment"] = candidate_treatment

    # Service helpers resolve short replies such as "B12", "Relaxing", and
    # natural filler amount replies using the active category.
    result = apply_structured_service_resolution(
        result,
        latest_message,
    )

    if parsed_duration is not None:
        candidate_duration = format_minutes_as_duration(
            parsed_duration
        )

        if duration_is_valid_for_treatment(
            result.get("treatment"),
            candidate_duration,
        ):
            result["duration"] = candidate_duration

    if (
        previous_treatment
        and result.get("treatment")
        and normalise_treatment_name(previous_treatment)
        != normalise_treatment_name(result.get("treatment"))
        and not duration_is_valid_for_treatment(
            result.get("treatment"),
            result.get("duration"),
        )
    ):
        result["duration"] = None

    # Apply defaults from the owner-editable services table. Legacy fixed
    # metadata remains a temporary fallback for categories not migrated yet.
    result = hydrate_configured_service_defaults(
        result
    )

    result["notes"] = merge_notes(
        result.get("notes"),
        patch.get("notes_append"),
    )

    # Recover previously supplied customer schedule details after a later
    # variant choice. Assistant suggestions are never used as saved facts.
    result = recover_schedule_from_customer_history(
        result,
        compact_recent_history(history),
        latest_message,
    )

    return result


def has_booking_intent_state(
    existing_state: dict,
    extractor_result: dict,
    latest_message: str,
) -> bool:
    intent = str(
        extractor_result.get("intent") or ""
    ).strip().lower()

    if existing_state.get("conversation_mode") == "booking_request":
        return True

    if intent in BOOKING_INTENTS:
        return True

    if any(
        re.search(pattern, latest_message.lower())
        for pattern in BOOKING_INTENT_PATTERNS
    ):
        return True

    return False


def compute_next_required_detail(
    state: dict,
    booking_flow_active: bool,
) -> str | None:
    if not booking_flow_active:
        return None

    if state.get("treatment") in [None, ""]:
        return "treatment"

    if (
        needs_massage_variant(state)
        or needs_vitamin_shot_variant(state)
        or needs_ultrasound_variant(state)
        or needs_microneedling_variant(state)
        or needs_facial_variant(state)
        or needs_dermal_filler_variant(state)
    ):
        return "service_variant"

    if state.get("duration") in [None, ""]:
        return "duration"

    if state.get("preferred_date") in [None, ""]:
        return "preferred_date"

    if state.get("preferred_time") in [None, ""]:
        return "preferred_time"

    if (
        state.get("name") in [None, ""]
        and state.get("phone") in [None, ""]
    ):
        return "name_and_phone"

    if state.get("name") in [None, ""]:
        return "name"

    if state.get("phone") in [None, ""]:
        return "phone"

    return "handoff"


def render_next_question(
    state: dict,
    next_required_detail: str | None,
) -> str:
    if not next_required_detail or next_required_detail == "handoff":
        return ""

    if next_required_detail == "treatment":
        return "Which treatment are you interested in?"

    if next_required_detail == "service_variant":
        return build_schedule_first_question(state)

    if next_required_detail == "duration":
        allowed = allowed_durations_for_treatment(
            state.get("treatment")
        )

        if allowed:
            return (
                "How long would you like the session for: "
                + format_duration_options(allowed)
                + "?"
            )

        return "How long would you like the session for?"

    if next_required_detail == "preferred_date":
        return "Which day would you prefer?"

    if next_required_detail == "preferred_time":
        return (
            "What time would suit you on "
            + format_customer_date(
                state.get("preferred_date")
            )
            + "?"
        )

    if next_required_detail == "name_and_phone":
        return "Could I take your name and phone number, please?"

    if next_required_detail == "name":
        return "Could I take your name, please?"

    if next_required_detail == "phone":
        return "Could I take your phone number, please?"

    return ""


def apply_canonical_controller_state(
    state: dict,
    booking_flow_active: bool,
    calendar_result: dict | None = None,
) -> dict:
    result = hydrate_configured_service_defaults(
        state
    )
    next_detail = compute_next_required_detail(
        result,
        booking_flow_active,
    )

    result["conversation_mode"] = (
        "booking_request"
        if booking_flow_active
        else "browsing"
    )
    result["next_required_detail"] = next_detail
    result["slot_status"] = normalise_controller_slot_status(
        calendar_result or {"status": "not_checked"}
    )
    result["status"] = calculate_lead_status(
        result,
        result.get("status"),
    )

    if result["status"] != "booking_request_complete":
        result["notification_sent_at"] = None

    return result


def verified_calendar_result_for_state(
    state: dict,
    booking_flow_active: bool,
    latest_message: str,
) -> dict:
    if not booking_flow_active:
        return {
            "status": "not_requested",
            "message": "No booking slot check is required.",
        }

    if (
        state.get("preferred_date") in [None, ""]
        or state.get("preferred_time") in [None, ""]
    ):
        return {
            "status": "not_checked",
            "message": "A date and time are not both saved yet.",
        }

    return check_requested_calendar_slot(
        state,
        latest_message,
    )


def safe_calendar_customer_text(
    state: dict,
    calendar_result: dict,
) -> str:
    status = str(
        calendar_result.get("status") or "not_checked"
    )

    if calendar_result.get("suggestions"):
        return build_calendar_alternative_reply(
            calendar_result
        )

    if status == "free":
        return (
            format_customer_slot(
                state.get("preferred_date"),
                state.get("preferred_time"),
            )
            + " currently appears free."
        )

    if status == "unknown":
        if (
            state.get("preferred_date") not in [None, ""]
            and state.get("preferred_time") not in [None, ""]
        ):
            return (
                "I could not verify "
                + format_customer_slot(
                    state.get("preferred_date"),
                    state.get("preferred_time"),
                )
                + " automatically. Veronika will check the availability "
                "manually before confirming."
            )

        return (
            "Veronika will need to check the availability manually."
        )

    return ""


def build_responder_context(
    state: dict,
    extractor_result: dict,
    calendar_result: dict,
    next_question: str,
    business_context: str,
    services_context: str,
    latest_message: str,
    history: list[ChatMessage],
) -> str:
    return f"""
You are Veronika's human-sounding receptionist assistant.
Write a short, warm and natural reply to the customer's newest message.

IMPORTANT LOCKED RULES:
- The saved state below is authoritative.
- Never invent, infer, rewrite, or calculate a date, weekday, time, price, duration, or calendar slot.
- Never mention availability unless it is explicitly supplied in VERIFIED CALENDAR CUSTOMER TEXT.
- Never claim that an appointment is confirmed. Veronika confirms manually.
- Do not ask workflow questions. Python appends the one allowed next question separately.
- Answer a customer's side question briefly when possible.
- Do not repeat information unnecessarily.
- If the customer asks what services are offered, answer using the supplied business information and service catalogue.

NEWEST CUSTOMER MESSAGE:
{latest_message}

HIDDEN EXTRACTOR RESULT:
{json.dumps(extractor_result, ensure_ascii=False, indent=2)}

CURRENT SAVED STATE AFTER VALIDATION:
{json.dumps(serialisable_saved_state(state), ensure_ascii=False, indent=2)}

VERIFIED CALENDAR CUSTOMER TEXT:
{safe_calendar_customer_text(state, calendar_result) or "None"}

ONE ALLOWED NEXT QUESTION APPENDED BY PYTHON:
{next_question or "None"}

RECENT CONVERSATION:
{compact_recent_history(history)}

AUTHORITATIVE STRUCTURED SERVICES:
{services_context}

BUSINESS INFORMATION:
{business_context}

Return only the natural reply body. Do not append a workflow question.
"""


def generate_natural_reply_body(
    state: dict,
    extractor_result: dict,
    calendar_result: dict,
    next_question: str,
    business_context: str,
    services_context: str,
    latest_message: str,
    history: list[ChatMessage],
) -> str:
    prompt = build_responder_context(
        state,
        extractor_result,
        calendar_result,
        next_question,
        business_context,
        services_context,
        latest_message,
        history,
    )

    try:
        completion = groq_client.chat.completions.create(
            model=LIVE_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort=LIVE_CHAT_REASONING_EFFORT,
            include_reasoning=False,
            temperature=0.35,
            max_completion_tokens=LIVE_CHAT_MAX_COMPLETION_TOKENS,
        )

        reply = normalise_reply_line_breaks(
            completion.choices[0].message.content
        )
        reply = sanitise_booking_claims(reply)
        reply = strip_model_booking_questions(reply)
        reply = collapse_repeated_reply_content(reply)
        return reply.strip()

    except Exception as error:
        print(f"Natural responder failed: {error}")
        return ""


def strip_unverified_calendar_claims(
    reply: str,
    calendar_result: dict,
) -> str:
    """
    Remove model-written calendar claims unless Python has verified that the
    wording is allowed. Calendar text is appended deterministically later.
    """
    cleaned = str(reply or "")
    status = str(
        calendar_result.get("status") or "not_checked"
    )

    if status == "free":
        return cleaned

    patterns = [
        r"(?is)(?:\s*\n\s*)?i(?:'ll| will)? have the therapist check that slot and confirm[.!]?",
        r"(?is)(?:\s*\n\s*)?the therapist will check (?:that|the) slot and confirm[.!]?",
        r"(?is)(?:\s*\n\s*)?the therapist will check[^.!?]*(?:and )?confirm[.!]?",
        r"(?is)(?:\s*\n\s*)?i(?:'ll| will) have the therapist check[^.!?]*(?:and )?confirm[.!]?",
        r"(?is)(?:\s*\n\s*)?(?:that|the requested) slot (?:looks|appears|is) free[.!]?",
        r"(?is)(?:\s*\n\s*)?(?:that|the requested) time (?:looks|appears|is) free[.!]?",
    ]

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
        )

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def compose_verified_customer_reply(
    state: dict,
    extractor_result: dict,
    calendar_result: dict,
    next_required_detail: str | None,
    business_context: str,
    services_context: str,
    latest_message: str,
    history: list[ChatMessage],
) -> str:
    next_question = render_next_question(
        state,
        next_required_detail,
    )

    # Suggested slots are fully deterministic because the model must never
    # invent alternatives or rewrite date labels.
    if calendar_result.get("suggestions"):
        return collapse_repeated_reply_content(
            build_calendar_alternative_reply(
                calendar_result
            )
        )

    body = generate_natural_reply_body(
        state,
        extractor_result,
        calendar_result,
        next_question,
        business_context,
        services_context,
        latest_message,
        history,
    )
    calendar_text = safe_calendar_customer_text(
        state,
        calendar_result,
    )

    parts = []

    body = strip_unverified_calendar_claims(
        body,
        calendar_result,
    )

    if body:
        parts.append(body)

    if calendar_text and calendar_text.lower() not in body.lower():
        parts.append(calendar_text)

    if next_required_detail == "handoff":
        handoff = (
            "Veronika will confirm the appointment with you shortly."
        )

        if handoff.lower() not in "\n\n".join(parts).lower():
            parts.append(handoff)

    elif next_question:
        combined = "\n\n".join(parts).lower()

        if next_question.lower() not in combined:
            parts.append(next_question)

    if not parts:
        parts.append(
            next_question
            or "Thanks. Veronika will follow up with you shortly."
        )

    return collapse_repeated_reply_content(
        "\n\n".join(parts)
    )


def add_manual_multi_treatment_note(
    state: dict,
    extractor_result: dict,
) -> dict:
    result = dict(state)
    mentioned = []

    for value in extractor_result.get("mentioned_treatments") or []:
        cleaned = str(value).strip()

        if cleaned and cleaned not in mentioned:
            mentioned.append(cleaned)

    if len(mentioned) < 2:
        return result

    note = (
        "Customer mentioned multiple treatments: "
        + ", ".join(mentioned)
        + ". Exact arrangement should be confirmed manually by Veronika."
    )
    result["notes"] = merge_notes(
        result.get("notes"),
        note,
    )

    return result


def simple_request_price_pence(
    metadata: dict | None,
    duration_minutes: int | None,
) -> int | None:
    if not metadata:
        return None

    fixed_price = metadata.get("price_pence")

    if fixed_price not in [None, ""]:
        return int(fixed_price)

    prices = metadata.get("price_by_duration") or {}

    if duration_minutes is None:
        return None

    value = prices.get(
        str(duration_minutes)
    )

    if value in [None, ""]:
        return None

    return int(value)


def load_simple_draft_request(
    session_id: str,
) -> dict | None:
    try:
        response = (
            supabase.table("booking_requests")
            .select("*")
            .eq("session_id", session_id)
            .eq("request_status", "draft")
            .order("request_number")
            .limit(1)
            .execute()
        )

        if response.data:
            return response.data[0]

    except Exception as error:
        print(
            "Could not load simple draft request: "
            f"{error}"
        )

    return None


def next_simple_request_number(
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
                response.data[0].get(
                    "request_number"
                ) or 0
            ) + 1

    except Exception as error:
        print(
            "Could not calculate simple request number: "
            f"{error}"
        )

    return 1


def sync_simple_single_request_projection(
    session_id: str,
    state: dict,
    booking_flow_active: bool,
):
    """
    Mirror the canonical single-treatment lead into the relational dashboard.

    This is intentionally a projection only. It does not implement carts,
    separate appointment drafts, or multi-booking workflow logic.
    """
    if not booking_flow_active:
        return

    metadata = authoritative_service_metadata(
        state.get("treatment")
    )

    if not metadata:
        return

    duration_minutes = parse_duration_minutes(
        state.get("duration")
    )

    if duration_minutes is None:
        return

    price_pence = simple_request_price_pence(
        metadata,
        duration_minutes,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    request_row = load_simple_draft_request(
        session_id
    )

    request_payload = {
        "session_id": session_id,
        "request_status": "draft",
        "appointment_structure": "single_service",
        "preferred_date": state.get("preferred_date"),
        "preferred_time": state.get("preferred_time"),
        "slot_status": state.get("slot_status") or "not_checked",
        "item_count": 1,
        "total_duration_minutes": duration_minutes,
        "total_price_pence": price_pence or 0,
        "missing_detail": state.get("next_required_detail"),
        "updated_at": now_iso,
    }

    try:
        if request_row:
            request_id = int(
                request_row["id"]
            )
            (
                supabase.table("booking_requests")
                .update(request_payload)
                .eq("id", request_id)
                .execute()
            )
        else:
            request_payload["request_number"] = (
                next_simple_request_number(
                    session_id
                )
            )
            response = (
                supabase.table("booking_requests")
                .insert(request_payload)
                .execute()
            )

            if not response.data:
                return

            request_id = int(
                response.data[0]["id"]
            )

        item_payload = {
            "booking_request_id": request_id,
            "service_id": metadata.get("service_id"),
            "service_name": metadata.get("service_name"),
            "category": metadata.get("category"),
            "duration_minutes": duration_minutes,
            "price_pence": price_pence,
            "item_order": 1,
            "updated_at": now_iso,
        }

        existing_item_response = (
            supabase.table("booking_items")
            .select("id")
            .eq("booking_request_id", request_id)
            .eq("item_order", 1)
            .limit(1)
            .execute()
        )

        if existing_item_response.data:
            (
                supabase.table("booking_items")
                .update(item_payload)
                .eq(
                    "id",
                    existing_item_response.data[0]["id"],
                )
                .execute()
            )
        else:
            (
                supabase.table("booking_items")
                .insert(item_payload)
                .execute()
            )

        try:
            (
                supabase.table("conversations")
                .update({
                    "active_booking_request_id": request_id,
                })
                .eq("session_id", session_id)
                .execute()
            )
        except Exception as error:
            print(
                "Could not link simple request projection: "
                f"{error}"
            )

    except Exception as error:
        print(
            "Could not sync simple request projection: "
            f"{error}"
        )


@app.post("/chat")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
):
    """
    State-first two-pass receptionist flow.

    Supabase is the source of truth. The extractor proposes a hidden patch.
    Python validates, saves, checks the calendar, and calculates one next
    action. The responder then writes a natural reply using verified state.
    """
    request_source = normalise_source(
        request.source
    )

    save_message(
        session_id=request.session_id,
        role="user",
        content=request.message,
        source=request_source,
    )

    current_state = get_existing_conversation(
        request.session_id
    )
    services_context = build_authoritative_services_context()
    business_context = load_business_documents_context()

    extractor_result = extract_hidden_state_patch(
        current_state=current_state,
        latest_message=request.message,
        history=request.history,
        services_context=services_context,
    )

    merged_state = apply_validated_state_patch(
        existing_state=current_state,
        extractor_result=extractor_result,
        latest_message=request.message,
        history=request.history,
    )
    merged_state = add_manual_multi_treatment_note(
        merged_state,
        extractor_result,
    )

    booking_flow_active = has_booking_intent_state(
        current_state,
        extractor_result,
        request.message,
    )

    merged_state = apply_canonical_controller_state(
        merged_state,
        booking_flow_active,
        {"status": "not_checked"},
    )

    calendar_result = verified_calendar_result_for_state(
        merged_state,
        booking_flow_active,
        request.message,
    )

    # Clear an unusable slot before calculating the next question. The
    # suggested alternatives remain available in calendar_result.
    if calendar_result.get("status") in {
        "outside_hours",
        "closed_day",
        "past",
        "busy",
    }:
        merged_state["preferred_time"] = None

        if calendar_result.get("clear_date"):
            merged_state["preferred_date"] = None

    merged_state = apply_canonical_controller_state(
        merged_state,
        booking_flow_active,
        calendar_result,
    )

    save_conversation_overview(
        session_id=request.session_id,
        lead_data=merged_state,
        source=request_source,
    )

    sync_simple_single_request_projection(
        session_id=request.session_id,
        state=merged_state,
        booking_flow_active=booking_flow_active,
    )

    reply = compose_verified_customer_reply(
        state=merged_state,
        extractor_result=extractor_result,
        calendar_result=calendar_result,
        next_required_detail=merged_state.get(
            "next_required_detail"
        ),
        business_context=business_context,
        services_context=services_context,
        latest_message=request.message,
        history=request.history,
    )

    save_message(
        session_id=request.session_id,
        role="assistant",
        content=reply,
        source=request_source,
    )

    background_tasks.add_task(
        send_booking_notification,
        request.session_id,
    )

    return {"reply": reply}
