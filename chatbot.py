from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, PlainTextResponse
from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr
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
import logging
import os
import re
import secrets
import time
from zoneinfo import ZoneInfo

load_dotenv()

logger = logging.getLogger(__name__)


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
GOOGLE_CALENDAR_SCOPE = (
    "https://www.googleapis.com/auth/calendar.freebusy "
    "https://www.googleapis.com/auth/calendar.events"
)
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")
GOOGLE_CALENDAR_LAST_DIAGNOSTIC = "not_checked"

INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
INSTAGRAM_ACCOUNT_ID = os.getenv("INSTAGRAM_ACCOUNT_ID")
INSTAGRAM_APP_SECRET = os.getenv("INSTAGRAM_APP_SECRET")
INSTAGRAM_VERIFY_TOKEN = os.getenv("INSTAGRAM_VERIFY_TOKEN")
INSTAGRAM_GRAPH_VERSION = os.getenv("INSTAGRAM_GRAPH_VERSION", "v25.0")

SUMUP_CHECKOUT_ENDPOINT = "https://api.sumup.com/v0.1/checkouts"

# Instagram session and moderation safeguards.
# A completed or stale Instagram conversation starts a fresh lead session.
INSTAGRAM_SESSION_IDLE_HOURS = 12
INSTAGRAM_MAX_MESSAGES_PER_10_MINUTES = 18
INSTAGRAM_MAX_MESSAGES_PER_DAY = 120
INSTAGRAM_RATE_LIMIT_NOTICE_COOLDOWN_MINUTES = 60
INACTIVE_BOOKING_REQUEST_STATUSES = {
    "confirmed",
    "expired",
    "completed",
    "cancelled_by_customer",
    "abandoned",
}

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


def required_booking_fields(lead_data: dict) -> list[str]:
    if (
        normalise_source(lead_data.get("source")) == "instagram"
        or lead_data.get("phone_declined") is True
    ):
        return [
            field
            for field in REQUIRED_BOOKING_FIELDS
            if field != "phone"
        ]

    return REQUIRED_BOOKING_FIELDS


def booking_request_is_complete(lead_data: dict) -> bool:
    """
    A booking request is complete only when every required field is present.
    Never trust a status label on its own.
    """
    return (
        not lead_data.get("verified_alternatives")
        and all(
            lead_data.get(field) not in [None, ""]
            for field in required_booking_fields(lead_data)
        )
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    source: str = "website"


class ConfirmAppointmentRequest(BaseModel):
    booking_request_id: int


class DepositRequest(BaseModel):
    amount_pence: StrictInt


class AdminServicePayload(BaseModel):
    category_key: StrictStr | None = None
    service_name: StrictStr | None = None
    variant_name: StrictStr | None = None
    description: StrictStr | None = None
    duration_minutes: StrictInt | None = None
    price_pence: StrictInt | None = None
    is_active: StrictBool | None = None
    aliases: list[StrictStr] | None = None
    requires_duration_choice: StrictBool | None = None
    sort_order: StrictInt | None = None
    duration_prices: dict[str, StrictInt] | None = None


class AdminAutoReplyPayload(BaseModel):
    auto_reply_enabled: StrictBool


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


def active_booking_request_status(conversation: dict) -> str | None:
    request_id = conversation.get("active_booking_request_id")

    try:
        query = supabase.table("booking_requests").select("request_status")

        if request_id not in [None, ""]:
            query = query.eq("id", request_id)
        else:
            session_id = conversation.get("session_id")

            if session_id in [None, ""]:
                return None

            query = (
                query.eq("session_id", session_id)
                .order("request_number", desc=True)
            )

        response = query.limit(1).execute()

        if response.data:
            return str(response.data[0].get("request_status") or "")

    except Exception as error:
        print(f"Could not load active booking request status: {error}")

    return None


def prepare_conversation_for_resume(conversation: dict) -> dict:
    """Never carry inactive appointment details into a fresh request flow."""
    status = active_booking_request_status(conversation)

    if status not in INACTIVE_BOOKING_REQUEST_STATUSES:
        return conversation

    return {
        "session_id": conversation.get("session_id"),
        "source": conversation.get("source"),
        "owner_status": conversation.get("owner_status"),
        "active_booking_request_id": None,
        "name": None,
        "phone": None,
        "treatment": None,
        "duration": None,
        "preferred_date": None,
        "preferred_time": None,
        "notes": None,
        "summary": None,
        "verified_alternatives": None,
        "notification_sent_at": None,
        "status": "new_chat",
    }


def customer_explicitly_cancels_request(message: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(message or "").strip().lower())
    return bool(re.search(
        r"\b(?:cancel|cancelled|canceled|do not book|don't book)\b",
        cleaned,
    ))


def cancel_active_booking_request(conversation: dict) -> bool:
    request_id = conversation.get("active_booking_request_id")

    if request_id in [None, ""]:
        return False

    now_iso = datetime.now(timezone.utc).isoformat()
    response = (
        supabase.table("booking_requests")
        .update({
            "request_status": "cancelled_by_customer",
            "confirmation_lock_token": None,
            "confirmation_started_at": None,
            "updated_at": now_iso,
        })
        .eq("id", request_id)
        .in_("request_status", ["draft", "ready_for_review", "awaiting_owner_confirmation"])
        .execute()
    )

    if not response.data:
        return False

    supabase.table("conversations").update({
        "active_booking_request_id": None,
        "treatment": None,
        "duration": None,
        "preferred_date": None,
        "preferred_time": None,
        "verified_alternatives": None,
        "notification_sent_at": None,
        "status": "new_chat",
        "updated_at": now_iso,
    }).eq(
        "session_id",
        conversation.get("session_id"),
    ).execute()
    return True


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
        r"\b(?:at\s*)?([01]?\d|2[0-3])([:.])([0-5]\d)\b",
        lower_message,
    )

    if twenty_four_hour_match:
        hour = int(twenty_four_hour_match.group(1))
        separator = twenty_four_hour_match.group(2)
        minute = twenty_four_hour_match.group(3)

        # Colon-form inputs are treated as explicit 24-hour times. Keep
        # "07:00" as 07:00; dotted casual inputs like "1.30" stay afternoon.
        if separator == "." and 1 <= hour <= 7:
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


SCHEDULE_WEEKDAYS = {
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

SCHEDULE_MONTH_NAMES = (
    "january|february|march|april|may|june|july|august|"
    "september|october|november|december"
)


def weekday_number_from_text(message: str) -> int | None:
    cleaned = str(message or "").lower()

    for name, number in SCHEDULE_WEEKDAYS.items():
        if re.search(rf"\b{name}\b", cleaned):
            return number

    return None


def has_explicit_calendar_date_signal(message: str) -> bool:
    cleaned = str(message or "").lower()
    return bool(
        re.search(r"\b20\d{2}-\d{2}-\d{2}\b", cleaned)
        or re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]20\d{2})?\b", cleaned)
        or re.search(
            rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?(?:\s+of)?\s+"
            rf"(?:{SCHEDULE_MONTH_NAMES})|(?:{SCHEDULE_MONTH_NAMES})\s+"
            r"\d{1,2}(?:st|nd|rd|th)?)\b",
            cleaned,
        )
    )


def parse_validated_date_from_text(
    message: str,
    reference_date: str | None = None,
) -> str | None:
    parsed = parse_relative_date_from_text(
        message,
        reference_date=reference_date,
    )

    if not parsed:
        return None

    named_weekday = weekday_number_from_text(message)

    if named_weekday is None or not has_explicit_calendar_date_signal(message):
        return parsed

    try:
        parsed_date = datetime.fromisoformat(parsed).date()
    except ValueError:
        return None

    return parsed if parsed_date.weekday() == named_weekday else None


def bare_hour_from_schedule_reply(message: str) -> int | None:
    cleaned = str(message or "").strip().lower()
    cleaned = re.sub(
        r"\b(?:next\s+week\s+|next\s+)?(?:"
        + "|".join(SCHEDULE_WEEKDAYS)
        + r")\b",
        "",
        cleaned,
    )
    cleaned = re.sub(r"[\s?!.]+", " ", cleaned).strip()
    match = re.fullmatch(r"(?:at\s+)?(1[0-2]|0?[1-9])", cleaned)
    return int(match.group(1)) if match else None


def format_bare_hour_for_business(hour: int) -> str:
    if 1 <= hour <= 7:
        hour += 12

    return f"{hour:02d}:00"


def format_ordinal_day(day: int) -> str:
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    return f"{day}{suffix}"


def resolve_schedule_input(
    latest_message: str,
    existing_state: dict,
) -> dict:
    """Resolve date/time input using the detail Python most recently asked."""
    pending = existing_state.get("next_required_detail")
    date_source = str(latest_message or "")
    time_source = str(latest_message or "")
    bare_hour = bare_hour_from_schedule_reply(latest_message)
    named_weekday = weekday_number_from_text(latest_message)
    explicit_time_signal = bool(re.search(
        r"\b(?:at|around)\b|\b(?:am|pm)\b|\d[:.]\d",
        str(latest_message or "").lower(),
    ))
    parsed_time = parse_time_from_text(time_source)
    if parsed_time:
        print(
            "requested_time_parsed "
            f"source_message={latest_message} parsed_time={parsed_time}"
        )
    parsed_date = parse_validated_date_from_text(
        date_source,
        reference_date=existing_state.get("preferred_date"),
    )
    clarification = None

    if (
        named_weekday is not None
        and has_explicit_calendar_date_signal(date_source)
        and parsed_date is None
    ):
        clarification = (
            "That weekday and calendar date do not match. "
            "Which date did you mean?"
        )
        parsed_time = None

    if pending == "preferred_time" and clarification is None:
        if parsed_time is None and bare_hour is not None:
            parsed_time = format_bare_hour_for_business(bare_hour)

        if (
            named_weekday is not None
            and bare_hour is not None
            and existing_state.get("preferred_date") not in [None, ""]
            and not explicit_time_signal
        ):
            try:
                existing_date = datetime.fromisoformat(
                    str(existing_state.get("preferred_date"))
                ).date()
            except ValueError:
                existing_date = None

            if existing_date and existing_date.weekday() == named_weekday:
                parsed_date = None

    elif pending == "preferred_date" and clarification is None:
        if bare_hour is not None and named_weekday is not None and not explicit_time_signal:
            clarification = (
                f"Do you mean {format_customer_weekday(named_weekday)} at "
                f"{format_bare_hour_for_business(bare_hour)}, or "
                f"{format_customer_weekday(named_weekday)} the "
                f"{format_ordinal_day(bare_hour)}?"
            )
            parsed_date = None
            parsed_time = None
        elif (
            bare_hour is not None
            and named_weekday is None
            and not explicit_time_signal
        ):
            clarification = (
                f"Do you mean {format_bare_hour_for_business(bare_hour)}, "
                f"or the {format_ordinal_day(bare_hour)}?"
            )
            parsed_date = None
            parsed_time = None

    elif (
        clarification is None
        and bare_hour is not None
        and named_weekday is not None
        and not explicit_time_signal
    ):
        if existing_state.get("preferred_date") not in [None, ""]:
            parsed_date = None
            parsed_time = format_bare_hour_for_business(bare_hour)
        else:
            clarification = (
                f"Do you mean {format_customer_weekday(named_weekday)} at "
                f"{format_bare_hour_for_business(bare_hour)}, or "
                f"{format_customer_weekday(named_weekday)} the "
                f"{format_ordinal_day(bare_hour)}?"
            )
            parsed_date = None
            parsed_time = None

    return {
        "preferred_date": parsed_date,
        "preferred_time": parsed_time,
        "clarification": clarification,
        "schedule_input_detected": bool(
            parsed_date
            or parsed_time
            or clarification
            or (
                pending in {"preferred_date", "preferred_time"}
                and (
                    named_weekday is not None
                    or has_explicit_calendar_date_signal(latest_message)
                )
            )
        ),
    }


def format_customer_weekday(weekday_number: int) -> str:
    return [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ][weekday_number]


def parse_phone_from_text(message: str) -> str | None:
    digit_groups = re.findall(r"\+?\d[\d\s()-]{6,}\d", message)

    for candidate in digit_groups:
        digits = re.sub(r"\D", "", candidate)

        if 9 <= len(digits) <= 15:
            return candidate.strip()

    return None


def parse_name_from_contact_message(
    message: str,
    phone: str | None,
) -> str | None:
    """Extract a name from simple contact replies like "John Doe, 123456789"."""
    if not phone:
        return None

    cleaned = str(message or "")
    cleaned = cleaned.replace(str(phone), " ")
    cleaned = re.sub(r"\+?\d[\d\s()-]{6,}\d", " ", cleaned)
    cleaned = re.sub(
        r"\b(?:my name is|name is|i am|i'm|im|phone|number|mobile|tel)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[,;:|/]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned if is_valid_customer_name(cleaned) else None


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


def customer_has_explicit_booking_intent(message: str) -> bool:
    cleaned = str(message or "").lower()
    return any(
        re.search(pattern, cleaned)
        for pattern in BOOKING_INTENT_PATTERNS
    )


def message_has_service_with_valid_duration(message: str) -> bool:
    metadata = resolve_service_metadata_from_message(message)

    if not metadata:
        identity = resolve_latest_structured_service(message)

        if identity and identity.get("booking_mode") != "choose_variant":
            metadata = (
                authoritative_service_metadata(identity.get("service_name"))
                or identity
            )

    if (
        not metadata
        or not metadata.get("service_name")
        or metadata.get("booking_mode") == "choose_variant"
    ):
        return False

    resolution = duration_reply_resolution(
        message,
        treatment=metadata.get("service_name"),
    )
    return resolution.get("minutes") is not None


def customer_asks_price_question(message: str) -> bool:
    cleaned = normalise_service_match_text(message)
    return any(
        phrase in cleaned
        for phrase in [
            "how much",
            "price",
            "prices",
            "cost",
            "costs",
            "what is the price",
            "what are the prices",
        ]
    )


def customer_asks_children_pets_policy(message: str) -> bool:
    cleaned = normalise_service_match_text(message)
    child_terms = {"child", "children", "kid", "kids", "baby", "babies"}
    pet_terms = {"dog", "dogs", "pet", "pets", "cat", "cats"}
    tokens = set(cleaned.split())

    return bool(tokens.intersection(child_terms | pet_terms)) and any(
        phrase in cleaned
        for phrase in [
            "bring",
            "come with",
            "take my",
            "allowed",
            "can i",
            "could i",
            "is it ok",
            "is it okay",
        ]
    )


def build_children_pets_policy_reply(message: str) -> str:
    cleaned = normalise_service_match_text(message)
    tokens = set(cleaned.split())
    asks_child = bool(
        tokens.intersection({"child", "children", "kid", "kids", "baby", "babies"})
    )
    asks_pet = bool(
        tokens.intersection({"dog", "dogs", "pet", "pets", "cat", "cats"})
    )
    closing = (
        "If you'd like, I can still help with treatment information or "
        "availability."
    )

    if asks_child and asks_pet:
        return (
            "Children are okay to bring if needed, but dogs and other pets "
            f"are not allowed. {closing}"
        )

    if asks_pet:
        return (
            "Sorry, dogs and other pets are not allowed. "
            f"{closing}"
        )

    if asks_child:
        return (
            "Yes, children are okay to bring if needed. "
            f"{closing}"
        )

    return ""


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
        ambiguous_weekday_number = (
            weekday_number_from_text(message) is not None
            and bare_hour_from_schedule_reply(message) is not None
            and not re.search(
                r"\b(?:at|around)\b|\b(?:am|pm)\b|\d[:.]\d",
                str(message or "").lower(),
            )
        )

        if ambiguous_weekday_number:
            continue

        if recovered_date is None:
            recovered_date = parse_validated_date_from_text(
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


def recover_duration_from_customer_history(
    lead_data: dict,
    conversation_history: str,
    latest_message: str,
) -> dict:
    """
    Recover a duration supplied before the exact treatment was resolved.

    The newest explicit customer duration wins, but it is applied only when
    valid for the currently resolved treatment. Assistant messages are never
    treated as customer facts.
    """
    result = dict(lead_data)

    if (
        result.get("duration") not in [None, ""]
        and duration_is_valid_for_treatment(
            result.get("treatment"),
            result.get("duration"),
        )
    ):
        return result

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

    for message in reversed(customer_messages):
        minutes = parse_duration_minutes(message)

        if minutes is None:
            continue

        candidate = format_minutes_as_duration(minutes)

        if duration_is_valid_for_treatment(
            result.get("treatment"),
            candidate,
        ):
            result["duration"] = candidate

        break

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
    return duration_reply_resolution(
        latest_message,
        previous_assistant_message,
        treatment,
    ).get("minutes")


def duration_reply_resolution(
    latest_message: str,
    previous_assistant_message: str = "",
    treatment: str | None = None,
) -> dict:
    """Resolve the newest duration reply without guessing between choices."""
    lower_previous = str(previous_assistant_message or "").lower()
    allowed = allowed_durations_for_treatment(treatment)
    duration_was_requested = (
        "how long" in lower_previous
        or "duration" in lower_previous
    )
    bare_options = allowed if allowed and (
        duration_was_requested or len(allowed) > 1
    ) else None
    candidates = duration_candidates_from_text(
        latest_message,
        allowed_bare_minutes=bare_options,
    )
    valid_candidates = (
        candidates & allowed
        if allowed
        else candidates
    )

    return {
        "minutes": (
            next(iter(valid_candidates))
            if len(valid_candidates) == 1 and len(candidates) == 1
            else None
        ),
        "ambiguous": len(candidates) > 1,
        "candidates": sorted(candidates),
        "explicit_invalid": len(candidates) == 1 and not valid_candidates,
    }


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

        _MASSAGE_SERVICE_CACHE.update({"loaded_at": now, "rows": [], "from_supabase": True})
        return [], True

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

    if service:
        return {
            int(value)
            for value in service.get("allowed_durations_minutes") or []
        }

    metadata = authoritative_service_metadata(treatment)

    if not metadata:
        metadata = structured_service_identity(treatment)

    if not metadata:
        return None

    allowed = {
        int(value)
        for value in metadata.get("allowed_durations_minutes") or []
    }

    fixed_minutes = metadata.get("fixed_duration_minutes")

    if fixed_minutes not in [None, ""]:
        allowed.add(int(fixed_minutes))

    return allowed or None


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


def service_duration_price_options(service: dict | None) -> str:
    """Format one service's duration/price choices from structured metadata."""
    if not service:
        return ""

    durations = sorted({
        int(value)
        for value in service.get("allowed_durations_minutes") or []
        if str(value).isdigit() and int(value) > 0
    })
    prices = service.get("price_by_duration") or {}
    parts = []

    for duration in durations:
        duration_text = f"{duration} minutes"
        price = format_price_pence(
            prices.get(str(duration))
        )
        parts.append(
            f"{duration_text} for {price}" if price else duration_text
        )

    if parts:
        return format_natural_list(parts)

    fixed_duration = format_minutes_as_duration(
        service.get("fixed_duration_minutes")
    )
    fixed_price = format_price_pence(
        service.get("price_pence")
    )

    return " for ".join(
        part for part in [fixed_duration, fixed_price] if part
    )


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

        _VITAMIN_SHOT_SERVICE_CACHE.update({"loaded_at": now, "rows": [], "from_supabase": True})
        return [], True

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

        _ULTRASOUND_SERVICE_CACHE.update({"loaded_at": now, "rows": [], "from_supabase": True})
        return [], True

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

        _BODY_TREATMENT_SERVICE_CACHE.update({"loaded_at": now, "rows": [], "from_supabase": True})
        return [], True

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

        _MICRONEEDLING_SERVICE_CACHE.update({"loaded_at": now, "rows": [], "from_supabase": True})
        return [], True

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
        "service_name": "Hydraface / Hydrafacial",
        "booking_mode": "choose_duration",
        "fixed_duration_minutes": None,
        "allowed_durations_minutes": [60, 90],
        "price_pence": None,
        "price_by_duration": {
            "60": 6000,
            "90": 8000,
        },
        "aliases": [
            "hydraface",
            "hydraface facial",
            "hydraface facial treatment",
            "hydrafacial",
            "hydra facial",
            "hydra facial treatment",
        ],
        "description": (
            "Hydraface / Hydrafacial facial treatment with 60-minute "
            "and 90-minute options."
        ),
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
                "allowed_durations_minutes, price_pence, "
                "price_by_duration, booking_mode, description, "
                "is_active, display_order"
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

            allowed_durations = sorted({
                int(value)
                for value in row.get("allowed_durations_minutes") or []
                if str(value).isdigit() and int(value) > 0
            })

            rows.append({
                "service_name": service_name,
                "aliases": list(row.get("aliases") or []),
                "fixed_duration_minutes": fixed_duration,
                "allowed_durations_minutes": allowed_durations,
                "price_pence": row.get("price_pence"),
                "price_by_duration": dict(
                    row.get("price_by_duration") or {}
                ),
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

        _FACIAL_SERVICE_CACHE.update({"loaded_at": now, "rows": [], "from_supabase": True})
        return [], True

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
    specific facial. Fixed-duration services receive their duration
    automatically; selectable-duration services keep duration open.
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
        options = service_duration_price_options(service)
        details = []

        if options:
            details.append(options)

        if service.get("description"):
            details.append(service["description"])

        lines.append(
            f"- {service.get('service_name')} "
            f"[{service.get('booking_mode')}]: "
            + "; ".join(details)
        )

    return "\n".join(lines), from_supabase


def build_facial_variant_prompt() -> str:
    services, _ = load_facial_services()
    options = []

    for service in services:
        service_name = str(service.get("service_name") or "").strip()

        if not service_name:
            continue

        details = service_duration_price_options(service)
        options.append(
            f"{service_name} ({details})" if details else service_name
        )

    if not options:
        return "Which facial would you like?"

    return (
        "We offer "
        + format_natural_list(options)
        + ". Which facial would you like?"
    )


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

        _DERMAL_FILLER_SERVICE_CACHE.update({"loaded_at": now, "rows": [], "from_supabase": True})
        return [], True

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
        or (
            dermal_filler_area_from_text(treatment) is not None
            and find_dermal_filler_service(treatment) is None
        )
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
                "price_pence": service.get("price_pence"),
                "price_by_duration": service.get("price_by_duration"),
                "source": "structured_catalogue",
            }

    cleaned = normalise_treatment_name(treatment)

    if cleaned in GENERIC_STRUCTURED_CATEGORIES:
        return {
            "category": GENERIC_STRUCTURED_CATEGORIES[cleaned],
            "service_name": cleaned.title(),
            "booking_mode": "choose_variant",
            "fixed_duration_minutes": None,
            "allowed_durations_minutes": None,
            "price_pence": None,
            "price_by_duration": None,
            "source": "structured_catalogue",
        }

    if is_partial_dermal_filler_treatment(treatment):
        return {
            "category": "dermal_fillers",
            "service_name": str(treatment),
            "booking_mode": "choose_variant",
            "fixed_duration_minutes": None,
            "allowed_durations_minutes": None,
            "price_pence": None,
            "price_by_duration": None,
            "source": "structured_catalogue",
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
            "price_pence": None,
            "price_by_duration": None,
            "source": "structured_catalogue",
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
            ["dermal filler", "dermal fillers", "filler", "fillers"],
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
                "price_pence": service.get("price_pence"),
                "price_by_duration": service.get("price_by_duration"),
                "source": "structured_catalogue",
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
                "price_pence": service.get("price_pence"),
                "price_by_duration": service.get("price_by_duration"),
                "source": "structured_catalogue",
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
    configured = load_service_metadata_from_table(treatment)

    if configured:
        return configured

    if inactive_service_matches(treatment):
        return None

    return fixed_treatment_metadata(treatment)


def active_service_metadata_from_state(
    state: dict,
) -> dict | None:
    """
    Resolve metadata for the locked active service.

    Booking projection must use the current active service identity, not
    historic summaries, model guesses, or catalogue text from older turns.
    """
    active_name = state.get("active_service_name") or state.get("treatment")
    metadata = authoritative_service_metadata(active_name)

    if not metadata and active_name:
        metadata = structured_service_identity(active_name)

    if (
        metadata
        and state.get("active_service_name")
        and normalise_treatment_name(metadata.get("service_name"))
        != normalise_treatment_name(state.get("active_service_name"))
    ):
        print(
            "booking_service_lock_conflict "
            f"locked={state.get('active_service_name')} "
            f"resolved={metadata.get('service_name')}"
        )
        return None

    return metadata


def clear_service_specific_state(state: dict) -> dict:
    """Clear fields that must be rebuilt when the active service changes."""
    result = dict(state)

    for field in [
        "treatment",
        "duration",
        "active_service_id",
        "active_service_name",
        "active_service_source",
        "active_category",
        "verified_alternatives",
    ]:
        result[field] = None

    result["slot_status"] = "not_checked"
    return result


def lock_active_service_from_metadata(
    state: dict,
    metadata: dict | None,
    source: str,
    preserve_treatment: bool = False,
) -> dict:
    result = dict(state)

    if not metadata or not metadata.get("service_name"):
        return result

    service_name = str(metadata.get("service_name") or "").strip()

    if not service_name:
        return result

    if not preserve_treatment:
        result["treatment"] = service_name
    result["active_service_name"] = service_name
    result["active_service_id"] = metadata.get("service_id")
    result["active_service_source"] = source

    if metadata.get("category"):
        result["active_category"] = metadata.get("category")

    return result


def lock_active_service_from_treatment(
    state: dict,
    treatment: str | None,
    source: str,
    preserve_treatment: bool = False,
) -> dict:
    if treatment in [None, ""]:
        return dict(state)

    metadata = authoritative_service_metadata(treatment)

    if not metadata:
        metadata = structured_service_identity(treatment)

    return lock_active_service_from_metadata(
        state,
        metadata,
        source,
        preserve_treatment=preserve_treatment,
    )


def latest_explicit_service_metadata(
    latest_message: str,
    existing_treatment: str | None = None,
) -> dict | None:
    identity = resolve_latest_structured_service(
        latest_message,
        existing_treatment=existing_treatment,
    )

    if (
        not identity
        or not identity.get("service_name")
        or identity.get("booking_mode") == "choose_variant"
    ):
        return None

    return (
        authoritative_service_metadata(identity.get("service_name"))
        or identity
    )


def inactive_service_matches(value: str | None) -> bool:
    """Prevent legacy fallbacks from resurrecting owner-deactivated services."""
    cleaned = normalise_service_match_text(value)

    if not cleaned:
        return False

    try:
        response = (
            supabase.table("services")
            .select("service_name, aliases")
            .eq("business_slug", BUSINESS_SLUG)
            .eq("is_active", False)
            .execute()
        )

        return any(
            service_metadata_match_score(
                value,
                row.get("service_name"),
                list(row.get("aliases") or []),
            )
            for row in response.data or []
        )
    except Exception:
        return False


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
        "a", "an", "and", "area", "book", "booking", "can", "could",
        "for", "i", "id", "im", "in", "injection", "injections",
        "interested", "looking", "my", "of", "please", "session",
        "the", "to", "treatment", "want", "would", "duration",
        "durations", "hour", "hours", "hr", "hrs", "min", "mins",
        "minute", "minutes",
    }

    return {
        token
        for token in normalise_service_match_text(value).split()
        if (
            token
            and token not in ignored
            and not re.fullmatch(r"\d+(?:\.\d+)?", token)
        )
    }


def bounded_levenshtein_distance(
    left: str,
    right: str,
    max_distance: int,
) -> int:
    """Return edit distance, stopping once it cannot be a safe match."""
    if left == right:
        return 0

    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1

    previous = list(range(len(right) + 1))

    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        row_minimum = current[0]

        for right_index, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + cost,
            )
            current.append(value)
            row_minimum = min(row_minimum, value)

        if row_minimum > max_distance:
            return max_distance + 1

        previous = current

    return previous[-1]


def fuzzy_metadata_token_match_score(
    value_tokens: set[str],
    candidate_tokens: set[str],
) -> int:
    """
    Score only highly confident typo matches on distinctive service words.

    Generic category words are ignored so "facial" or "massage" cannot pick a
    specific treatment by fuzzy matching alone.
    """
    ignored = {
        "facial",
        "facials",
        "filler",
        "fillers",
        "injection",
        "injections",
        "massage",
        "service",
        "shot",
        "shots",
        "treatment",
        "treatments",
    }
    best_score = 0

    for candidate in candidate_tokens:
        if len(candidate) < 5 or candidate in ignored:
            continue

        for token in value_tokens:
            if len(token) < 5 or token in ignored:
                continue

            max_length = max(len(candidate), len(token))
            max_distance = 1 if max_length <= 7 else 2
            distance = bounded_levenshtein_distance(
                token,
                candidate,
                max_distance,
            )

            if distance == 0 or distance > max_distance:
                continue

            similarity = 1 - (distance / max_length)

            if similarity < 0.78:
                continue

            best_score = max(
                best_score,
                6000 + len(candidate) * 20 - distance * 100,
            )

    return best_score


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

        fuzzy_score = fuzzy_metadata_token_match_score(
            value_tokens,
            candidate_tokens,
        )

        if fuzzy_score:
            best_score = max(best_score, fuzzy_score)

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

    if not matches and inactive_service_matches(message):
        return None

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
    metadata = (
        active_service_metadata_from_state(result)
        if result.get("active_service_name")
        else authoritative_service_metadata(result.get("treatment"))
    )

    if not metadata:
        return result

    service_name = metadata.get(
        "service_name"
    )

    if service_name:
        result["treatment"] = service_name
        result["active_service_name"] = service_name
        result["active_service_id"] = metadata.get("service_id")
        result["active_service_source"] = (
            result.get("active_service_source")
            or metadata.get("source")
            or "service_metadata"
        )

    fixed_minutes = metadata.get(
        "fixed_duration_minutes"
    )

    if fixed_minutes not in [None, ""] and not result.get("_fixed_duration_correction"):
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
        "hydra face": "hydraface",
    }

    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)

    # Preserve decimal service variants such as 0.5 ml, but discard sentence
    # punctuation so "relaxing massage." still grounds the specific service.
    cleaned = re.sub(r"\.(?!\d)", " ", cleaned)
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
    "radio frequency", "stretch marks", and "hydrafacial" remain useful.
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

    return 9 <= len(digits) <= 15


def working_hours_for_date(
    requested_date,
) -> tuple[datetime, datetime] | None:
    hours = get_business_hours_for_date(requested_date)

    if not hours["is_open"]:
        return None

    open_value = hours["open_time"]
    close_value = hours["close_time"]

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
    business_hours = get_business_hours_for_date(start_time.date())

    if not business_hours["loaded"]:
        return {
            "status": "hours_unknown",
            "message": (
                "Working hours could not be checked. Do not guess availability. "
                "Say the therapist will check and confirm."
            ),
        }

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

    if status == "alternative_ambiguous":
        return (
            f"I found more than one verified option matching that time: "
            f"{options}.\n\nWhich one would you prefer?"
        )

    if status == "alternative_no_match":
        return (
            f"That time is not one of the current verified options. "
            f"The available options are {options}.\n\n"
            "Which one would you prefer?"
        )

    if status == "pending_alternatives":
        return (
            f"The current verified options are {options}.\n\n"
            "Which one would you prefer?"
        )

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
        return build_facial_variant_prompt()

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
) -> bool:
    optional_projection_fields = {
        "active_service_id",
        "active_service_name",
        "active_service_source",
        "phone_declined",
    }

    try:
        persisted_lead_data = {
            key: value
            for key, value in lead_data.items()
            if not str(key).startswith("_")
        }
        if persisted_lead_data.get("phone_declined") is None:
            persisted_lead_data["phone_declined"] = False
        payload = {
            "session_id": session_id,
            "source": normalise_source(source),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **persisted_lead_data,
        }

        supabase.table("conversations").upsert(
            payload,
            on_conflict="session_id"
        ).execute()
        return True

    except Exception as error:
        try:
            fallback_payload = {
                key: value
                for key, value in payload.items()
                if key not in optional_projection_fields
            }
            supabase.table("conversations").upsert(
                fallback_payload,
                on_conflict="session_id"
            ).execute()
            print(
                "Canonical conversation state save retried without "
                f"optional active-service fields: {type(error).__name__}"
            )
            return True
        except Exception as retry_error:
            print(
                "Canonical conversation state fallback save failed: "
                f"{type(retry_error).__name__}"
            )

        print(
            "Canonical conversation state save failed: "
            f"{type(error).__name__}"
        )
        return False


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


def structured_verified_alternatives(
    suggestions: list[str],
    duration_minutes: int | None,
) -> list[dict]:
    if not duration_minutes or duration_minutes <= 0:
        return []

    alternatives = []

    for label in suggestions:
        cleaned_label = str(label or "").strip()
        relative_match = re.fullmatch(
            r"(today|tomorrow) at ([0-2]\d:[0-5]\d)",
            cleaned_label,
            flags=re.IGNORECASE,
        )
        dated_match = re.fullmatch(
            r"[A-Za-z]+ (\d{1,2}) ([A-Za-z]+) at ([0-2]\d:[0-5]\d)",
            cleaned_label,
        )

        if not relative_match and not dated_match:
            continue

        try:
            if relative_match:
                day_offset = (
                    1
                    if relative_match.group(1).lower() == "tomorrow"
                    else 0
                )
                parsed_date = (
                    datetime.now(BUSINESS_TIMEZONE).date()
                    + timedelta(days=day_offset)
                )
                parsed = datetime.fromisoformat(
                    f"{parsed_date.isoformat()}T{relative_match.group(2)}"
                )
            else:
                parsed = datetime.strptime(
                    f"{dated_match.group(1)} {dated_match.group(2)} "
                    f"{datetime.now(BUSINESS_TIMEZONE).year} "
                    f"{dated_match.group(3)}",
                    "%d %B %Y %H:%M",
                )

            alternatives.append({
                "date": parsed.date().isoformat(),
                "time": parsed.strftime("%H:%M"),
                "label": format_customer_slot(
                    parsed.date().isoformat(),
                    parsed.strftime("%H:%M"),
                ),
                "duration_minutes": duration_minutes,
            })
        except ValueError:
            continue

    return alternatives


def active_verified_alternatives(state: dict) -> list[dict]:
    alternatives = state.get("verified_alternatives") or []

    if not isinstance(alternatives, list):
        return []

    return [
        option
        for option in alternatives
        if (
            isinstance(option, dict)
            and option.get("date")
            and option.get("time")
            and option.get("label")
            and isinstance(option.get("duration_minutes"), int)
            and option.get("duration_minutes") > 0
        )
    ]


def message_explicitly_replaces_requested_slot(message: str) -> bool:
    cleaned = str(message or "").lower()
    return any(
        phrase in cleaned
        for phrase in [
            "actually",
            "instead",
            "change it",
            "change the",
            "different day",
            "different time",
            "new day",
            "new time",
            "i would like",
            "i'd like",
            "can i do",
            "could i do",
            "how about",
            "what about",
        ]
    )


def resolve_verified_alternative_reply(
    state: dict,
    latest_message: str,
) -> dict:
    alternatives = active_verified_alternatives(state)

    if not alternatives:
        return {"status": "not_applicable", "matches": []}

    cleaned = re.sub(r"\s+", " ", str(latest_message or "").strip().lower())

    if message_explicitly_replaces_requested_slot(cleaned):
        return {"status": "explicit_replacement", "matches": []}

    ordinal_patterns = [
        (r"\b(?:the\s+)?first(?:\s+(?:one|option|slot))?\b", 0),
        (r"\b(?:the\s+)?second(?:\s+(?:one|option|slot))?\b", 1),
        (r"\b(?:the\s+)?third(?:\s+(?:one|option|slot))?\b", 2),
        (r"\b(?:the\s+)?last(?:\s+(?:one|option|slot))?\b", -1),
    ]

    for pattern, index in ordinal_patterns:
        if re.search(pattern, cleaned):
            try:
                return {
                    "status": "selected",
                    "matches": [alternatives[index]],
                }
            except IndexError:
                return {"status": "no_match", "matches": []}

    requested_time = parse_time_from_text(cleaned)
    requested_times = [requested_time] if requested_time else []

    if not requested_times:
        bare_hour_match = re.fullmatch(
            r"(?:yes[\s,]+)?(?:the\s+)?(1[0-2]|[1-9])"
            r"(?:\s+(?:one|works|suits|please))?\??",
            cleaned,
        )

        if bare_hour_match:
            hour = int(bare_hour_match.group(1))
            requested_times = [
                f"{hour:02d}:00",
                f"{(hour + 12) % 24:02d}:00",
            ]

    if not requested_times:
        return {"status": "not_applicable", "matches": []}

    matches = [
        option
        for option in alternatives
        if option["time"] in requested_times
    ]
    date_words = re.findall(
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"january|february|march|april|may|june|july|august|september|"
        r"october|november|december|\d{4}-\d{2}-\d{2})\b",
        cleaned,
    )

    if date_words:
        matches = [
            option
            for option in matches
            if all(
                word in option["label"].lower()
                or word == option["date"]
                for word in date_words
            )
        ]

        day_match = re.search(r"\b([0-3]?\d)\s+[a-z]+\b", cleaned)

        if day_match:
            matches = [
                option
                for option in matches
                if int(option["date"][-2:]) == int(day_match.group(1))
            ]

    if len(matches) == 1:
        return {"status": "selected", "matches": matches}

    if len(matches) > 1:
        return {"status": "ambiguous", "matches": matches}

    return {"status": "no_match", "matches": []}


def apply_verified_alternative_resolution(
    state: dict,
    existing_state: dict,
    resolution: dict,
) -> dict:
    result = dict(state)
    alternatives = active_verified_alternatives(existing_state)
    status = resolution.get("status")

    if status == "selected":
        selected = resolution["matches"][0]
        result["preferred_date"] = selected["date"]
        result["preferred_time"] = selected["time"]
        result["verified_alternatives"] = []
        result["slot_status"] = "not_checked"
        print(
            "verified_slot_selected "
            f"date={selected['date']} time={selected['time']}"
        )
        return result

    if status in {"ambiguous", "no_match"}:
        result["preferred_date"] = existing_state.get("preferred_date")
        result["preferred_time"] = existing_state.get("preferred_time")
        result["verified_alternatives"] = alternatives
        return result

    treatment_changed = (
        existing_state.get("treatment")
        and result.get("treatment")
        and normalise_treatment_name(existing_state.get("treatment"))
        != normalise_treatment_name(result.get("treatment"))
    )
    duration_changed = (
        parse_duration_minutes(existing_state.get("duration"))
        != parse_duration_minutes(result.get("duration"))
    )
    schedule_changed = (
        existing_state.get("preferred_date") != result.get("preferred_date")
        or existing_state.get("preferred_time") != result.get("preferred_time")
    )

    result["verified_alternatives"] = (
        []
        if treatment_changed
        or duration_changed
        or schedule_changed
        or status == "explicit_replacement"
        else alternatives
    )
    return result


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
                "status, source, phone_declined, notification_sent_at"
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
            for field in required_booking_fields(lead)
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
        "auto_reply_enabled": True,
    }

    try:
        response = (
            supabase.table("chatbot_settings")
            .select(
                "tone, reply_length, emoji_style, personality_notes, "
                "enable_multi_booking, auto_reply_enabled"
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
        auto_reply_enabled = saved.get("auto_reply_enabled")

        if tone in ["friendly", "professional", "relaxed", "luxury"]:
            defaults["tone"] = tone

        if reply_length in ["short", "normal"]:
            defaults["reply_length"] = reply_length

        if emoji_style in ["none", "light", "friendly"]:
            defaults["emoji_style"] = emoji_style

        defaults["personality_notes"] = personality_notes[:600]
        defaults["enable_multi_booking"] = enable_multi_booking
        if isinstance(auto_reply_enabled, bool):
            defaults["auto_reply_enabled"] = auto_reply_enabled

    except Exception as error:
        print(f"Could not load chatbot settings: {error}")

    return defaults


def auto_reply_enabled() -> bool:
    """Fail open until the migration exists, preserving current behaviour."""
    return bool(load_chatbot_settings().get("auto_reply_enabled", True))


def google_oauth_is_configured() -> bool:
    return all([
        GOOGLE_CLIENT_ID,
        GOOGLE_CLIENT_SECRET,
        GOOGLE_REDIRECT_URI,
        GOOGLE_OAUTH_STATE_SECRET,
    ])


def set_google_calendar_diagnostic(code: str):
    global GOOGLE_CALENDAR_LAST_DIAGNOSTIC
    GOOGLE_CALENDAR_LAST_DIAGNOSTIC = code
    print(f"google_calendar_diagnostic={code}")


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

    except Exception:
        set_google_calendar_diagnostic("refresh_token_load_failed")

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


def refresh_token_http_error_diagnostic(error: urllib_error.HTTPError) -> str:
    oauth_error = None

    try:
        response_data = json.loads(error.read().decode("utf-8"))

        if isinstance(response_data, dict):
            oauth_error = response_data.get("error")
    except Exception:
        pass
    finally:
        error.close()

    if oauth_error == "invalid_grant":
        return "refresh_token_invalid_grant"

    if oauth_error == "invalid_client":
        return "refresh_token_invalid_client"

    if error.code == 400:
        return "refresh_token_http_400"

    if error.code == 401:
        return "refresh_token_http_401"

    return "refresh_token_unexpected_error"


def refresh_google_access_token() -> str | None:
    refresh_token = get_google_refresh_token()

    if not refresh_token:
        set_google_calendar_diagnostic("missing_stored_refresh_token")
        return None

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        set_google_calendar_diagnostic("oauth_credentials_missing")
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
            try:
                token_data = json.loads(response.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                set_google_calendar_diagnostic(
                    "refresh_token_malformed_response"
                )
                return None

        if not isinstance(token_data, dict):
            set_google_calendar_diagnostic("refresh_token_malformed_response")
            return None

        access_token = token_data.get("access_token")

        if not access_token:
            set_google_calendar_diagnostic(
                "refresh_token_missing_access_token"
            )
            return None

        set_google_calendar_diagnostic("access_token_refreshed")
        return access_token

    except urllib_error.HTTPError as error:
        set_google_calendar_diagnostic(
            refresh_token_http_error_diagnostic(error)
        )

    except TimeoutError:
        set_google_calendar_diagnostic("refresh_token_timeout")

    except urllib_error.URLError as error:
        set_google_calendar_diagnostic(
            "refresh_token_timeout"
            if isinstance(error.reason, TimeoutError)
            else "refresh_token_network_error"
        )

    except Exception:
        set_google_calendar_diagnostic("refresh_token_unexpected_error")

    return None


def load_business_hours_with_status(
    business_slug: str = BUSINESS_SLUG,
) -> tuple[dict, bool]:
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
            .eq("business_slug", business_slug)
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

        return hours, True

    except Exception as error:
        print(f"Could not load live business hours: {error}")

    return hours, False


def load_business_hours() -> dict:
    return load_business_hours_with_status()[0]


def get_business_hours_for_date(
    requested_date,
    business_slug: str = BUSINESS_SLUG,
) -> dict:
    """Return the owner-edited opening window and whether it was loadable."""
    hours, loaded = load_business_hours_with_status(business_slug)
    configured = hours.get(requested_date.weekday())

    if not loaded:
        return {
            "loaded": False,
            "is_open": False,
            "open_time": None,
            "close_time": None,
            "timezone": str(BUSINESS_TIMEZONE),
        }

    if not configured:
        return {
            "loaded": True,
            "is_open": False,
            "open_time": None,
            "close_time": None,
            "timezone": str(BUSINESS_TIMEZONE),
        }

    open_value, close_value = configured
    return {
        "loaded": True,
        "is_open": True,
        "open_time": open_value,
        "close_time": close_value,
        "timezone": str(BUSINESS_TIMEZONE),
    }


def parse_duration_minutes(value: str | None) -> int | None:
    candidates = duration_candidates_from_text(value)

    if len(candidates) == 1:
        return next(iter(candidates))

    return None


def duration_candidates_from_text(
    value: str | None,
    allowed_bare_minutes: set[int] | None = None,
) -> set[int]:
    """Return every explicit duration in text; multiple values stay ambiguous."""
    if value in [None, ""]:
        return set()

    cleaned = str(value).strip().lower()
    candidates: set[int] = set()
    occupied: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < used_end and end > used_start for used_start, used_end in occupied)

    def add(minutes: int, match: re.Match) -> None:
        if minutes > 0:
            candidates.add(minutes)
            occupied.append(match.span())

    for match in re.finditer(
        r"\b(?:either\s+)?(\d{1,3})\s*(?:or|/)\s*(\d{1,3})"
        r"\s*(?:minutes?|mins?|min|hours?|hrs?|hr|h)?\b",
        cleaned,
    ):
        candidates.update({int(match.group(1)), int(match.group(2))})
        occupied.append(match.span())

    for pattern, minutes in [
        (r"\bhour\s+and\s+a\s+half\b", 90),
        (r"\bhalf\s+(?:an?|one)\s+hour\b", 30),
    ]:
        for match in re.finditer(pattern, cleaned):
            if not overlaps(*match.span()):
                add(minutes, match)

    hour_pattern = re.compile(
        r"\b(?P<hours>\d+(?:\.\d+)?|an?|one|two)[\s-]*"
        r"(?:hours?|hrs?|hr|h)(?![a-z])"
        r"(?:\s*(?:and\s+)?(?P<minutes>\d{1,2})"
        r"\s*(?:minutes?|mins?|min|m)?\b)?"
    )
    for match in hour_pattern.finditer(cleaned):
        if overlaps(*match.span()):
            continue

        hours_value = match.group("hours")
        hours = (
            2.0
            if hours_value == "two"
            else 1.0
            if hours_value in {"a", "an", "one"}
            else float(hours_value)
        )
        add(round(hours * 60) + int(match.group("minutes") or 0), match)

    for match in re.finditer(
        r"(?<![\d:])(\d{1,3})[\s-]*(?:minutes?|mins?|min)\b",
        cleaned,
    ):
        if not overlaps(*match.span()):
            add(int(match.group(1)), match)

    for match in re.finditer(r"\bhour\b", cleaned):
        if not overlaps(*match.span()):
            add(60, match)

    if re.fullmatch(r"\d{1,3}", cleaned):
        candidates.add(int(cleaned))

    if allowed_bare_minutes:
        for match in re.finditer(r"(?<![\d:£$])(\d{1,3})(?![\d:])", cleaned):
            value_minutes = int(match.group(1))

            if value_minutes in allowed_bare_minutes and not overlaps(*match.span()):
                candidates.add(value_minutes)

    return candidates


def build_requested_slot(lead_data: dict):
    preferred_date = lead_data.get("preferred_date")
    preferred_time = lead_data.get("preferred_time")
    duration_minutes = parse_duration_minutes(lead_data.get("duration"))

    if (
        not preferred_date
        or not preferred_time
        or not duration_minutes
        or duration_minutes <= 0
    ):
        return None

    try:
        start_time = datetime.fromisoformat(
            f"{preferred_date}T{preferred_time}"
        ).replace(tzinfo=BUSINESS_TIMEZONE)
        end_time = start_time + timedelta(minutes=duration_minutes)

        return start_time, end_time, duration_minutes

    except ValueError:
        return None


def expire_pending_payment_holds() -> None:
    """Release elapsed holds without deleting their payment history."""
    now_iso = datetime.now(timezone.utc).isoformat()
    logger.info("Expired payment hold cleanup started")

    try:
        cleanup_response = (
            supabase.table("payment_requests")
            .select("id, status, calendar_hold_event_id, calendar_hold_status")
            .in_("status", ["pending", "expired"])
            .lte("hold_expires_at", now_iso)
            .eq("calendar_hold_status", "active")
            .execute()
        )
        (
            supabase.table("payment_requests")
            .update({
                "status": "expired",
                "expired_at": now_iso,
                "updated_at": now_iso,
            })
            .eq("status", "pending")
            .lte("hold_expires_at", now_iso)
            .execute()
        )
    except Exception as error:
        logger.warning(
            "Could not expire elapsed payment holds: %s",
            type(error).__name__,
        )
        return

    for payment_request in cleanup_response.data or []:
        event_id = payment_request.get("calendar_hold_event_id")

        if not event_id:
            continue

        payment_request_id = str(payment_request["id"])

        try:
            delete_google_calendar_event(str(event_id))
            (
                supabase.table("payment_requests")
                .update({
                    "calendar_hold_status": "deleted",
                    "calendar_hold_deleted_at": now_iso,
                    "calendar_hold_error": None,
                    "updated_at": now_iso,
                })
                .eq("id", payment_request_id)
                .execute()
            )
            logger.info(
                "Temporary calendar event deletion succeeded; event_id_present=true"
            )
        except Exception as error:
            if isinstance(error, urllib_error.HTTPError):
                error.close()
            logger.warning(
                "Temporary calendar event deletion failed: %s",
                type(error).__name__,
            )
            try:
                (
                    supabase.table("payment_requests")
                    .update({
                        "calendar_hold_error": (
                            "Google Calendar temporary hold could not be deleted."
                        ),
                        "updated_at": now_iso,
                    })
                    .eq("id", payment_request_id)
                    .execute()
                )
            except Exception:
                logger.exception(
                    "Could not save temporary calendar deletion failure"
                )


def active_payment_hold_busy_periods(
    start_time: datetime,
    end_time: datetime,
    exclude_booking_request_id: int | None = None,
) -> list[dict]:
    """Return active SumUp holds overlapping a requested interval."""
    expire_pending_payment_holds()
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        response = (
            supabase.table("payment_requests")
            .select(
                "booking_request_id, hold_expires_at, held_start_at, held_end_at"
            )
            .eq("business_slug", BUSINESS_SLUG)
            .eq("status", "pending")
            .gt("hold_expires_at", now_iso)
            .execute()
        )
    except Exception as error:
        logger.warning(
            "Could not load active payment holds: %s",
            type(error).__name__,
        )
        return []

    busy_periods = []

    for row in response.data or []:
        request_id = row.get("booking_request_id")

        if (
            exclude_booking_request_id is not None
            and str(request_id) == str(exclude_booking_request_id)
        ):
            continue

        hold_start = parse_utc_datetime(row.get("held_start_at"))
        hold_end = parse_utc_datetime(row.get("held_end_at"))

        if not hold_start or not hold_end:
            continue

        hold_start = hold_start.astimezone(BUSINESS_TIMEZONE)
        hold_end = hold_end.astimezone(BUSINESS_TIMEZONE)

        if hold_start < end_time and hold_end > start_time:
            busy_periods.append({
                "start": hold_start.isoformat(),
                "end": hold_end.isoformat(),
                "source": "payment_hold",
                "booking_request_id": request_id,
            })

    return busy_periods


def query_google_freebusy(
    start_time: datetime,
    end_time: datetime,
) -> list[dict] | None:
    if not GOOGLE_CALENDAR_ID:
        set_google_calendar_diagnostic("calendar_id_missing")
        return None

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

        if not isinstance(freebusy_data, dict):
            set_google_calendar_diagnostic("malformed_api_response")
            return None

        calendars = freebusy_data.get("calendars")

        if not isinstance(calendars, dict):
            set_google_calendar_diagnostic("malformed_api_response")
            return None

        calendar_data = calendars.get(GOOGLE_CALENDAR_ID)

        if not isinstance(calendar_data, dict):
            set_google_calendar_diagnostic("malformed_api_response")
            return None

        if calendar_data.get("errors"):
            set_google_calendar_diagnostic("freebusy_api_failure")
            return None

        busy = calendar_data.get("busy")

        if not isinstance(busy, list):
            set_google_calendar_diagnostic("malformed_api_response")
            return None

        set_google_calendar_diagnostic("freebusy_success")
        return busy

    except urllib_error.HTTPError:
        set_google_calendar_diagnostic("freebusy_http_error")

    except TimeoutError:
        set_google_calendar_diagnostic("network_timeout")

    except urllib_error.URLError as error:
        set_google_calendar_diagnostic(
            "network_timeout"
            if isinstance(error.reason, TimeoutError)
            else "unexpected_exception"
        )

    except json.JSONDecodeError:
        set_google_calendar_diagnostic("malformed_api_response")

    except Exception:
        set_google_calendar_diagnostic("unexpected_exception")

    return None


def query_calendar_and_hold_busy_periods(
    start_time: datetime,
    end_time: datetime,
    exclude_booking_request_id: int | None = None,
) -> list[dict] | None:
    calendar_busy = query_google_freebusy(start_time, end_time)

    if calendar_busy is None:
        return None

    return calendar_busy + active_payment_hold_busy_periods(
        start_time,
        end_time,
        exclude_booking_request_id=exclude_booking_request_id,
    )


def create_google_calendar_event(
    start_time: datetime,
    end_time: datetime,
    title: str,
    description: str,
    booking_request_id: int | None = None,
    private_properties: dict[str, str] | None = None,
) -> dict:
    """Create one busy event on the connected calendar."""
    access_token = refresh_google_access_token()

    if not access_token:
        raise RuntimeError("Google Calendar is not connected.")

    calendar_id = urllib_parse.quote(
        GOOGLE_CALENDAR_ID,
        safe="",
    )
    event_private_properties = private_properties or {}

    if booking_request_id is not None:
        event_private_properties.setdefault(
            "bookingRequestId",
            str(booking_request_id),
        )

    payload = {
        "summary": title,
        "description": description,
        "transparency": "opaque",
        "start": {
            "dateTime": start_time.isoformat(),
            "timeZone": "Europe/London",
        },
        "end": {
            "dateTime": end_time.isoformat(),
            "timeZone": "Europe/London",
        },
        "extendedProperties": {
            "private": event_private_properties,
        },
    }
    event_request = urllib_request.Request(
        f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": "receptionist-chatbot/1.0",
        },
        method="POST",
    )

    with urllib_request.urlopen(event_request, timeout=15) as response:
        event = json.loads(response.read().decode("utf-8"))

    if not isinstance(event, dict) or not event.get("id"):
        raise RuntimeError("Google Calendar did not return an event ID.")

    return event


def find_google_calendar_event_by_private_property(
    property_name: str,
    property_value: str,
) -> dict | None:
    """Find a previously created event after a partial persistence failure."""
    access_token = refresh_google_access_token()

    if not access_token:
        raise RuntimeError("Google Calendar is not connected.")

    calendar_id = urllib_parse.quote(GOOGLE_CALENDAR_ID, safe="")
    query = urllib_parse.urlencode({
        "privateExtendedProperty": f"{property_name}={property_value}",
        "singleEvents": "true",
        "maxResults": "1",
    })
    event_request = urllib_request.Request(
        f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events?{query}",
        headers={
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "receptionist-chatbot/1.0",
        },
        method="GET",
    )

    with urllib_request.urlopen(event_request, timeout=15) as response:
        result = json.loads(response.read().decode("utf-8"))

    items = result.get("items") if isinstance(result, dict) else None
    return items[0] if items else None


def find_google_calendar_event_for_request(
    booking_request_id: int,
) -> dict | None:
    return find_google_calendar_event_by_private_property(
        "bookingRequestId",
        str(booking_request_id),
    )


def delete_google_calendar_event(event_id: str) -> None:
    """Delete one event from the connected owner calendar."""
    access_token = refresh_google_access_token()

    if not access_token:
        raise RuntimeError("Google Calendar is not connected.")

    calendar_id = urllib_parse.quote(GOOGLE_CALENDAR_ID, safe="")
    encoded_event_id = urllib_parse.quote(event_id, safe="")
    event_request = urllib_request.Request(
        (
            "https://www.googleapis.com/calendar/v3/calendars/"
            f"{calendar_id}/events/{encoded_event_id}"
        ),
        headers={
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "receptionist-chatbot/1.0",
        },
        method="DELETE",
    )

    try:
        with urllib_request.urlopen(event_request, timeout=15):
            pass
    except urllib_error.HTTPError as error:
        if error.code in {404, 410}:
            error.close()
            return
        raise


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

    busy_periods = query_calendar_and_hold_busy_periods(
        search_start,
        search_end,
    )

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

    busy_periods = query_calendar_and_hold_busy_periods(
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
    exclude_booking_request_id: int | None = None,
) -> dict:
    duration_minutes = parse_duration_minutes(
        lead_data.get("duration")
    )

    if not duration_minutes or duration_minutes <= 0:
        return {
            "status": "not_checked",
            "message": "A positive treatment duration is not saved yet.",
        }

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

    busy_periods = query_calendar_and_hold_busy_periods(
        start_time,
        end_time,
        exclude_booking_request_id=exclude_booking_request_id,
    )

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
) -> dict:
    if not INSTAGRAM_ACCESS_TOKEN:
        logger.warning("Instagram message skipped because the access token is missing.")
        return {
            "success": False,
            "status": "not_configured",
            "message": "Instagram messaging is not configured.",
        }

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
            response_body = response.read().decode("utf-8", errors="replace")

        try:
            meta_response = (
                json.loads(response_body)
                if response_body.strip()
                else {}
            )
        except json.JSONDecodeError:
            meta_response = {}

        logger.info("Instagram message accepted by Meta.")
        return {
            "success": True,
            "status": "sent",
            "message": "Instagram message sent.",
            "meta_response": meta_response,
        }

    except urllib_error.HTTPError as error:
        error.close()
        logger.warning(
            "Instagram message rejected by Meta with HTTP status %s.",
            error.code,
        )
        return {
            "success": False,
            "status": "rejected",
            "message": f"Meta rejected the Instagram message (HTTP {error.code}).",
        }

    except Exception as error:
        logger.error(
            "Instagram message request failed with %s.",
            type(error).__name__,
        )
        return {
            "success": False,
            "status": "error",
            "message": "Instagram messaging could not be reached.",
        }


def fetch_instagram_user_profile(sender_id: str) -> dict:
    """Fetch owner-facing Instagram profile metadata for an Instagram-scoped ID."""
    if not INSTAGRAM_ACCESS_TOKEN:
        return {}

    endpoint = (
        f"https://graph.instagram.com/"
        f"{INSTAGRAM_GRAPH_VERSION}/{urllib_parse.quote(sender_id, safe='')}?"
        + urllib_parse.urlencode({
            "fields": "name,username",
            "access_token": INSTAGRAM_ACCESS_TOKEN,
        })
    )
    profile_request = urllib_request.Request(
        endpoint,
        headers={"User-Agent": "receptionist-chatbot/1.0"},
        method="GET",
    )

    try:
        with urllib_request.urlopen(profile_request, timeout=15) as response:
            profile = json.loads(response.read().decode("utf-8"))

        if not isinstance(profile, dict):
            return {}

        return {
            "profile_name": str(profile.get("name") or "").strip() or None,
            "profile_username": str(profile.get("username") or "").strip() or None,
            "profile_fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    except urllib_error.HTTPError as error:
        error.close()
        logger.warning(
            "Instagram profile lookup was rejected with HTTP status %s.",
            error.code,
        )
    except Exception as error:
        logger.error(
            "Instagram profile lookup failed with %s.",
            type(error).__name__,
        )

    return {}


def instagram_profile_needs_refresh(contact: dict) -> bool:
    fetched_at = contact.get("profile_fetched_at")

    if not fetched_at:
        return True

    try:
        fetched = datetime.fromisoformat(
            str(fetched_at).replace("Z", "+00:00")
        )

        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)

        return (
            datetime.now(timezone.utc) - fetched.astimezone(timezone.utc)
            >= timedelta(hours=24)
        )
    except ValueError:
        return True


def save_instagram_channel_contact(
    sender_id: str,
    session_id: str,
    customer_message_at: str,
    profile: dict | None = None,
) -> dict:
    """Store one Instagram-scoped identity and link its current conversation."""
    try:
        payload = {
            "business_slug": BUSINESS_SLUG,
            "source": "instagram",
            "external_user_id": sender_id,
            "last_customer_message_at": customer_message_at,
            "updated_at": customer_message_at,
        }
        payload.update({
            key: value
            for key, value in (profile or {}).items()
            if key in {
                "profile_name",
                "profile_username",
                "profile_fetched_at",
            }
            and value not in [None, ""]
        })
        contact_response = (
            supabase.table("channel_contacts")
            .upsert(
                payload,
                on_conflict="business_slug,source,external_user_id",
            )
            .execute()
        )
        contact = (
            contact_response.data[0]
            if contact_response.data
            else {}
        )
        contact_id = contact.get("id")

        if contact_id:
            (
                supabase.table("conversations")
                .update({"channel_contact_id": contact_id})
                .eq("session_id", session_id)
                .execute()
            )

        return contact
    except Exception:
        logger.exception(
            "Could not save Instagram channel contact for session %s",
            session_id,
        )
        return {}


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

    request_status = active_booking_request_status(latest)

    if request_status in INACTIVE_BOOKING_REQUEST_STATUSES:
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

            session_id, previous_conversation = resolve_instagram_session(
                sender_id,
                message_text,
            )
            customer_message_at = datetime.now(timezone.utc).isoformat()
            contact = save_instagram_channel_contact(
                sender_id,
                session_id,
                customer_message_at,
            )
            instagram_profile = {}

            if instagram_profile_needs_refresh(contact):
                instagram_profile = fetch_instagram_user_profile(sender_id)

                if instagram_profile:
                    save_instagram_channel_contact(
                        sender_id,
                        session_id,
                        customer_message_at,
                        instagram_profile,
                    )

            if not auto_reply_enabled():
                history = load_recent_messages(session_id)
                asyncio.run(chat(
                    ChatRequest(
                        session_id=session_id,
                        message=message_text,
                        history=history,
                        source="instagram",
                    ),
                    BackgroundTasks(),
                ))
                save_instagram_channel_contact(
                    sender_id,
                    session_id,
                    customer_message_at,
                    instagram_profile,
                )
                continue

            rate_limit_reply = instagram_rate_limit_message(sender_id)

            if rate_limit_reply is not None:
                if rate_limit_reply:
                    send_instagram_text_message(
                        recipient_id=sender_id,
                        text=rate_limit_reply,
                    )

                continue

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
                save_instagram_channel_contact(
                    sender_id,
                    session_id,
                    customer_message_at,
                    instagram_profile,
                )

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
            <p>You can close this page. The backend can now check availability and create owner-confirmed appointments.</p>
            """
        )

    except urllib_error.HTTPError as oauth_error:
        oauth_error.close()
        print(f"Google OAuth token exchange failed: HTTP {oauth_error.code}")

        return HTMLResponse(
            """
            <h2>Google Calendar connection failed.</h2>
            <p>The Google token exchange was rejected. Check the Render logs for details.</p>
            """,
            status_code=500,
        )

    except Exception:
        print("Google Calendar connection failed.")

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
        "last_diagnostic_code": GOOGLE_CALENDAR_LAST_DIAGNOSTIC,
    }


def require_authenticated_admin(request: Request):
    """Use the dashboard's existing Supabase Auth session for admin actions."""
    authorization = request.headers.get("authorization", "")

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Admin sign-in required.")

    access_token = authorization.split(" ", 1)[1].strip()

    try:
        user_response = supabase.auth.get_user(access_token)
        user = getattr(user_response, "user", None)

        if not user:
            raise HTTPException(status_code=401, detail="Admin sign-in required.")

        return user
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Admin sign-in required.")


@app.get("/admin/auto-reply")
async def get_admin_auto_reply(request: Request):
    require_authenticated_admin(request)
    return {"auto_reply_enabled": auto_reply_enabled()}


@app.patch("/admin/auto-reply")
async def set_admin_auto_reply(
    payload: AdminAutoReplyPayload,
    request: Request,
):
    require_authenticated_admin(request)
    response = (
        supabase.table("chatbot_settings")
        .update({
            "auto_reply_enabled": payload.auto_reply_enabled,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("business_slug", BUSINESS_SLUG)
        .execute()
    )

    if not response.data:
        raise HTTPException(
            status_code=500,
            detail="Auto-reply setting could not be updated.",
        )

    return {"auto_reply_enabled": payload.auto_reply_enabled}


ADMIN_SERVICE_EDITABLE_FIELDS = {
    "category_key",
    "service_name",
    "variant_name",
    "description",
    "duration_minutes",
    "price_pence",
    "is_active",
    "aliases",
    "requires_duration_choice",
    "sort_order",
    "duration_prices",
}


def clean_admin_service_payload(values: dict, require_all: bool = True) -> dict:
    """Validate owner-edited services before any database write."""
    cleaned = {
        key: value
        for key, value in values.items()
        if key in ADMIN_SERVICE_EDITABLE_FIELDS
    }

    def clean_text(key: str, maximum: int, required: bool = False):
        value = cleaned.get(key)

        if value is None and not required:
            return None

        value = str(value or "").strip()

        if required and not value:
            raise HTTPException(status_code=422, detail=f"{key} is required.")
        if len(value) > maximum:
            raise HTTPException(status_code=422, detail=f"{key} is too long.")
        if re.search(r"[<>]", value):
            raise HTTPException(status_code=422, detail=f"{key} cannot contain HTML markup.")
        return value or None

    service_name = clean_text("service_name", 80, require_all)
    category_key = clean_text("category_key", 80, require_all)
    variant_name = clean_text("variant_name", 80)
    description = clean_text("description", 500)

    if service_name is not None and len(service_name) < 2:
        raise HTTPException(status_code=422, detail="service_name must be at least 2 characters.")
    if category_key is not None and not re.fullmatch(r"[a-z0-9_]+", category_key):
        raise HTTPException(status_code=422, detail="category_key must use lowercase letters, numbers, and underscores only.")

    aliases = cleaned.get("aliases")
    if aliases is None:
        aliases = []
    if not isinstance(aliases, list):
        raise HTTPException(status_code=422, detail="aliases must be a list.")
    clean_aliases = []
    for alias in aliases:
        alias = str(alias or "").strip()
        if not alias:
            continue
        if len(alias) > 60:
            raise HTTPException(status_code=422, detail="Each alias must be 60 characters or fewer.")
        if re.search(r"[<>]", alias):
            raise HTTPException(status_code=422, detail="Aliases cannot contain HTML markup.")
        if alias not in clean_aliases:
            clean_aliases.append(alias)

    requires_choice = cleaned.get("requires_duration_choice")
    if require_all and not isinstance(requires_choice, bool):
        raise HTTPException(status_code=422, detail="requires_duration_choice must be true or false.")
    requires_choice = bool(requires_choice)

    duration = cleaned.get("duration_minutes")
    if not requires_choice and (not isinstance(duration, int) or not 5 <= duration <= 300):
        raise HTTPException(status_code=422, detail="duration_minutes must be an integer between 5 and 300.")
    if duration is not None and (not isinstance(duration, int) or not 5 <= duration <= 300):
        raise HTTPException(status_code=422, detail="duration_minutes must be an integer between 5 and 300.")

    price = cleaned.get("price_pence")
    if requires_choice:
        if price is not None and (
            not isinstance(price, int) or not 0 <= price <= 100000
        ):
            raise HTTPException(status_code=422, detail="price_pence must be an integer between 0 and 100000.")
    elif not isinstance(price, int) or not 0 <= price <= 100000:
        raise HTTPException(status_code=422, detail="price_pence must be an integer between 0 and 100000.")

    sort_order = cleaned.get("sort_order", 0)
    if not isinstance(sort_order, int) or not 0 <= sort_order <= 9999:
        raise HTTPException(status_code=422, detail="sort_order must be an integer between 0 and 9999.")

    is_active = cleaned.get("is_active", True)
    if not isinstance(is_active, bool):
        raise HTTPException(status_code=422, detail="is_active must be true or false.")

    duration_prices = cleaned.get("duration_prices") or {}
    if not isinstance(duration_prices, dict):
        raise HTTPException(status_code=422, detail="duration_prices must be an object.")
    clean_duration_prices = {}
    for minutes, amount in duration_prices.items():
        if not str(minutes).isdigit() or not 5 <= int(minutes) <= 300:
            raise HTTPException(status_code=422, detail="Duration price keys must be minutes between 5 and 300.")
        if not isinstance(amount, int) or not 0 <= amount <= 100000:
            raise HTTPException(status_code=422, detail="Duration prices must be between 0 and 100000 pence.")
        clean_duration_prices[str(int(minutes))] = amount

    if requires_choice and not clean_duration_prices:
        raise HTTPException(status_code=422, detail="Add at least one selectable duration price.")

    booking_mode = "choose_duration" if requires_choice else "fixed_duration"
    allowed_durations = sorted(int(value) for value in clean_duration_prices)

    return {
        "business_slug": BUSINESS_SLUG,
        "category_key": category_key,
        "category": category_key,
        "service_name": service_name,
        "variant_name": variant_name,
        "description": description,
        "duration_minutes": duration,
        "fixed_duration_minutes": None if requires_choice else duration,
        "price_pence": None if requires_choice else price,
        "is_active": is_active,
        "aliases": clean_aliases,
        "requires_duration_choice": requires_choice,
        "booking_mode": booking_mode,
        "sort_order": sort_order,
        "display_order": sort_order,
        "allowed_durations_minutes": allowed_durations if requires_choice else [duration],
        "price_by_duration": clean_duration_prices if requires_choice else None,
    }


def invalidate_service_caches() -> None:
    for name, value in globals().items():
        if name.endswith("_SERVICE_CACHE") and isinstance(value, dict):
            value["loaded_at"] = 0.0
            value["rows"] = None


def admin_service_response(row: dict) -> dict:
    result = dict(row)
    duration_prices = row.get("price_by_duration") or {}
    fallback_price = next(iter(duration_prices.values()), None)
    result["category_key"] = row.get("category_key") or row.get("category")
    result["duration_minutes"] = row.get("duration_minutes") or row.get("fixed_duration_minutes")
    result["requires_duration_choice"] = bool(
        row.get("requires_duration_choice")
        if row.get("requires_duration_choice") is not None
        else row.get("booking_mode") == "choose_duration"
    )
    result["sort_order"] = row.get("sort_order") or row.get("display_order") or 0
    result["duration_prices"] = duration_prices
    result["price_pence"] = row.get("price_pence") if row.get("price_pence") is not None else fallback_price
    result["display_price"] = format_price_pence(result["price_pence"])
    return result


def service_alias_identity_set(service_name, aliases) -> set[str]:
    if isinstance(aliases, str):
        aliases = [aliases]

    values = [
        service_name,
        *(aliases or []),
    ]
    return {
        normalise_service_match_text(value)
        for value in values
        if normalise_service_match_text(value)
    }


def reject_active_service_alias_conflicts(
    cleaned: dict,
    service_id: int | None = None,
) -> None:
    if cleaned.get("is_active") is False:
        return

    new_aliases = service_alias_identity_set(
        cleaned.get("service_name"),
        cleaned.get("aliases") or [],
    )

    if not new_aliases:
        return

    response = (
        supabase.table("services")
        .select("id, service_name, aliases, is_active")
        .eq("business_slug", BUSINESS_SLUG)
        .eq("is_active", True)
        .execute()
    )

    for row in response.data or []:
        if service_id is not None and int(row.get("id") or 0) == int(service_id):
            continue

        overlap = new_aliases.intersection(
            service_alias_identity_set(
                row.get("service_name"),
                row.get("aliases") or [],
            )
        )

        if overlap:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Another active service already uses one of these "
                    f"aliases: {', '.join(sorted(overlap))}."
                ),
            )


@app.get("/admin/services")
async def list_admin_services(request: Request):
    require_authenticated_admin(request)
    response = (
        supabase.table("services")
        .select("*")
        .eq("business_slug", BUSINESS_SLUG)
        .execute()
    )
    rows = sorted(
        (admin_service_response(row) for row in response.data or []),
        key=lambda row: (
            str(row.get("category_key") or ""),
            int(row.get("sort_order") or 0),
            str(row.get("service_name") or ""),
            str(row.get("variant_name") or ""),
        ),
    )
    return {"services": rows}


@app.post("/admin/services")
async def create_admin_service(payload: AdminServicePayload, request: Request):
    require_authenticated_admin(request)
    values = payload.model_dump(exclude_unset=True)
    cleaned = clean_admin_service_payload(values, require_all=True)
    reject_active_service_alias_conflicts(cleaned)
    response = supabase.table("services").insert(cleaned).execute()
    if not response.data:
        raise HTTPException(status_code=500, detail="Service could not be created.")
    invalidate_service_caches()
    return {"service": admin_service_response(response.data[0])}


@app.patch("/admin/services/{service_id}")
async def update_admin_service(service_id: int, payload: AdminServicePayload, request: Request):
    require_authenticated_admin(request)
    existing = (
        supabase.table("services")
        .select("*")
        .eq("business_slug", BUSINESS_SLUG)
        .eq("id", service_id)
        .limit(1)
        .execute()
    )
    if not existing.data:
        raise HTTPException(status_code=404, detail="Service not found.")
    current = admin_service_response(existing.data[0])
    current.update(payload.model_dump(exclude_unset=True))
    cleaned = clean_admin_service_payload(current, require_all=True)
    reject_active_service_alias_conflicts(cleaned, service_id)
    response = (
        supabase.table("services")
        .update(cleaned)
        .eq("business_slug", BUSINESS_SLUG)
        .eq("id", service_id)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=500, detail="Service could not be updated.")
    invalidate_service_caches()
    return {"service": admin_service_response(response.data[0])}


async def set_admin_service_active(service_id: int, active: bool, request: Request):
    require_authenticated_admin(request)
    if active:
        existing = (
            supabase.table("services")
            .select("*")
            .eq("business_slug", BUSINESS_SLUG)
            .eq("id", service_id)
            .limit(1)
            .execute()
        )
        if not existing.data:
            raise HTTPException(status_code=404, detail="Service not found.")
        cleaned = admin_service_response(existing.data[0])
        cleaned["is_active"] = True
        reject_active_service_alias_conflicts(cleaned, service_id)

    response = (
        supabase.table("services")
        .update({"is_active": active})
        .eq("business_slug", BUSINESS_SLUG)
        .eq("id", service_id)
        .execute()
    )
    if not response.data:
        raise HTTPException(status_code=404, detail="Service not found.")
    invalidate_service_caches()
    return {"service": admin_service_response(response.data[0])}


@app.patch("/admin/services/{service_id}/deactivate")
async def deactivate_admin_service(service_id: int, request: Request):
    return await set_admin_service_active(service_id, False, request)


@app.patch("/admin/services/{service_id}/activate")
async def activate_admin_service(service_id: int, request: Request):
    return await set_admin_service_active(service_id, True, request)


def load_confirmation_context(booking_request_id: int) -> tuple[dict, dict, list[dict]]:
    request_response = (
        supabase.table("booking_requests")
        .select("*")
        .eq("id", booking_request_id)
        .limit(1)
        .execute()
    )

    if not request_response.data:
        raise HTTPException(status_code=404, detail="Appointment request not found.")

    booking_request = request_response.data[0]
    conversation_response = (
        supabase.table("conversations")
        .select("*")
        .eq("session_id", booking_request.get("session_id"))
        .limit(1)
        .execute()
    )
    item_response = (
        supabase.table("booking_items")
        .select("*")
        .eq("booking_request_id", booking_request_id)
        .order("item_order")
        .execute()
    )

    conversation = (
        conversation_response.data[0]
        if conversation_response.data
        else {}
    )
    return booking_request, conversation, list(item_response.data or [])


def confirmation_missing_fields(
    booking_request: dict,
    conversation: dict,
    items: list[dict],
) -> list[str]:
    checks = {
        "treatment": bool(items and items[0].get("service_name")),
        "duration": bool(booking_request.get("total_duration_minutes")),
        "requested date": bool(booking_request.get("preferred_date")),
        "requested time": bool(booking_request.get("preferred_time")),
        "name": bool(conversation.get("name")),
    }

    if normalise_source(conversation.get("source")) != "instagram":
        checks["phone"] = bool(conversation.get("phone"))

    return [label for label, present in checks.items() if not present]


def confirmation_json_response(
    success: bool,
    status: str,
    message: str,
    status_code: int = 200,
    extra: dict | None = None,
) -> JSONResponse:
    content = {
        "success": success,
        "status": status,
        "message": message,
    }

    if extra:
        content.update(extra)

    return JSONResponse(
        status_code=status_code,
        content=content,
    )


def release_confirmation_claim(
    booking_request_id: int,
    claim_token: str,
) -> None:
    try:
        supabase.rpc(
            "release_booking_request_confirmation",
            {
                "target_request_id": booking_request_id,
                "lock_token": claim_token,
            },
        ).execute()
    except Exception:
        logger.exception(
            "Could not release confirmation claim for booking request %s",
            booking_request_id,
        )


def load_channel_contact(conversation: dict) -> dict:
    contact_id = conversation.get("channel_contact_id")

    if not contact_id and conversation.get("source") == "instagram":
        session_id = str(conversation.get("session_id") or "")
        session_parts = session_id.split(":")

        if len(session_parts) >= 3 and session_parts[0] == "instagram":
            message_response = (
                supabase.table("messages")
                .select("created_at")
                .eq("session_id", session_id)
                .eq("role", "user")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            last_customer_message_at = (
                message_response.data[0].get("created_at")
                if message_response.data
                else None
            )

            if last_customer_message_at:
                return save_instagram_channel_contact(
                    session_parts[1],
                    session_id,
                    last_customer_message_at,
                )

        return {}

    response = (
        supabase.table("channel_contacts")
        .select("*")
        .eq("id", contact_id)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else {}


def parse_utc_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return (
            parsed.replace(tzinfo=timezone.utc)
            if parsed.tzinfo is None
            else parsed.astimezone(timezone.utc)
        )
    except ValueError:
        return None


def instagram_reply_window_is_open(contact: dict) -> bool:
    last_customer_message_at = parse_utc_datetime(
        contact.get("last_customer_message_at")
    )

    if not last_customer_message_at:
        return False

    elapsed = datetime.now(timezone.utc) - last_customer_message_at
    return timedelta(0) <= elapsed <= timedelta(hours=24)


def save_confirmation_message_status(
    booking_request_id: int,
    status: str,
    error_message: str | None = None,
    sent_at: str | None = None,
) -> bool:
    try:
        response = (
            supabase.table("booking_requests")
            .update({
                "confirmation_message_status": status,
                "confirmation_message_sent_at": sent_at,
                "confirmation_message_error": error_message,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            .eq("id", booking_request_id)
            .execute()
        )
        return bool(response.data)
    except Exception:
        logger.exception(
            "Could not save confirmation message status for booking request %s",
            booking_request_id,
        )
        return False


def instagram_confirmation_text(
    booking_request: dict,
    items: list[dict],
) -> str:
    treatment = " + ".join(
        str(item.get("service_name"))
        for item in items
        if item.get("service_name")
    ) or "your treatment"
    requested_date = format_customer_date(booking_request.get("preferred_date"))
    requested_time = format_customer_time(booking_request.get("preferred_time"))
    return (
        f"Your appointment has been confirmed for {treatment} on "
        f"{requested_date} at {requested_time}. We look forward to seeing you."
    )


def confirmation_delivery_result(
    booking_request: dict,
    conversation: dict,
    items: list[dict],
) -> dict:
    booking_request_id = int(booking_request["id"])
    existing_status = booking_request.get("confirmation_message_status")

    if existing_status == "sent":
        return {
            "status": "already_confirmed",
            "message": "This appointment was already confirmed and its Instagram confirmation was already sent.",
            "retry_available": False,
        }

    if conversation.get("source") != "instagram":
        save_confirmation_message_status(
            booking_request_id,
            "skipped_non_instagram",
        )
        return {
            "status": "confirmed",
            "message": "Appointment confirmed and added to Google Calendar. This was not an Instagram lead.",
            "retry_available": False,
        }

    try:
        contact = load_channel_contact(conversation)
    except Exception:
        logger.exception(
            "Could not load Instagram contact for booking request %s",
            booking_request_id,
        )
        contact = {}

    recipient_id = contact.get("external_user_id")

    if not recipient_id:
        error_message = "Instagram contact details are unavailable."
        save_confirmation_message_status(
            booking_request_id,
            "failed",
            error_message,
        )
        return {
            "status": "confirmed_message_failed",
            "message": f"Appointment confirmed, but the Instagram message could not be sent: {error_message}",
            "retry_available": False,
        }

    if not instagram_reply_window_is_open(contact):
        save_confirmation_message_status(
            booking_request_id,
            "skipped_reply_window_expired",
        )
        return {
            "status": "confirmed_reply_window_expired",
            "message": "Appointment confirmed, but the Instagram reply window has expired. Please message the customer manually.",
            "retry_available": False,
        }

    send_result = send_instagram_text_message(
        recipient_id=str(recipient_id),
        text=instagram_confirmation_text(booking_request, items),
    )

    if send_result.get("success"):
        sent_at = datetime.now(timezone.utc).isoformat()
        saved = save_confirmation_message_status(
            booking_request_id,
            "sent",
            sent_at=sent_at,
        )

        if not saved:
            return {
                "status": "confirmed_message_tracking_failed",
                "message": "Appointment confirmed and Instagram confirmation sent, but the delivery status could not be saved.",
                "retry_available": False,
            }

        return {
            "status": "confirmed_message_sent",
            "message": "Appointment confirmed and Instagram confirmation sent.",
            "retry_available": False,
        }

    error_message = str(
        send_result.get("message") or "Instagram messaging failed."
    )
    save_confirmation_message_status(
        booking_request_id,
        "failed",
        error_message,
    )
    return {
        "status": "confirmed_message_failed",
        "message": f"Appointment confirmed, but the Instagram message could not be sent: {error_message}",
        "retry_available": True,
    }


def safely_deliver_instagram_confirmation(
    booking_request: dict,
    conversation: dict,
    items: list[dict],
) -> dict:
    try:
        return confirmation_delivery_result(
            booking_request,
            conversation,
            items,
        )
    except Exception:
        booking_request_id = booking_request.get("id")
        logger.exception(
            "Unexpected Instagram confirmation error for booking request %s",
            booking_request_id,
        )

        if booking_request_id:
            save_confirmation_message_status(
                int(booking_request_id),
                "failed",
                "Instagram confirmation could not be processed.",
            )

        return {
            "status": "confirmed_message_failed",
            "message": "Appointment confirmed, but the Instagram message could not be sent: Instagram confirmation could not be processed.",
            "retry_available": False,
        }


def existing_confirmation_delivery_result(
    booking_request: dict,
    conversation: dict,
) -> dict:
    status = booking_request.get("confirmation_message_status")

    if status == "sent":
        return {
            "status": "already_confirmed",
            "message": "This appointment was already confirmed and its Instagram confirmation was already sent.",
            "retry_available": False,
        }

    retry_available = False

    if status == "failed" and conversation.get("source") == "instagram":
        try:
            contact = load_channel_contact(conversation)
            retry_available = bool(
                contact.get("external_user_id")
                and instagram_reply_window_is_open(contact)
            )
        except Exception:
            logger.exception(
                "Could not check Instagram retry eligibility for booking request %s",
                booking_request.get("id"),
            )

    return {
        "status": "already_confirmed",
        "message": "This appointment was already confirmed.",
        "retry_available": retry_available,
    }


def load_active_payment_request(
    booking_request_id: int,
) -> dict | None:
    expire_pending_payment_holds()

    response = (
        supabase.table("payment_requests")
        .select("*")
        .eq("booking_request_id", booking_request_id)
        .eq("status", "pending")
        .gt("hold_expires_at", datetime.now(timezone.utc).isoformat())
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


def load_sumup_config() -> dict:
    """Load required SumUp credentials and hold settings at request time."""
    api_key = str(os.getenv("SUMUP_API_KEY") or "").strip()
    merchant_code = str(os.getenv("SUMUP_MERCHANT_CODE") or "").strip()
    api_key_present = bool(api_key)
    merchant_code_present = bool(merchant_code)

    logger.info(
        "SumUp config check: api_key_present=%s, merchant_code_present=%s",
        str(api_key_present).lower(),
        str(merchant_code_present).lower(),
    )

    if not api_key_present or not merchant_code_present:
        raise RuntimeError("SumUp sandbox configuration is incomplete.")

    return {
        "api_key": api_key,
        "merchant_code": merchant_code,
        "hold_minutes": read_positive_int_environment_variable(
            "SUMUP_HOLD_MINUTES",
            30,
        ),
    }


def safe_sumup_error_details(response_body: str) -> tuple[str, str]:
    """Return a safe dashboard reason and redacted diagnostic response body."""
    cleaned_body = str(response_body or "").strip()
    diagnostic_body = re.sub(
        r"(?i)([\"']?(?:authorization|access[_ -]?token|api[_ -]?key|"
        r"bearer|token|secret|password)"
        r"[\"']?\s*[:=]\s*)[\"']?[^,\"'\s}]+",
        r"\1[redacted]",
        cleaned_body,
    )[:2000]
    diagnostic_body = re.sub(
        r"(?i)\b(?:sup_sk_|sk_test_|sk_live_)[A-Za-z0-9._-]+",
        "[redacted-key]",
        diagnostic_body,
    )
    diagnostic_body = re.sub(
        r"(?i)([\"']?merchant_code[\"']?\s*[:=]\s*)[\"']?[^,\"'\s}]+",
        r"\1[redacted]",
        diagnostic_body,
    )
    reason = ""

    try:
        parsed = json.loads(cleaned_body)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        for key in ("message", "error_message", "detail", "error", "code"):
            value = parsed.get(key)

            if isinstance(value, str) and value.strip():
                reason = value.strip()
                break

            if isinstance(value, dict):
                nested = value.get("message") or value.get("description")

                if isinstance(nested, str) and nested.strip():
                    reason = nested.strip()
                    break

    if not reason and cleaned_body and not cleaned_body.lstrip().startswith("<"):
        reason = cleaned_body

    reason = re.sub(
        r"(?i)\b(?:sup_sk_|sk_test_|sk_live_)[A-Za-z0-9._-]+",
        "[redacted-key]",
        reason,
    )
    reason = re.sub(
        r"(?i)(authorization|access[_ -]?token|api[_ -]?key|bearer|token|"
        r"secret|password)\s*[:=]\s*\S+",
        r"\1=[redacted]",
        reason,
    )
    reason = re.sub(
        r"(?i)merchant_code\s*[:=]\s*\S+",
        "merchant_code=[redacted]",
        reason,
    )
    reason = re.sub(r"\s+", " ", reason).strip()[:300]
    return reason, diagnostic_body or "[empty response body]"


def create_sumup_hosted_checkout(
    checkout_reference: str,
    amount_pence: int,
    description: str,
    valid_until: str,
) -> dict:
    sumup_config = load_sumup_config()
    sumup_api_key = sumup_config["api_key"]
    sumup_merchant_code = sumup_config["merchant_code"]
    merchant_code_present = bool(sumup_merchant_code)

    payload = {
        "checkout_reference": checkout_reference,
        "merchant_code": sumup_merchant_code,
        "amount": round(amount_pence / 100, 2),
        "currency": "GBP",
        "description": description[:255],
        "valid_until": valid_until,
        "hosted_checkout": {"enabled": True},
    }
    checkout_request = urllib_request.Request(
        SUMUP_CHECKOUT_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {sumup_api_key}",
            "Content-Type": "application/json",
            "User-Agent": "veronika-chatbot/1.0",
        },
        method="POST",
    )
    logger.info(
        "SumUp checkout request: endpoint=%s checkout_reference=%s "
        "amount=%s currency=%s merchant_code_present=%s "
        "hosted_checkout_enabled=%s",
        SUMUP_CHECKOUT_ENDPOINT,
        checkout_reference,
        payload["amount"],
        payload["currency"],
        str(merchant_code_present).lower(),
        str(payload["hosted_checkout"]["enabled"]).lower(),
    )

    try:
        with urllib_request.urlopen(checkout_request, timeout=20) as response:
            response_status = response.getcode()
            response_body = response.read().decode("utf-8", errors="replace")

        _, diagnostic_body = safe_sumup_error_details(response_body)
        logger.info(
            "SumUp checkout response: endpoint=%s http_status=%s "
            "response_body=%s checkout_reference=%s amount=%s currency=%s "
            "merchant_code_present=%s hosted_checkout_enabled=%s",
            SUMUP_CHECKOUT_ENDPOINT,
            response_status,
            diagnostic_body,
            checkout_reference,
            payload["amount"],
            payload["currency"],
            str(merchant_code_present).lower(),
            str(payload["hosted_checkout"]["enabled"]).lower(),
        )

        try:
            checkout = json.loads(response_body)
        except json.JSONDecodeError as error:
            logger.warning(
                "SumUp checkout returned invalid JSON: endpoint=%s "
                "http_status=%s response_body=%s checkout_reference=%s "
                "amount=%s currency=%s merchant_code_present=%s "
                "hosted_checkout_enabled=%s",
                SUMUP_CHECKOUT_ENDPOINT,
                response_status,
                diagnostic_body,
                checkout_reference,
                payload["amount"],
                payload["currency"],
                str(merchant_code_present).lower(),
                str(payload["hosted_checkout"]["enabled"]).lower(),
            )
            raise RuntimeError(
                "Could not create SumUp deposit link: SumUp returned an invalid response."
            ) from error
    except urllib_error.HTTPError as error:
        try:
            response_body = error.read().decode("utf-8", errors="replace")
        finally:
            error.close()

        safe_reason, diagnostic_body = safe_sumup_error_details(response_body)
        logger.warning(
            "SumUp checkout rejected: endpoint=%s http_status=%s "
            "response_body=%s checkout_reference=%s amount=%s currency=%s "
            "merchant_code_present=%s hosted_checkout_enabled=%s",
            SUMUP_CHECKOUT_ENDPOINT,
            error.code,
            diagnostic_body,
            checkout_reference,
            payload["amount"],
            payload["currency"],
            str(merchant_code_present).lower(),
            str(payload["hosted_checkout"]["enabled"]).lower(),
        )
        raise RuntimeError(
            f"Could not create SumUp deposit link. SumUp returned HTTP "
            f"{error.code}"
            + (f": {safe_reason}" if safe_reason else ".")
        ) from error
    except (TimeoutError, urllib_error.URLError) as error:
        raise RuntimeError("SumUp checkout creation could not be reached.") from error

    if not isinstance(checkout, dict):
        raise RuntimeError("SumUp returned an invalid checkout response.")

    checkout_id = checkout.get("id")
    hosted_checkout_url = checkout.get("hosted_checkout_url")

    if not checkout_id or not hosted_checkout_url:
        _, diagnostic_body = safe_sumup_error_details(
            json.dumps(checkout, ensure_ascii=False)
        )
        logger.warning(
            "SumUp checkout response missing hosted checkout data: endpoint=%s "
            "response_body=%s checkout_reference=%s amount=%s currency=%s "
            "merchant_code_present=%s hosted_checkout_enabled=%s",
            SUMUP_CHECKOUT_ENDPOINT,
            diagnostic_body,
            checkout_reference,
            payload["amount"],
            payload["currency"],
            str(merchant_code_present).lower(),
            str(payload["hosted_checkout"]["enabled"]).lower(),
        )
        raise RuntimeError(
            "SumUp created the checkout but did not return a hosted payment URL."
        )

    return {
        "provider_checkout_id": str(checkout_id),
        "hosted_checkout_url": str(hosted_checkout_url),
    }


def deposit_instagram_text(
    booking_request: dict,
    items: list[dict],
    amount_pence: int,
    hosted_checkout_url: str,
) -> str:
    treatment = " + ".join(
        str(item.get("service_name"))
        for item in items
        if item.get("service_name")
    ) or "your treatment"
    amount = f"{amount_pence / 100:.2f}"
    requested_date = format_customer_date(booking_request.get("preferred_date"))
    requested_time = format_customer_time(booking_request.get("preferred_time"))
    hold_minutes = load_sumup_config()["hold_minutes"]
    return (
        f"Veronika has reviewed your appointment request for {treatment} on "
        f"{requested_date} at {requested_time}.\n\n"
        f"To secure the slot, please pay the £{amount} deposit using this "
        f"secure SumUp link:\n{hosted_checkout_url}\n\n"
        f"The slot will be held for {hold_minutes} minutes. Your "
        "appointment will be confirmed once the deposit has been received."
    )


def update_payment_delivery_status(
    payment_request_id: str,
    status: str,
    error_message: str | None = None,
    sent_at: str | None = None,
) -> None:
    (
        supabase.table("payment_requests")
        .update({
            "message_delivery_status": status,
            "message_delivery_error": error_message,
            "message_sent_at": sent_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("id", payment_request_id)
        .execute()
    )


def deliver_deposit_link(
    payment_request: dict,
    booking_request: dict,
    conversation: dict,
    items: list[dict],
) -> dict:
    payment_request_id = str(payment_request["id"])

    if normalise_source(conversation.get("source")) != "instagram":
        update_payment_delivery_status(payment_request_id, "not_applicable")
        return {
            "delivery_status": "not_applicable",
            "delivery_message": (
                "Deposit link created. Copy the payment link from the dashboard."
            ),
        }

    try:
        contact = load_channel_contact(conversation)
    except Exception:
        logger.exception(
            "Could not load Instagram contact for deposit request %s",
            payment_request_id,
        )
        contact = {}

    if not contact.get("external_user_id"):
        error_message = "Instagram contact details are unavailable."
        update_payment_delivery_status(
            payment_request_id,
            "failed",
            error_message,
        )
        return {
            "delivery_status": "failed",
            "delivery_message": error_message,
        }

    if not instagram_reply_window_is_open(contact):
        error_message = "Instagram reply window has expired."
        update_payment_delivery_status(
            payment_request_id,
            "reply_window_expired",
            error_message,
        )
        return {
            "delivery_status": "reply_window_expired",
            "delivery_message": error_message,
        }

    logger.info(
        "Instagram deposit delivery attempted for booking request %s",
        booking_request.get("id"),
    )
    send_result = send_instagram_text_message(
        recipient_id=str(contact["external_user_id"]),
        text=deposit_instagram_text(
            booking_request,
            items,
            int(payment_request["amount_pence"]),
            str(payment_request["hosted_checkout_url"]),
        ),
    )

    if send_result.get("success"):
        sent_at = datetime.now(timezone.utc).isoformat()
        update_payment_delivery_status(
            payment_request_id,
            "sent",
            sent_at=sent_at,
        )
        logger.info(
            "Instagram deposit delivery succeeded for booking request %s",
            booking_request.get("id"),
        )
        return {
            "delivery_status": "sent",
            "delivery_message": "Deposit link sent by Instagram.",
        }

    error_message = str(
        send_result.get("message") or "Instagram delivery failed."
    )[:500]
    update_payment_delivery_status(
        payment_request_id,
        "failed",
        error_message,
    )
    logger.warning(
        "Instagram deposit delivery failed for booking request %s",
        booking_request.get("id"),
    )
    return {
        "delivery_status": "failed",
        "delivery_message": error_message,
    }


def temporary_calendar_hold_details(
    payment_request: dict,
    booking_request: dict,
    conversation: dict,
    items: list[dict],
) -> tuple[str, str]:
    treatment = " + ".join(
        str(item.get("service_name"))
        for item in items
        if item.get("service_name")
    ) or "appointment"
    customer_name = str(conversation.get("name") or "Customer")
    duration_minutes = int(booking_request["total_duration_minutes"])
    amount = f"{int(payment_request['amount_pence']) / 100:.2f}"
    title = f"TEMP HOLD \u2014 awaiting deposit \u2014 {customer_name}"
    description = "\n".join([
        "Temporary deposit hold.",
        f"Treatment: {treatment}",
        f"Duration: {duration_minutes} minutes",
        f"Deposit requested: \u00a3{amount}",
        f"Hold expires: {payment_request.get('hold_expires_at')}",
        f"Customer phone: {conversation.get('phone')}",
        f"Booking request ID: {booking_request.get('id')}",
        "",
        "This is not a confirmed appointment.",
    ])
    return title, description


def ensure_temporary_calendar_hold(
    payment_request: dict,
    booking_request: dict,
    conversation: dict,
    items: list[dict],
    creation_claimed: bool = False,
) -> dict:
    """Create or recover one temporary calendar event for a payment hold."""
    payment_request_id = str(payment_request["id"])

    if (
        payment_request.get("calendar_hold_event_id")
        and payment_request.get("calendar_hold_status") == "active"
    ):
        return {
            "calendar_hold_status": "active",
            "calendar_hold_event_id": payment_request["calendar_hold_event_id"],
            "calendar_hold_message": "Temporary Google Calendar hold added.",
        }

    if not creation_claimed:
        logger.info(
            "Temporary calendar hold retry attempted; payment_request_id_present=true"
        )
        try:
            existing_event = find_google_calendar_event_by_private_property(
                "paymentRequestId",
                payment_request_id,
            )
        except Exception:
            logger.exception("Could not check for an existing temporary event")
            existing_event = None

        if existing_event and existing_event.get("id"):
            now_iso = datetime.now(timezone.utc).isoformat()
            update_response = (
                supabase.table("payment_requests")
                .update({
                    "calendar_hold_event_id": existing_event["id"],
                    "calendar_hold_created_at": now_iso,
                    "calendar_hold_status": "active",
                    "calendar_hold_error": None,
                    "updated_at": now_iso,
                })
                .eq("id", payment_request_id)
                .execute()
            )

            if not update_response.data:
                raise RuntimeError("Recovered temporary event could not be saved.")

            return {
                "calendar_hold_status": "active",
                "calendar_hold_event_id": existing_event["id"],
                "calendar_hold_message": "Temporary Google Calendar hold added.",
            }

        claim_response = (
            supabase.table("payment_requests")
            .update({
                "calendar_hold_status": "creating",
                "calendar_hold_error": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            .eq("id", payment_request_id)
            .is_("calendar_hold_event_id", "null")
            .in_("calendar_hold_status", ["failed", "active"])
            .execute()
        )

        if not claim_response.data:
            return {
                "calendar_hold_status": (
                    payment_request.get("calendar_hold_status") or "creating"
                ),
                "calendar_hold_event_id": None,
                "calendar_hold_message": (
                    "Deposit link created. Temporary Google Calendar hold "
                    "creation is already in progress."
                ),
            }

    logger.info(
        "Temporary calendar hold creation started; payment_request_id_present=true"
    )
    safe_error = "Google Calendar temporary hold could not be added."

    try:
        start_time = parse_utc_datetime(payment_request.get("held_start_at"))
        end_time = parse_utc_datetime(payment_request.get("held_end_at"))

        if not start_time or not end_time:
            raise RuntimeError("Payment hold interval is invalid.")

        title, description = temporary_calendar_hold_details(
            payment_request,
            booking_request,
            conversation,
            items,
        )
        event = create_google_calendar_event(
            start_time.astimezone(BUSINESS_TIMEZONE),
            end_time.astimezone(BUSINESS_TIMEZONE),
            title,
            description,
            private_properties={"paymentRequestId": payment_request_id},
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        update_response = (
            supabase.table("payment_requests")
            .update({
                "calendar_hold_event_id": event["id"],
                "calendar_hold_created_at": now_iso,
                "calendar_hold_status": "active",
                "calendar_hold_error": None,
                "updated_at": now_iso,
            })
            .eq("id", payment_request_id)
            .execute()
        )

        if not update_response.data:
            raise RuntimeError("Temporary calendar event ID could not be saved.")

        logger.info(
            "Temporary calendar hold created; calendar_hold_event_id_present=true"
        )
        return {
            "calendar_hold_status": "active",
            "calendar_hold_event_id": event["id"],
            "calendar_hold_message": "Temporary Google Calendar hold added.",
        }
    except Exception as error:
        if isinstance(error, urllib_error.HTTPError):
            error.close()
        logger.warning(
            "Temporary calendar hold creation failed: %s",
            type(error).__name__,
        )
        try:
            (
                supabase.table("payment_requests")
                .update({
                    "calendar_hold_status": "failed",
                    "calendar_hold_error": safe_error,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                .eq("id", payment_request_id)
                .execute()
            )
        except Exception:
            logger.exception("Could not save temporary calendar hold failure")

        return {
            "calendar_hold_status": "failed",
            "calendar_hold_event_id": None,
            "calendar_hold_message": (
                "Deposit link created, but the temporary Google Calendar hold "
                "could not be added. The slot is still reserved in the system."
            ),
        }


@app.post("/admin/booking-requests/{booking_request_id}/request-deposit")
async def request_booking_deposit(
    booking_request_id: int,
    deposit: DepositRequest,
    request: Request,
):
    logger.info(
        "Deposit request endpoint reached: booking_request_id=%s amount_pence=%s",
        booking_request_id,
        deposit.amount_pence,
    )

    try:
        require_authenticated_admin(request)
    except HTTPException as error:
        return confirmation_json_response(
            False,
            "error",
            str(error.detail),
            error.status_code,
        )

    try:
        booking_request, conversation, items = load_confirmation_context(
            booking_request_id
        )
    except HTTPException as error:
        return confirmation_json_response(
            False,
            "error",
            str(error.detail),
            error.status_code,
        )
    except Exception:
        logger.exception(
            "Could not load deposit context for booking request %s",
            booking_request_id,
        )
        return confirmation_json_response(
            False,
            "error",
            "The appointment request could not be loaded. Please try again.",
            500,
        )

    if booking_request.get("request_status") not in {
        "draft",
        "ready_for_review",
        "awaiting_owner_confirmation",
    } or booking_request.get("calendar_event_id"):
        return confirmation_json_response(
            False,
            "error",
            "This appointment request is not eligible for a deposit request.",
            409,
        )

    missing = confirmation_missing_fields(
        booking_request,
        conversation,
        items,
    )

    if not conversation.get("phone"):
        missing.append("phone")

    if missing:
        return confirmation_json_response(
            False,
            "error",
            "Cannot request a deposit until these details are present: "
            + ", ".join(dict.fromkeys(missing)),
            400,
        )

    total_price_pence = int(booking_request.get("total_price_pence") or 0)

    if (
        not isinstance(deposit.amount_pence, int)
        or deposit.amount_pence < 100
        or not total_price_pence
        or deposit.amount_pence > total_price_pence
    ):
        return confirmation_json_response(
            False,
            "error",
            "Deposit must be at least £1.00 and no greater than the treatment total.",
            400,
        )

    try:
        existing_payment = load_active_payment_request(booking_request_id)
    except Exception:
        logger.exception(
            "Could not check existing deposit request for booking request %s",
            booking_request_id,
        )
        return confirmation_json_response(
            False,
            "error",
            "Existing deposit requests could not be checked. Please try again.",
            500,
        )

    if existing_payment:
        try:
            calendar_hold = ensure_temporary_calendar_hold(
                existing_payment,
                booking_request,
                conversation,
                items,
            )
        except Exception:
            logger.exception(
                "Could not retry temporary calendar hold for booking request %s",
                booking_request_id,
            )
            calendar_hold = {
                "calendar_hold_status": "failed",
                "calendar_hold_event_id": None,
                "calendar_hold_message": (
                    "Slot reserved in the system, but Google Calendar hold "
                    "could not be added."
                ),
            }
        return confirmation_json_response(
            True,
            "awaiting_deposit",
            "An active deposit link already exists.",
            extra={
                "deposit_amount_pence": existing_payment.get("amount_pence"),
                "hold_expires_at": existing_payment.get("hold_expires_at"),
                "hosted_checkout_url": existing_payment.get("hosted_checkout_url"),
                "delivery_status": existing_payment.get("message_delivery_status"),
                "delivery_message": existing_payment.get("message_delivery_error"),
                **calendar_hold,
            },
        )

    duration_minutes = int(booking_request["total_duration_minutes"])
    confirmation_state = {
        "preferred_date": booking_request["preferred_date"],
        "preferred_time": booking_request["preferred_time"],
        "duration": f"{duration_minutes} minutes",
    }
    slot = build_requested_slot(confirmation_state)

    if not slot:
        return confirmation_json_response(
            False,
            "error",
            "The requested appointment time is invalid.",
            400,
        )

    start_time, end_time, _ = slot

    try:
        availability = check_requested_calendar_slot(
            confirmation_state,
            "",
            exclude_booking_request_id=booking_request_id,
        )
    except Exception:
        logger.exception(
            "Could not recheck deposit availability for booking request %s",
            booking_request_id,
        )
        return confirmation_json_response(
            False,
            "error",
            "Calendar availability could not be verified. Please try again.",
            503,
        )
    logger.info(
        "Deposit slot availability recheck: booking_request_id=%s status=%s",
        booking_request_id,
        availability.get("status") if isinstance(availability, dict) else "invalid",
    )

    if availability.get("status") == "busy":
        competing_holds = active_payment_hold_busy_periods(
            start_time,
            end_time,
            exclude_booking_request_id=booking_request_id,
        )
        message = (
            "This slot is currently being held for another customer."
            if competing_holds
            else "This slot is no longer available. Please choose another time."
        )
        return confirmation_json_response(
            False,
            "slot_unavailable",
            message,
            409,
        )

    if availability.get("status") != "free":
        return confirmation_json_response(
            False,
            "error",
            "Calendar availability could not be verified. Please try again.",
            503,
        )

    checkout_reference = (
        f"{BUSINESS_SLUG}-deposit-{booking_request_id}-"
        f"{secrets.token_hex(8)}"
    )
    try:
        sumup_config = load_sumup_config()
    except RuntimeError as error:
        return confirmation_json_response(
            False,
            "error",
            str(error),
            502,
        )

    hold_expires_at = (
        datetime.now(timezone.utc)
        + timedelta(minutes=sumup_config["hold_minutes"])
    ).isoformat()
    service_names = " + ".join(
        str(item.get("service_name"))
        for item in items
        if item.get("service_name")
    ) or "appointment"
    logger.info(
        "SumUp checkout creation started for booking request %s",
        booking_request_id,
    )

    try:
        checkout = create_sumup_hosted_checkout(
            checkout_reference,
            deposit.amount_pence,
            f"Deposit for {service_names}",
            hold_expires_at,
        )
    except Exception as error:
        logger.warning(
            "SumUp checkout creation failed for booking request %s: %s: %s",
            booking_request_id,
            type(error).__name__,
            str(error)[:500],
        )
        return confirmation_json_response(
            False,
            "error",
            str(error),
            502,
        )

    logger.info(
        "SumUp checkout created successfully for booking request %s",
        booking_request_id,
    )
    try:
        payment_response = (
            supabase.table("payment_requests")
            .insert({
                "business_slug": BUSINESS_SLUG,
                "booking_request_id": booking_request_id,
                "provider": "sumup",
                "payment_type": "deposit",
                "amount_pence": deposit.amount_pence,
                "currency": "GBP",
                "provider_checkout_id": checkout["provider_checkout_id"],
                "checkout_reference": checkout_reference,
                "hosted_checkout_url": checkout["hosted_checkout_url"],
                "status": "pending",
                "held_start_at": start_time.isoformat(),
                "held_end_at": end_time.isoformat(),
                "hold_expires_at": hold_expires_at,
                "calendar_hold_status": "creating",
            })
            .execute()
        )
        payment_request = payment_response.data[0]
    except Exception:
        logger.exception(
            "Could not save payment hold for booking request %s",
            booking_request_id,
        )
        return confirmation_json_response(
            False,
            "error",
            "The checkout was created, but the temporary hold could not be saved. Do not send the link; please try again.",
            500,
        )

    logger.info(
        "Temporary hold saved for booking request %s until %s",
        booking_request_id,
        hold_expires_at,
    )

    calendar_hold = ensure_temporary_calendar_hold(
        payment_request,
        booking_request,
        conversation,
        items,
        creation_claimed=True,
    )

    try:
        delivery = deliver_deposit_link(
            payment_request,
            booking_request,
            conversation,
            items,
        )
    except Exception:
        logger.exception(
            "Unexpected deposit-link delivery error for booking request %s",
            booking_request_id,
        )
        delivery = {
            "delivery_status": "failed",
            "delivery_message": "Instagram delivery could not be processed.",
        }

    if calendar_hold["calendar_hold_status"] == "failed":
        message = calendar_hold["calendar_hold_message"]
    else:
        message = (
            "Deposit link created and sent."
            if delivery["delivery_status"] == "sent"
            else "Deposit link created, but it was not sent automatically."
        )
    return confirmation_json_response(
        True,
        "awaiting_deposit",
        message,
        extra={
            "deposit_amount_pence": deposit.amount_pence,
            "hold_expires_at": hold_expires_at,
            "hosted_checkout_url": checkout["hosted_checkout_url"],
            **delivery,
            **calendar_hold,
        },
    )


@app.post("/admin/confirm-appointment")
async def confirm_appointment(
    confirmation: ConfirmAppointmentRequest,
    request: Request,
):
    """
    Owner-only, idempotent appointment confirmation.

    The database claim prevents concurrent clicks from creating duplicate
    Google events. Calendar availability is always rechecked after the claim.
    """
    booking_request_id = confirmation.booking_request_id

    try:
        require_authenticated_admin(request)
    except HTTPException as error:
        return confirmation_json_response(
            False,
            "error",
            str(error.detail),
            error.status_code,
        )
    except Exception:
        logger.exception(
            "Unexpected admin authentication error while confirming booking request %s",
            booking_request_id,
        )
        return confirmation_json_response(
            False,
            "error",
            "Admin authentication could not be verified. Please sign in again.",
            401,
        )

    try:
        booking_request, conversation, items = load_confirmation_context(
            booking_request_id
        )
    except HTTPException as error:
        return confirmation_json_response(
            False,
            "error",
            str(error.detail),
            error.status_code,
        )
    except Exception:
        logger.exception(
            "Could not load confirmation context for booking request %s",
            booking_request_id,
        )
        return confirmation_json_response(
            False,
            "error",
            "The appointment request could not be loaded. Please try again.",
            500,
        )

    if (
        booking_request.get("request_status") == "confirmed"
        and booking_request.get("calendar_event_id")
    ):
        delivery = existing_confirmation_delivery_result(
            booking_request,
            conversation,
        )
        return confirmation_json_response(
            True,
            delivery["status"],
            delivery["message"],
            extra={"retry_available": delivery["retry_available"]},
        )

    if booking_request.get("request_status") in {
        "expired",
        "cancelled_by_customer",
        "completed",
        "abandoned",
    }:
        return confirmation_json_response(
            False,
            "error",
            "This appointment request is no longer active.",
            409,
        )

    missing = confirmation_missing_fields(
        booking_request,
        conversation,
        items,
    )

    if missing:
        return confirmation_json_response(
            False,
            "error",
            "Cannot confirm until these details are present: " + ", ".join(missing),
            400,
        )

    try:
        duration_minutes = int(booking_request["total_duration_minutes"])
        confirmation_state = {
            "preferred_date": booking_request["preferred_date"],
            "preferred_time": booking_request["preferred_time"],
            "duration": f"{duration_minutes} minutes",
        }
        slot = build_requested_slot(confirmation_state)
    except Exception:
        logger.exception(
            "Could not build requested slot for booking request %s",
            booking_request_id,
        )
        return confirmation_json_response(
            False,
            "error",
            "The requested appointment time is invalid.",
            400,
        )

    if not slot:
        return confirmation_json_response(
            False,
            "error",
            "The requested appointment time is invalid.",
            400,
        )

    start_time, end_time, _ = slot
    claim_token = secrets.token_urlsafe(24)

    try:
        claim_response = supabase.rpc(
            "claim_booking_request_confirmation",
            {
                "target_request_id": booking_request_id,
                "new_lock_token": claim_token,
            },
        ).execute()
    except Exception:
        logger.exception(
            "Could not claim confirmation for booking request %s",
            booking_request_id,
        )
        return confirmation_json_response(
            False,
            "error",
            "The appointment could not be locked for confirmation. Please try again.",
            500,
        )

    if not getattr(claim_response, "data", None):
        try:
            latest, _, _ = load_confirmation_context(booking_request_id)
        except Exception:
            logger.exception(
                "Could not reload booking request %s after confirmation claim conflict",
                booking_request_id,
            )
            return confirmation_json_response(
                False,
                "error",
                "The latest appointment status could not be loaded. Please refresh and try again.",
                500,
            )

        if latest.get("request_status") == "confirmed" and latest.get("calendar_event_id"):
            delivery = existing_confirmation_delivery_result(
                latest,
                conversation,
            )
            return confirmation_json_response(
                True,
                delivery["status"],
                delivery["message"],
                extra={"retry_available": delivery["retry_available"]},
            )

        return confirmation_json_response(
            False,
            "confirmation_in_progress",
            "This request is already being confirmed. Please refresh in a moment.",
            409,
        )

    try:
        event = find_google_calendar_event_for_request(booking_request_id)
    except Exception as error:
        if isinstance(error, urllib_error.HTTPError):
            error.close()
        logger.exception(
            "Could not check Google Calendar for booking request %s",
            booking_request_id,
        )
        release_confirmation_claim(booking_request_id, claim_token)
        return confirmation_json_response(
            False,
            "error",
            "Google Calendar could not be checked. Reconnect Calendar and try again.",
            502,
        )

    if not event:
        try:
            availability = check_requested_calendar_slot(
                confirmation_state,
                "",
                exclude_booking_request_id=booking_request_id,
            )
        except Exception:
            logger.exception(
                "Could not recheck availability for booking request %s",
                booking_request_id,
            )
            release_confirmation_claim(booking_request_id, claim_token)
            return confirmation_json_response(
                False,
                "error",
                "Calendar availability could not be verified. Please try again.",
                503,
            )

        if not isinstance(availability, dict):
            logger.error(
                "Calendar availability returned an invalid result for booking request %s",
                booking_request_id,
            )
            release_confirmation_claim(booking_request_id, claim_token)
            return confirmation_json_response(
                False,
                "error",
                "Calendar availability could not be verified. Please try again.",
                503,
            )

        if availability.get("status") in {
            "busy",
            "past",
            "outside_hours",
            "closed_day",
        }:
            release_confirmation_claim(booking_request_id, claim_token)
            return confirmation_json_response(
                False,
                "slot_unavailable",
                "This slot is no longer available. Please choose another time.",
                409,
            )

        if availability.get("status") != "free":
            release_confirmation_claim(booking_request_id, claim_token)
            return confirmation_json_response(
                False,
                "error",
                "Calendar availability could not be verified. Please try again.",
                503,
            )

        try:
            service_names = [
                str(item.get("service_name"))
                for item in items
                if item.get("service_name")
            ]
            treatment_title = " + ".join(service_names)
            title = f"{treatment_title} - {conversation.get('name')}"
            description = "\n".join([
                f"Customer: {conversation.get('name')}",
                f"Phone: {conversation.get('phone')}",
                f"Treatment: {treatment_title}",
                f"Duration: {duration_minutes} minutes",
                f"Source: {conversation.get('source') or 'website'}",
                f"Session ID: {booking_request.get('session_id')}",
                f"Notes: {conversation.get('notes') or 'None'}",
            ])
            event = create_google_calendar_event(
                start_time,
                end_time,
                title,
                description,
                booking_request_id,
            )
        except Exception as error:
            if isinstance(error, urllib_error.HTTPError):
                error.close()
            logger.exception(
                "Could not create Google Calendar event for booking request %s",
                booking_request_id,
            )
            release_confirmation_claim(booking_request_id, claim_token)
            return confirmation_json_response(
                False,
                "error",
                "Google Calendar could not create the appointment. Reconnect Calendar and try again.",
                502,
            )

    try:
        confirmed_at = datetime.now(timezone.utc).isoformat()
        update_response = (
            supabase.table("booking_requests")
            .update({
                "request_status": "confirmed",
                "calendar_event_id": event["id"],
                "confirmed_at": confirmed_at,
                "confirmation_lock_token": None,
                "confirmation_started_at": None,
                "updated_at": confirmed_at,
            })
            .eq("id", booking_request_id)
            .eq("confirmation_lock_token", claim_token)
            .execute()
        )

        if not update_response.data:
            raise RuntimeError("Confirmed event could not be saved.")
    except Exception:
        logger.exception(
            "Could not save confirmed state for booking request %s",
            booking_request_id,
        )
        release_confirmation_claim(booking_request_id, claim_token)
        return confirmation_json_response(
            False,
            "error",
            "The Calendar event was created, but the confirmed status could not be saved. Please try again.",
            500,
        )

    try:
        supabase.table("conversations").update({
            "active_booking_request_id": None,
            "updated_at": confirmed_at,
        }).eq(
            "session_id",
            booking_request.get("session_id"),
        ).execute()
    except Exception:
        logger.exception(
            "Confirmed booking request %s, but could not clear its active conversation request",
            booking_request_id,
        )

    confirmed_booking_request = update_response.data[0]
    delivery = safely_deliver_instagram_confirmation(
        confirmed_booking_request,
        conversation,
        items,
    )
    return confirmation_json_response(
        True,
        delivery["status"],
        delivery["message"],
        extra={"retry_available": delivery["retry_available"]},
    )


@app.post("/admin/retry-instagram-confirmation")
async def retry_instagram_confirmation(
    confirmation: ConfirmAppointmentRequest,
    request: Request,
):
    booking_request_id = confirmation.booking_request_id

    try:
        require_authenticated_admin(request)
        booking_request, conversation, items = load_confirmation_context(
            booking_request_id
        )
    except HTTPException as error:
        return confirmation_json_response(
            False,
            "error",
            str(error.detail),
            error.status_code,
        )
    except Exception:
        logger.exception(
            "Could not load Instagram confirmation retry context for booking request %s",
            booking_request_id,
        )
        return confirmation_json_response(
            False,
            "error",
            "The appointment request could not be loaded. Please try again.",
            500,
        )

    if (
        booking_request.get("request_status") != "confirmed"
        or not booking_request.get("calendar_event_id")
    ):
        return confirmation_json_response(
            False,
            "error",
            "The appointment must be confirmed before retrying the Instagram message.",
            409,
        )

    if booking_request.get("confirmation_message_status") == "sent":
        return confirmation_json_response(
            True,
            "already_sent",
            "The Instagram confirmation was already sent.",
            extra={"retry_available": False},
        )

    if booking_request.get("confirmation_message_status") != "failed":
        return confirmation_json_response(
            False,
            "error",
            "This Instagram confirmation is not eligible for retry.",
            409,
            {"retry_available": False},
        )

    delivery = safely_deliver_instagram_confirmation(
        booking_request,
        conversation,
        items,
    )
    return confirmation_json_response(
        True,
        delivery["status"],
        delivery["message"],
        extra={"retry_available": delivery["retry_available"]},
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
    "phone_declined",
    "treatment",
    "duration",
    "active_service_id",
    "active_service_name",
    "active_service_source",
    "preferred_date",
    "preferred_time",
    "notes",
    "summary",
    "verified_alternatives",
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
            "verified_alternatives",
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
        format_generic_active_services_context(),
    ]

    return "\n\n".join(
        block for block in blocks if block
    )


CUSTOMER_SERVICE_CATEGORY_ORDER = [
    "massage",
    "facials",
    "body_treatments",
    "vitamin_shots",
    "microneedling",
    "dermal_fillers",
    "fat_dissolving",
    "ultrasound",
]


CUSTOMER_SERVICE_CATEGORY_LABELS = {
    "massage": "Massage",
    "facials": "Facials",
    "body_treatments": "Body Treatments",
    "vitamin_shots": "Vitamin Shots",
    "microneedling": "Microneedling",
    "dermal_fillers": "Dermal Fillers",
    "fat_dissolving": "Fat-Dissolving Injections",
    "ultrasound": "Ultrasound",
    "other": "Other Treatments",
}


CUSTOMER_SERVICE_CATEGORY_PHRASES = {
    "massage": ["massage", "massages"],
    "facials": ["facial", "facials"],
    "body_treatments": ["body treatment", "body treatments", "body sculpting", "ems"],
    "vitamin_shots": ["vitamin shot", "vitamin shots", "b12", "vitamin c"],
    "microneedling": ["microneedling"],
    "dermal_fillers": ["dermal filler", "dermal fillers", "filler", "fillers"],
    "fat_dissolving": ["fat dissolving", "fat-dissolving", "fat dissolving injections"],
    "ultrasound": ["ultrasound"],
}


def customer_service_catalogue(
    requested_category: str | None = None,
) -> list[dict]:
    """Return customer-visible rows from the existing service loaders."""
    catalogue = []
    loaders = [
        ("massage", load_massage_services),
        ("vitamin_shots", load_vitamin_shot_services),
        ("ultrasound", load_ultrasound_services),
        ("body_treatments", load_body_treatment_services),
        ("microneedling", load_microneedling_services),
        ("facials", load_facial_services),
        ("dermal_fillers", load_dermal_filler_services),
    ]

    for category, loader in loaders:
        if requested_category and category != requested_category:
            continue

        services, _ = loader()

        for service in services:
            catalogue.append({
                **service,
                "category": category,
            })

    return catalogue


def load_active_services_for_customer_menu(
    requested_category: str | None = None,
) -> list[dict]:
    """Load customer-visible active services from public.services."""
    try:
        response = (
            supabase.table("services")
            .select(
                "service_name, category, category_key, variant_name, "
                "booking_mode, fixed_duration_minutes, "
                "allowed_durations_minutes, price_pence, "
                "price_by_duration, is_active, sort_order, display_order"
            )
            .eq("business_slug", BUSINESS_SLUG)
            .eq("is_active", True)
            .execute()
        )
    except Exception as error:
        print(f"Could not load customer service menu from Supabase: {error}")
        return customer_service_catalogue(requested_category)

    services = []

    for row in response.data or []:
        category = str(
            row.get("category_key")
            or row.get("category")
            or "other"
        ).strip() or "other"

        if requested_category and category != requested_category:
            continue

        services.append({
            **row,
            "category": category,
            "sort_order": row.get("sort_order") or row.get("display_order") or 0,
        })

    return sorted(
        services,
        key=lambda service: (
            CUSTOMER_SERVICE_CATEGORY_ORDER.index(service.get("category"))
            if service.get("category") in CUSTOMER_SERVICE_CATEGORY_ORDER
            else len(CUSTOMER_SERVICE_CATEGORY_ORDER),
            int(service.get("sort_order") or 0),
            str(service.get("service_name") or ""),
        ),
    )


def format_customer_menu_duration(minutes) -> str:
    """Use short durations that scan cleanly in a mobile menu."""
    try:
        minutes = int(minutes)
    except (TypeError, ValueError):
        return ""

    hours, remainder = divmod(minutes, 60)

    if hours and remainder:
        return f"{hours} hr {remainder} min"

    if hours:
        return f"{hours} hr"

    return f"{minutes} min"


def customer_menu_service_details(service: dict) -> str:
    durations = service.get("allowed_durations_minutes") or []
    prices = service.get("price_by_duration") or {}
    duration_parts = []

    for duration in durations:
        duration_text = format_customer_menu_duration(duration)
        price = format_price_pence(prices.get(str(duration)))
        detail = " ".join(part for part in [duration_text, price] if part)

        if detail:
            duration_parts.append(detail)

    if duration_parts:
        return " or ".join(duration_parts)

    duration = format_customer_menu_duration(
        service.get("fixed_duration_minutes")
    )
    price = format_price_pence(service.get("price_pence"))
    return " ".join(part for part in [duration, price] if part)


def compact_customer_service_name(
    service: dict,
    requested_category: str | None = None,
) -> str:
    name = str(service.get("service_name") or "").strip()
    category = str(service.get("category") or "").strip()

    if not name:
        return ""

    if requested_category:
        return name

    lowered = normalise_service_match_text(name)

    if category == "microneedling":
        return "Microneedling"

    if category == "dermal_fillers":
        return "Dermal Fillers"

    if category == "fat_dissolving" or lowered.startswith("fat dissolving"):
        return "Fat-Dissolving Injections"

    if category == "ultrasound":
        return "Ultrasound"

    if category == "vitamin_shots":
        if "b12" in lowered:
            return "B12"
        if "vitamin c" in lowered or " c shot" in f" {lowered} ":
            return "Vitamin C"
        if lowered in {"vitamin shot", "vitamin shots"}:
            return ""

    return name


def customer_menu_should_show_details(
    service: dict,
    requested_category: str | None,
) -> bool:
    if requested_category:
        return True

    category = str(service.get("category") or "")
    name = normalise_service_match_text(service.get("service_name"))

    return bool(
        category == "facials"
        and "hydra" in name
        and service.get("price_by_duration")
    )


def format_service_menu_for_customer(
    services: list[dict],
    requested_category: str | None = None,
) -> str:
    """Format active services as grouped, mobile-friendly customer text."""
    sections = []

    ordered_categories = [
        *CUSTOMER_SERVICE_CATEGORY_ORDER,
        "other",
    ]

    for category in ordered_categories:
        if requested_category and category != requested_category:
            continue

        label = CUSTOMER_SERVICE_CATEGORY_LABELS.get(
            category,
            category.replace("_", " ").title(),
        )
        bullets = []
        seen = set()

        for service in services:
            if service.get("category") != category:
                continue

            name = compact_customer_service_name(
                service,
                requested_category=requested_category,
            )

            if not name:
                continue

            details = ""

            if customer_menu_should_show_details(
                service,
                requested_category,
            ):
                details = customer_menu_service_details(service)

            key = (
                normalise_service_match_text(name),
                normalise_service_match_text(details),
            )

            if key in seen:
                continue

            seen.add(key)

            if details:
                bullets.append(f"\u2022 {name} - {details}")
            else:
                bullets.append(f"\u2022 {name}")

        if not bullets:
            continue

        sections.append("\n".join([label, *bullets]))

    if not sections:
        return ""

    intro = (
        "Sure - here are the prices:"
        if requested_category
        else "Sure - here's a quick overview:"
    )
    closing = (
        "Tell me what you're interested in and I can help check availability."
    )
    return "\n\n".join([intro, *sections, closing])


def format_services_for_customer(
    services: list[dict],
    requested_category: str | None = None,
) -> str:
    return format_service_menu_for_customer(
        services,
        requested_category=requested_category,
    )


def requested_customer_service_menu_category(
    latest_message: str,
) -> str | None:
    """Return a broad requested category, or an empty marker for a full menu."""
    cleaned = normalise_service_match_text(latest_message)

    if not cleaned:
        return None

    general_phrases = [
        "what do you offer",
        "what do you do",
        "what services",
        "which services",
        "what treatments",
        "which treatments",
        "services do you offer",
        "services do you do",
        "treatments do you offer",
        "treatments do you do",
        "treatments are available",
        "what treatments are available",
        "available treatments",
        "full service list",
        "full treatment list",
        "full price list",
        "price list",
        "service menu",
        "treatment menu",
    ]

    if (
        any(phrase in cleaned for phrase in general_phrases)
        or cleaned in {
            "prices",
            "prices please",
            "services",
            "treatments",
            "menu",
        }
    ):
        return ""

    category_finders = {
        "massage": find_massage_service,
        "vitamin_shots": find_vitamin_shot_service,
        "ultrasound": find_ultrasound_service,
        "body_treatments": find_body_treatment_service,
        "microneedling": find_microneedling_service,
        "facials": find_facial_service,
        "dermal_fillers": find_dermal_filler_service,
    }
    menu_language = [
        "price",
        "prices",
        "cost",
        "how much",
        "do you do",
        "do you offer",
        "do you have",
        "what",
        "which",
        "list",
        "menu",
    ]

    for category, phrases in CUSTOMER_SERVICE_CATEGORY_PHRASES.items():
        broad_category = next(
            (phrase for phrase in phrases if phrase in cleaned),
            None,
        )

        if not broad_category:
            continue

        matched_service = (
            category_finders[category](latest_message)
            if category in category_finders
            else None
        )

        if (
            matched_service
            and matched_service.get("booking_mode") != "choose_variant"
        ):
            return None

        if cleaned in phrases or any(
            phrase in cleaned for phrase in menu_language
        ):
            return category

    return None


def build_customer_service_menu_reply(latest_message: str) -> str:
    requested_category = requested_customer_service_menu_category(
        latest_message
    )

    if requested_category is None:
        return ""

    return format_service_menu_for_customer(
        load_active_services_for_customer_menu(requested_category or None),
        requested_category=requested_category or None,
    )


def metadata_for_specific_price_question(message: str) -> dict | None:
    metadata = resolve_service_metadata_from_message(message)

    if metadata and metadata.get("service_name"):
        return metadata

    identity = resolve_latest_structured_service(message)

    if not identity or not identity.get("service_name"):
        return None

    return authoritative_service_metadata(identity.get("service_name")) or identity


def build_specific_price_reply(latest_message: str) -> str:
    if not customer_asks_price_question(latest_message):
        return ""

    metadata = metadata_for_specific_price_question(latest_message)

    if not metadata or not metadata.get("service_name"):
        return ""

    if metadata.get("booking_mode") == "choose_variant":
        return ""

    service_name = str(metadata.get("service_name") or "").strip()
    allowed = {
        int(value)
        for value in metadata.get("allowed_durations_minutes") or []
        if str(value).isdigit()
    }
    fixed_minutes = metadata.get("fixed_duration_minutes")
    duration_candidates = duration_candidates_from_text(
        latest_message,
        allowed_bare_minutes=allowed or None,
    )
    duration_minutes = None

    if fixed_minutes not in [None, ""]:
        if not duration_candidates:
            return ""
        duration_minutes = int(fixed_minutes)
    elif len(duration_candidates) > 1:
        options = format_duration_options(allowed) if allowed else "the duration"
        return f"Which duration would you like the price for: {options}?"
    elif len(duration_candidates) == 1:
        duration_minutes = next(iter(duration_candidates))
    elif len(allowed) == 1:
        duration_minutes = next(iter(allowed))

    if duration_minutes is None:
        if allowed:
            price_parts = []
            prices = metadata.get("price_by_duration") or {}

            for minutes in sorted(allowed):
                price = format_price_pence(prices.get(str(minutes)))

                if price:
                    price_parts.append(
                        f"{format_customer_menu_duration(minutes)} {price}"
                    )

            if price_parts:
                return (
                    f"{service_name} prices are: "
                    + ", ".join(price_parts)
                    + ". Would you like me to check availability?"
                )

        return ""

    if allowed and duration_minutes not in allowed:
        return (
            f"For {service_name}, the available durations are "
            f"{format_duration_options(allowed)}."
        )

    price = simple_request_price_pence(metadata, duration_minutes)

    if price is None:
        return ""

    return (
        f"{service_name} for {duration_minutes} minutes is "
        f"{format_price_pence(price)}. Would you like me to check availability?"
    )


def format_generic_active_services_context() -> str:
    """Include valid active owner-edited services, including new categories."""
    try:
        response = (
            supabase.table("services")
            .select(
                "service_name, category, booking_mode, fixed_duration_minutes, "
                "allowed_durations_minutes, price_pence, price_by_duration, is_active"
            )
            .eq("business_slug", BUSINESS_SLUG)
            .eq("is_active", True)
            .execute()
        )
    except Exception as error:
        print(f"Could not load generic active services context: {error}")
        return ""

    lines = ["All active owner-edited services (authoritative):"]

    for row in response.data or []:
        name = str(row.get("service_name") or "").strip()
        mode = row.get("booking_mode")
        fixed = row.get("fixed_duration_minutes")
        allowed = row.get("allowed_durations_minutes") or []

        if not 2 <= len(name) <= 80:
            print("Skipped invalid active service row: invalid service_name")
            continue

        if mode == "choose_duration":
            durations = [
                int(value) for value in allowed
                if str(value).isdigit() and 5 <= int(value) <= 300
            ]
            if not durations:
                print(f"Skipped invalid active service row: {name}")
                continue
            duration_text = format_duration_options(set(durations))
            prices = row.get("price_by_duration") or {}
            price_text = ", ".join(
                f"{minutes} minutes {format_price_pence(prices.get(str(minutes)))}"
                for minutes in durations
                if prices.get(str(minutes)) not in [None, ""]
            )
        else:
            if fixed in [None, ""] or not 5 <= int(fixed) <= 300:
                print(f"Skipped invalid active service row: {name}")
                continue
            duration_text = format_minutes_as_duration(int(fixed))
            price_text = format_price_pence(row.get("price_pence")) or ""

        lines.append(
            f"- {name} [{row.get('category') or 'other'}]: "
            f"{duration_text}; {price_text}".rstrip("; ")
        )

    if len(lines) == 1:
        return (
            "No valid active owner-edited services are currently available. "
            "Do not invent or recommend a service; apologise and ask the "
            "customer to contact the business."
        )

    return "\n".join(lines)


def migrated_structured_service_catalogue() -> list[dict]:
    """
    Return service identities backed by live public.services rows.

    Fallback catalogues keep booking operational during a Supabase outage, but
    must not hide documents for services that may not have been migrated yet.
    """
    catalogue = []
    loaders = [
        ("massage", load_massage_services),
        ("vitamin_shots", load_vitamin_shot_services),
        ("ultrasound", load_ultrasound_services),
        ("body_treatments", load_body_treatment_services),
        ("microneedling", load_microneedling_services),
        ("facials", load_facial_services),
        ("dermal_fillers", load_dermal_filler_services),
    ]

    for category, loader in loaders:
        services, from_supabase = loader()

        if not from_supabase:
            continue

        for service in services:
            catalogue.append({
                "category": category,
                "service_name": service.get("service_name"),
                "aliases": list(service.get("aliases") or []),
            })

    return catalogue


def normalise_catalogue_document_text(value: str | None) -> str:
    """Normalise catalogue prose for conservative service-name matching."""
    cleaned = str(value or "").lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def catalogue_document_contains_phrase(
    normalised_content: str,
    phrase: str | None,
) -> bool:
    normalised_phrase = normalise_catalogue_document_text(phrase)

    if not normalised_phrase:
        return False

    return f" {normalised_phrase} " in f" {normalised_content} "


def is_legacy_catalogue_document(
    content: str,
    structured_catalogue: list[dict],
) -> bool:
    """
    Identify clear legacy price/duration catalogue rows duplicated by services.

    Category mentions alone are deliberately insufficient so FAQs, policies,
    and useful descriptions remain available to the responder.
    """
    normalised = normalise_catalogue_document_text(content)

    if not normalised or not structured_catalogue:
        return False

    has_price = bool(
        re.search(r"(?:£|Â£)\s*\d", content)
        or re.search(r"\b(?:pounds?|gbp)\b", normalised)
    )
    has_duration = bool(re.search(
        r"\b(?:\d+(?:\.\d+)?\s*)?"
        r"(?:minutes?|mins?|hours?|hrs?|sessions?)\b",
        normalised,
    ))
    migrated_categories = {
        str(service.get("category") or "")
        for service in structured_catalogue
    }

    if has_price and has_duration:
        for service in structured_catalogue:
            names = [
                service.get("service_name"),
                *(service.get("aliases") or []),
            ]

            if any(
                catalogue_document_contains_phrase(normalised, name)
                for name in names
            ):
                return True

    if has_price and "microneedling" in migrated_categories:
        microneedling_summary_markers = [
            "stretch marks",
            "face and neck",
            "selected area",
            "1 area",
            "one area",
        ]

        if (
            catalogue_document_contains_phrase(normalised, "microneedling")
            and sum(
                catalogue_document_contains_phrase(normalised, marker)
                for marker in microneedling_summary_markers
            ) >= 2
        ):
            return True

    if has_price and "dermal_fillers" in migrated_categories:
        filler_area_markers = [
            "lip filler",
            "marionette lines",
            "nasolabial folds",
        ]

        if (
            catalogue_document_contains_phrase(normalised, "dermal filler")
            and (
                sum(
                    catalogue_document_contains_phrase(normalised, marker)
                    for marker in filler_area_markers
                ) >= 2
                or (
                    catalogue_document_contains_phrase(normalised, "0.5 ml")
                    and catalogue_document_contains_phrase(normalised, "1 ml")
                )
            )
        ):
            return True

    return False


def load_business_documents_context() -> str:
    try:
        response = (
            supabase.table("documents")
            .select("content")
            .execute()
        )

        rows = []
        document_rows = response.data or []
        structured_catalogue = migrated_structured_service_catalogue()
        legacy_catalogue_filtered = 0

        for item in document_rows:
            content = str(item.get("content") or "").strip()

            if not content:
                continue

            # Structured services and live dashboard hours are authoritative.
            # Ignore old rows that could conflict with current configuration.
            if re.match(r"(?i)^\s*opening\s+hours\s*:", content):
                continue

            if is_legacy_catalogue_document(content, structured_catalogue):
                legacy_catalogue_filtered += 1
                continue

            rows.append(content)

        logger.info(
            "business_documents_loaded total=%d "
            "legacy_catalogue_filtered=%d retained=%d",
            len(document_rows),
            legacy_catalogue_filtered,
            len(rows),
        )

        return "\n".join(rows)

    except Exception as error:
        print(f"Could not load business documents: {error}")
        return ""


def compact_recent_history(history: list[ChatMessage], limit: int = 12) -> str:
    lines = []

    for message in history[-limit:]:
        role = (
            "Customer"
            if message.role in {"user", "customer"}
            else "Receptionist"
        )
        lines.append(f"{role}: {message.content}")

    return "\n".join(lines)


def merge_recent_history(
    client_history: list[ChatMessage],
    persisted_history: list[ChatMessage],
    limit: int = 20,
) -> list[ChatMessage]:
    """Prefer persisted messages while retaining unsaved client context."""
    merged = []
    seen = set()

    for message in [*persisted_history, *client_history]:
        role = "user" if message.role in {"user", "customer"} else "assistant"
        key = (role, str(message.content or "").strip())

        if not key[1] or key in seen:
            continue

        seen.add(key)
        merged.append(ChatMessage(role=role, content=key[1]))

    return merged[-limit:]


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


def customer_declines_phone_number(
    latest_message: str,
    previous_assistant_message: str = "",
) -> bool:
    cleaned = normalise_service_match_text(latest_message)

    if not cleaned:
        return False

    explicit_phrases = [
        "no phone",
        "no phone number",
        "without phone",
        "without a phone",
        "dont have phone",
        "dont have a phone",
        "dont want to give phone",
        "dont want to give my phone",
        "dont want to share phone",
        "dont want to share my phone",
        "don t want to give phone",
        "don t want to give my phone",
        "don t want to share phone",
        "don t want to share my phone",
        "do not want to give phone",
        "do not want to share phone",
        "dont want to give number",
        "dont want to give my number",
        "dont want to share number",
        "dont want to share my number",
        "don t want to give number",
        "don t want to give my number",
        "don t want to share number",
        "don t want to share my number",
        "do not want to give number",
        "do not want to share number",
        "rather not give phone",
        "rather not share phone",
        "rather not give number",
        "rather not share number",
        "prefer not to give phone",
        "prefer not to share phone",
        "prefer not to give number",
        "prefer not to share number",
        "i refuse phone",
        "i wont give phone",
        "i will not give phone",
    ]

    if any(phrase in cleaned for phrase in explicit_phrases):
        return True

    previous = normalise_service_match_text(previous_assistant_message)
    previous_asked_phone = (
        "phone" in previous
        or "contact number" in previous
        or "mobile number" in previous
    )

    if not previous_asked_phone:
        return False

    return cleaned in {
        "no",
        "no thanks",
        "no thank you",
        "rather not",
        "id rather not",
        "i would rather not",
        "prefer not",
        "i prefer not",
        "skip",
    }


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
- vague booking intent such as "I'd like to book" is not a treatment;
- include treatment only when the newest customer message names a configured
  service or clearly answers a pending service-variant question;
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


def grounded_treatment_from_latest_message(
    latest_message: str,
    existing_treatment: str | None,
) -> str | None:
    """
    Resolve treatment only from customer-visible evidence in the newest turn.

    Existing treatment is supplied solely to support clear short variant
    answers while a category is active. Extractor-proposed treatment text is
    deliberately excluded from this admission check.
    """
    metadata = resolve_service_metadata_from_message(
        latest_message
    )

    if metadata and metadata.get("service_name"):
        return str(metadata["service_name"])

    identity = resolve_latest_structured_service(
        latest_message,
        existing_treatment=existing_treatment,
    )

    if identity and identity.get("service_name"):
        return str(identity["service_name"])

    partial = resolve_latest_generic_or_partial_service(
        latest_message
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
    result["phone_declined"] = bool(result.get("phone_declined"))

    patch = extractor_result.get("state_patch") or {}

    # Date and time are calendar-critical fields. Resolve them deterministically
    # using the workflow detail Python most recently requested.
    schedule_resolution = resolve_schedule_input(
        latest_message,
        result,
    )
    parsed_date = schedule_resolution.get("preferred_date")
    parsed_time = schedule_resolution.get("preferred_time")
    extractor_result["schedule_clarification"] = schedule_resolution.get(
        "clarification"
    )
    extractor_result["schedule_input_detected"] = schedule_resolution.get(
        "schedule_input_detected"
    )
    parsed_phone = parse_phone_from_text(
        str(patch.get("phone") or latest_message)
    )
    phone_declined = customer_declines_phone_number(
        latest_message,
        latest_assistant_reply(history),
    )
    # Duration is a calendar-critical field. Accept it only when the newest
    # customer message explicitly supplies it; never trust an extractor guess.
    if parsed_date:
        result["preferred_date"] = parsed_date

    if parsed_time:
        result["preferred_time"] = parsed_time

    if phone_declined:
        result["phone"] = None
        result["phone_declined"] = True

    if parsed_phone and is_valid_phone_number(parsed_phone):
        result["phone"] = parsed_phone
        result["phone_declined"] = False
        print(
            "contact_phone_extracted "
            f"phone={parsed_phone} source_message={latest_message}"
        )

    name_candidate = str(patch.get("name") or "").strip()

    if not name_candidate and parsed_phone:
        name_candidate = parse_name_from_contact_message(
            latest_message,
            parsed_phone,
        ) or ""

    if (
        name_candidate
        and is_valid_customer_name(name_candidate)
        and not message_looks_like_schedule_input(name_candidate)
    ):
        result["name"] = name_candidate
        print(
            "contact_name_extracted "
            f"name={name_candidate} source_message={latest_message}"
        )

    previous_treatment = result.get("treatment")
    explicit_service_metadata = latest_explicit_service_metadata(
        latest_message,
        existing_treatment=previous_treatment,
    )
    grounded_treatment = (
        explicit_service_metadata.get("service_name")
        if explicit_service_metadata
        else grounded_treatment_from_latest_message(
            latest_message,
            previous_treatment,
        )
    )
    extractor_treatment = str(
        patch.get("treatment") or ""
    ).strip()

    if explicit_service_metadata and extractor_treatment:
        extracted_metadata = (
            authoritative_service_metadata(extractor_treatment)
            or structured_service_identity(extractor_treatment)
        )
        extracted_name = (
            extracted_metadata.get("service_name")
            if extracted_metadata
            else extractor_treatment
        )
        deterministic_name = explicit_service_metadata.get("service_name")

        if (
            extracted_name
            and deterministic_name
            and normalise_treatment_name(extracted_name)
            != normalise_treatment_name(deterministic_name)
        ):
            extractor_result["service_validation"] = {
                "accepted": False,
                "extracted_service": extracted_name,
                "deterministic_service": deterministic_name,
                "reason": "latest_explicit_customer_service_wins",
            }
            print(
                "booking_service_extractor_rejected "
                f"extracted={extracted_name} "
                f"deterministic={deterministic_name} "
                f"source_message={latest_message}"
            )

    if (
        not explicit_service_metadata
        and extractor_treatment
        and previous_treatment
    ):
        extracted_metadata = (
            authoritative_service_metadata(extractor_treatment)
            or structured_service_identity(extractor_treatment)
        )
        extractor_result["service_validation"] = {
            "accepted": False,
            "extracted_service": (
                extracted_metadata.get("service_name")
                if extracted_metadata
                else extractor_treatment
            ),
            "deterministic_service": previous_treatment,
            "reason": "slot_filling_message_preserved_active_service",
        }
        print(
            "booking_service_extractor_rejected "
            f"extracted={extractor_result['service_validation']['extracted_service']} "
            f"deterministic={previous_treatment} "
            f"source_message={latest_message}"
        )

    if grounded_treatment:
        if (
            previous_treatment
            and normalise_treatment_name(previous_treatment)
            != normalise_treatment_name(grounded_treatment)
        ):
            result = clear_service_specific_state(result)

        result["treatment"] = grounded_treatment

    duration_before_service_resolution = result.get("duration")

    # Service helpers resolve short replies such as "B12", "Relaxing", and
    # natural filler amount replies using the active category.
    variant_choice_pending = (
        needs_massage_variant(result)
        or needs_vitamin_shot_variant(result)
        or needs_ultrasound_variant(result)
        or needs_microneedling_variant(result)
        or needs_facial_variant(result)
        or needs_dermal_filler_variant(result)
    )
    allow_service_resolution = bool(
        not previous_treatment
        or grounded_treatment
        or variant_choice_pending
    )
    if allow_service_resolution:
        result = apply_structured_service_resolution(
            result,
            latest_message,
        )

    if explicit_service_metadata:
        result = lock_active_service_from_metadata(
            result,
            explicit_service_metadata,
            "latest_explicit_customer_intent",
        )
        extractor_result.setdefault("service_validation", {
            "accepted": True,
            "extracted_service": extractor_treatment or None,
            "deterministic_service": explicit_service_metadata.get(
                "service_name"
            ),
            "reason": "latest_explicit_customer_service",
        })
    elif result.get("treatment"):
        result = lock_active_service_from_treatment(
            result,
            result.get("treatment"),
            result.get("active_service_source") or "canonical_state",
            preserve_treatment=True,
        )

    # With no active treatment, service helpers may only create one when the
    # newest customer message itself grounded that service choice.
    if not previous_treatment and not grounded_treatment:
        result["treatment"] = None
        result["active_category"] = None
        result["active_service_id"] = None
        result["active_service_name"] = None
        result["active_service_source"] = None

    if (
        previous_treatment
        and result.get("treatment")
        and normalise_treatment_name(previous_treatment)
        != normalise_treatment_name(result.get("treatment"))
    ):
        if result.get("duration") == duration_before_service_resolution:
            result["duration"] = None
        result["verified_alternatives"] = []
        result["slot_status"] = "not_checked"

    duration_resolution = duration_reply_resolution(
        latest_message,
        previous_assistant_message=latest_assistant_reply(history),
        treatment=result.get("treatment"),
    )
    parsed_duration = duration_resolution.get("minutes")
    extractor_result["duration_resolution"] = duration_resolution

    if duration_resolution.get("ambiguous"):
        result["_duration_ambiguous"] = True

    fixed_duration_correction = None

    if parsed_duration is not None:
        candidate_duration = format_minutes_as_duration(
            parsed_duration
        )

        if duration_is_valid_for_treatment(
            result.get("treatment"),
            candidate_duration,
        ):
            result["duration"] = candidate_duration
        else:
            result["duration"] = None
    elif duration_resolution.get("explicit_invalid"):
        metadata = active_service_metadata_from_state(result)
        fixed_minutes = (metadata or {}).get("fixed_duration_minutes")
        if fixed_minutes not in [None, ""]:
            fixed_duration_correction = {
                "service_name": (metadata or {}).get("service_name") or result.get("treatment"),
                "fixed_minutes": int(fixed_minutes),
                "price_pence": (metadata or {}).get("price_pence"),
            }
        result["duration"] = None

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

    if fixed_duration_correction:
        result["duration"] = None
        result["_fixed_duration_correction"] = fixed_duration_correction
        extractor_result["fixed_duration_correction"] = fixed_duration_correction

    result["notes"] = merge_notes(
        result.get("notes"),
        patch.get("notes_append"),
    )

    # Recover previously supplied customer schedule details after a later
    # variant choice. Assistant suggestions are never used as saved facts.
    if not extractor_result.get("schedule_clarification"):
        result = recover_schedule_from_customer_history(
            result,
            compact_recent_history(history),
            latest_message,
        )
    if not result.get("_duration_ambiguous"):
        result = recover_duration_from_customer_history(
            result,
            compact_recent_history(history),
            latest_message,
        )

    result["_latest_source_message"] = latest_message
    result["_service_validation"] = extractor_result.get(
        "service_validation"
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

    if (
        (customer_asks_price_question(latest_message)
        or customer_asks_children_pets_policy(latest_message))
        and not customer_has_explicit_booking_intent(latest_message)
    ):
        return False

    if intent in BOOKING_INTENTS:
        return True

    if customer_has_explicit_booking_intent(latest_message):
        return True

    if message_has_service_with_valid_duration(latest_message):
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

    if state.get("_duration_ambiguous") or state.get("duration") in [None, ""]:
        return "duration"

    if state.get("preferred_date") in [None, ""]:
        return "preferred_date"

    if state.get("preferred_time") in [None, ""]:
        return "preferred_time"

    if active_verified_alternatives(state):
        return "verified_alternative"

    instagram_source = normalise_source(state.get("source")) == "instagram"
    phone_required = (
        not instagram_source
        and state.get("phone_declined") is not True
    )

    if (
        state.get("name") in [None, ""]
        and state.get("phone") in [None, ""]
    ):
        if state.get("phone_declined") is True:
            return "name"

        return (
            "name_and_optional_phone"
            if not phone_required
            else "name_and_phone"
        )

    if state.get("name") in [None, ""]:
        return "name"

    if state.get("phone") in [None, ""] and phone_required:
        return "phone"

    return "handoff"


def handoff_is_allowed(
    state: dict,
    calendar_result: dict | None,
) -> bool:
    """Return true only after a fully saved, backend-verified lead is ready."""
    treatment = state.get("treatment")
    canonical_treatment = canonical_treatment_from_text(
        treatment,
        str(treatment or ""),
        treatment,
    )
    duration_minutes = parse_duration_minutes(
        state.get("duration")
    )
    unresolved_variant = (
        needs_massage_variant(state)
        or needs_vitamin_shot_variant(state)
        or needs_ultrasound_variant(state)
        or needs_microneedling_variant(state)
        or needs_facial_variant(state)
        or needs_dermal_filler_variant(state)
    )
    calendar_status = str(
        (calendar_result or {}).get("status") or ""
    ).strip()
    slot_status = str(
        state.get("slot_status") or ""
    ).strip()

    return bool(
        state.get("_canonical_save_succeeded") is True
        and treatment not in [None, ""]
        and canonical_treatment not in [None, ""]
        and normalise_treatment_name(canonical_treatment)
        == normalise_treatment_name(treatment)
        and not unresolved_variant
        and duration_minutes
        and duration_minutes > 0
        and duration_is_valid_for_treatment(
            treatment,
            state.get("duration"),
        )
        and state.get("preferred_date") not in [None, ""]
        and state.get("preferred_time") not in [None, ""]
        and calendar_status in {"free", "provisional_free"}
        and slot_status in {"free", "provisional_free"}
        and not active_verified_alternatives(state)
        and state.get("name") not in [None, ""]
        and (
            normalise_source(state.get("source")) == "instagram"
            or state.get("phone_declined") is True
            or state.get("phone") not in [None, ""]
        )
        and compute_next_required_detail(state, True) == "handoff"
    )


def is_ready_for_owner_review(state: dict) -> bool:
    """Strict readiness gate for dashboard status and owner review."""
    treatment = state.get("treatment")
    canonical_treatment = canonical_treatment_from_text(
        treatment,
        str(treatment or ""),
        treatment,
    )
    duration_minutes = parse_duration_minutes(
        state.get("duration")
    )
    unresolved_variant = (
        needs_massage_variant(state)
        or needs_vitamin_shot_variant(state)
        or needs_ultrasound_variant(state)
        or needs_microneedling_variant(state)
        or needs_facial_variant(state)
        or needs_dermal_filler_variant(state)
    )
    slot_status = str(
        state.get("slot_status") or ""
    ).strip()

    return bool(
        state.get("_canonical_save_succeeded", True) is True
        and treatment not in [None, ""]
        and canonical_treatment not in [None, ""]
        and normalise_treatment_name(canonical_treatment)
        == normalise_treatment_name(treatment)
        and not unresolved_variant
        and duration_minutes
        and duration_minutes > 0
        and duration_is_valid_for_treatment(
            treatment,
            state.get("duration"),
        )
        and state.get("preferred_date") not in [None, ""]
        and state.get("preferred_time") not in [None, ""]
        and slot_status in {"free", "provisional_free"}
        and not active_verified_alternatives(state)
        and state.get("name") not in [None, ""]
        and (
            normalise_source(state.get("source")) == "instagram"
            or state.get("phone_declined") is True
            or state.get("phone") not in [None, ""]
        )
        and compute_next_required_detail(state, True) == "handoff"
    )


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
            metadata = active_service_metadata_from_state(state)
            options_with_prices = service_duration_price_options(metadata)
            has_duration_prices = bool(
                (metadata or {}).get("price_by_duration")
            )
            service_name = str(
                (metadata or {}).get("service_name")
                or state.get("treatment")
                or ""
            ).strip()

            if service_name and options_with_prices and has_duration_prices:
                return (
                    f"{service_name} is available as "
                    f"{options_with_prices}. Which would you prefer?"
                )

            return (
                "How long would you like the session for: "
                + format_duration_options(allowed)
                + "?"
            )

        return "How long would you like the session for?"

    if next_required_detail == "preferred_date":
        return "Which day would you prefer?"

    if next_required_detail == "preferred_time":
        if state.get("preferred_date") == datetime.now(BUSINESS_TIMEZONE).date().isoformat():
            return "What time works best for you today?"

        return (
            "What time would suit you on "
            + format_customer_date(
                state.get("preferred_date")
            )
            + "?"
        )

    if next_required_detail == "name_and_phone":
        return "Could I take your name and phone number, please?"

    if next_required_detail == "name_and_optional_phone":
        return (
            "Could I take your name, please? You can also share a phone "
            "number as an optional backup contact."
        )

    if next_required_detail == "name":
        return "Could I take your name, please?"

    if next_required_detail == "phone":
        return "Could I take your phone number, please?"

    if next_required_detail == "verified_alternative":
        return ""

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
    print(
        "booking_state_next_detail "
        f"active_service={result.get('active_service_name') or result.get('treatment')} "
        f"active_duration={result.get('duration')} "
        f"next_required_detail={next_detail}"
    )

    suggestions = (calendar_result or {}).get("suggestions") or []

    if suggestions:
        result["verified_alternatives"] = structured_verified_alternatives(
            suggestions,
            parse_duration_minutes(result.get("duration")),
        )
    elif (calendar_result or {}).get("status") == "free":
        result["verified_alternatives"] = []

    if not handoff_is_allowed(
        {**result, "_canonical_save_succeeded": True},
        calendar_result,
    ) and result["status"] == "booking_request_complete":
        result["status"] = "details_incomplete"

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

    duration_minutes = parse_duration_minutes(
        state.get("duration")
    )

    if (
        state.get("_duration_ambiguous")
        or
        state.get("treatment") in [None, ""]
        or state.get("preferred_date") in [None, ""]
        or state.get("preferred_time") in [None, ""]
        or not duration_minutes
        or duration_minutes <= 0
    ):
        return {
            "status": "not_checked",
            "message": (
                "A treatment, positive duration, date, and time are not all "
                "saved yet."
            ),
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
                + " automatically."
            )

        return "I could not verify the requested availability automatically."

    return ""


def option_numbers(value: str) -> set[str]:
    """Return customer-visible numeric option values from a reply or prompt."""
    return set(re.findall(
        r"(?<!\w)\d+(?:\.\d+)?",
        str(value or "").lower(),
    ))


def duration_option_numbers(value: str) -> set[str]:
    """Return numeric duration choices while ignoring customer-visible prices."""
    cleaned = str(value or "").lower()

    if not re.search(r"\b(?:minutes?|mins?|hours?|hrs?)\b", cleaned):
        return set()

    without_prices = re.sub(
        r"(?:Â£|£|\$)\s*\d+(?:\.\d+)?",
        "",
        cleaned,
    )
    return option_numbers(without_prices)


def service_variant_options_for_state(state: dict) -> list[dict]:
    """Return the relevant catalogue choices for an unresolved service."""
    treatment = state.get("treatment")
    category = normalise_treatment_name(treatment)
    loaders = {
        "massage": load_massage_services,
        "vitamin shot": load_vitamin_shot_services,
        "ultrasound": load_ultrasound_services,
        "microneedling": load_microneedling_services,
        "facial": load_facial_services,
        "dermal filler": load_dermal_filler_services,
    }

    if is_partial_dermal_filler_treatment(treatment):
        area = dermal_filler_partial_area(treatment)
        return dermal_filler_services_for_area(area)

    loader = loaders.get(category)
    return loader()[0] if loader else []


def response_already_contains_options(
    response: str,
    state: dict,
    next_required_detail: str | None,
    full_question: str,
) -> bool:
    """
    Detect whether the information body already presents the pending choices.

    The check uses structured catalogue identities first, then numeric option
    overlap for choices such as durations, prices, packages, and filler amounts.
    """
    cleaned_response = normalise_service_match_text(response)

    if not cleaned_response:
        return False

    if next_required_detail == "duration":
        allowed = {
            str(value)
            for value in (
                allowed_durations_for_treatment(
                    state.get("treatment")
                )
                or set()
            )
        }
        return len(duration_option_numbers(response) & allowed) >= 2

    if next_required_detail != "service_variant":
        return False

    matched_services = 0
    response_tokens = set(cleaned_response.split())
    generic_service_tokens = {
        "dermal",
        "facial",
        "filler",
        "fillers",
        "massage",
        "microneedling",
        "service",
        "session",
        "shot",
        "treatment",
        "ultrasound",
        "vitamin",
    }

    def label_is_present(label) -> bool:
        normalised_label = normalise_service_match_text(label)
        distinctive_tokens = (
            set(normalised_label.split())
            - generic_service_tokens
        )
        return bool(
            normalised_label
            and (
                normalised_label in cleaned_response
                or (
                    distinctive_tokens
                    and distinctive_tokens.issubset(response_tokens)
                )
            )
        )

    for service in service_variant_options_for_state(state):
        aliases = service.get("aliases") or []

        if isinstance(aliases, str):
            aliases = [aliases]

        labels = [
            service.get("service_name"),
            *aliases,
        ]

        if any(
            label_is_present(label)
            for label in labels
        ):
            matched_services += 1

    if matched_services >= 2:
        return True

    prompt_numbers = option_numbers(full_question)
    response_numbers = option_numbers(response)
    return len(prompt_numbers & response_numbers) >= 3


def build_short_next_question(
    state: dict,
    next_required_detail: str | None,
) -> str:
    """Ask for the missing choice without repeating its option list."""
    if next_required_detail == "duration":
        return "Which duration would you prefer?"

    if next_required_detail != "service_variant":
        return render_next_question(state, next_required_detail)

    treatment = state.get("treatment")
    category = normalise_treatment_name(treatment)

    if category == "massage":
        return "Which massage would you like?"

    if category == "facial":
        return "Which facial would you like?"

    if is_partial_dermal_filler_treatment(treatment):
        return "Which amount would you like?"

    if category == "dermal filler":
        return "Which area and amount would you like?"

    return "Which option would you like?"


def split_customer_sentences(response: str) -> list[str]:
    """Split customer-facing prose without treating decimal points as stops."""
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[!?])\s+|(?<=\.)\s+(?=[A-Z])|\n+",
            str(response or ""),
        )
        if sentence.strip()
    ]


def strip_questions_from_response(response: str) -> str:
    """Remove model-written questions before Python appends the one next step."""
    cleaned = re.sub(
        r"(?is)(?:\s*[,;]?\s*(?:and\s+)?)"
        r"(?:which|what|how|could|can|would|do|does|is|are)\b[^?]*\?",
        "",
        str(response or ""),
    )
    return " ".join(
        sentence
        for sentence in split_customer_sentences(cleaned)
        if not sentence.endswith("?")
    ).strip()


def build_responder_context(
    state: dict,
    extractor_result: dict,
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
- Mention a duration only when it exactly matches the duration in saved state.
- Never mention, paraphrase, invent, or calculate availability, calendar results, dates, weekdays, times, or slots.
- Never echo or acknowledge the customer's raw date or time wording. Python validates and writes all customer-facing schedule wording.
- Never mention handoff, contacting Veronika, confirmation, review, follow-up, or passing the request on. Python will append final handoff wording if eligible.
- Never imply the business will proceed or finish the request. Do not say "we'll take care of the rest", "we'll take care of the next steps", "we'll get everything set up", "we'll make sure everything is ready", "we'll arrange the remaining details", or similar.
- Never claim that an appointment is confirmed.
- Do not ask any question or request any missing booking detail.
- Answer a customer's side question briefly when possible.
- Do not repeat information unnecessarily.
- Return clean plain text only. Do not use Markdown emphasis, tables, or backticks.
- If the customer asks what services are offered, answer using the supplied business information and service catalogue.
- For menu or service-list replies, use concise mobile-friendly formatting:
  a short intro, plain category headings, bullet points, blank lines between
  categories, and an optional short closing question. Keep each bullet short
  and never put a long list of services or prices into one paragraph.
- If the customer asks about one category only, list only that category.

NEWEST CUSTOMER MESSAGE:
{latest_message}

HIDDEN EXTRACTOR RESULT:
{json.dumps(extractor_result, ensure_ascii=False, indent=2)}

CURRENT SAVED STATE AFTER VALIDATION:
{json.dumps(serialisable_saved_state(state), ensure_ascii=False, indent=2)}

RECENT CONVERSATION:
{compact_recent_history(history)}

AUTHORITATIVE STRUCTURED SERVICES:
{services_context}

BUSINESS INFORMATION:
{business_context}

Return only a question-free natural reply body. Python adds any calendar text
and workflow question afterward.
"""


def sanitise_natural_responder_body(reply: str) -> str:
    """
    Keep the model-written body free of calendar wording and workflow prompts.

    Python owns all calendar statements and workflow questions. Filter only
    explicit booking or availability claims so factual FAQ answers containing
    addresses, prices, durations, opening hours, or phone numbers survive.
    """
    cleaned = strip_model_booking_questions(
        strip_customer_markdown(
            normalise_reply_line_breaks(reply)
        )
    )
    safe_sentences = []

    for sentence in split_customer_sentences(cleaned):
        if sentence_contains_forbidden_booking_claim(sentence):
            continue

        safe_sentences.append(sentence)

    return collapse_repeated_reply_content(
        " ".join(safe_sentences)
    )


def strip_customer_markdown(reply: str) -> str:
    """Render model-written customer replies as clean, safe plain text."""
    cleaned = normalise_reply_line_breaks(reply)
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*+]\s+", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$", "", cleaned)
    cleaned = re.sub(
        r"(?m)^\s*\|(.+)\|\s*$",
        lambda match: " - ".join(
            cell.strip()
            for cell in match.group(1).split("|")
            if cell.strip()
        ),
        cleaned,
    )
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", cleaned)
    cleaned = re.sub(r"`([^`\n]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def sentence_contains_forbidden_booking_claim(sentence: str) -> bool:
    """Identify explicit model-owned booking claims, not generic numbers."""
    cleaned = str(sentence or "").strip()
    forbidden_patterns = [
        r"(?i)\b(?:i have|i've|we have|we've)\s+(?:booked|confirmed|reserved|scheduled)\b",
        r"(?i)\b(?:you(?:'re| are)|your (?:booking|appointment|visit))\s+(?:is\s+)?(?:booked|confirmed|reserved|scheduled)\b",
        r"(?i)\b(?:booking|appointment)\s+(?:is|has been)\s+(?:booked|confirmed|reserved|scheduled)\b",
        r"(?i)\b(?:i checked|i have checked|we checked|we have checked)\s+(?:the\s+)?(?:calendar|schedule|availability)\b",
        r"(?i)\b(?:i|we)\s+have\s+availability\b",
        r"(?i)\b(?:next available|alternative)\s+(?:slot|appointment|time)\b",
        r"(?i)\b(?:fit you in|found you a slot|offer you a slot)\b",
        r"(?i)^try\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s+instead\b",
        r"(?i)\b(?:around\s+)?(?:noon|midday|midnight|morning|afternoon|evening|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s+could\s+also\s+work\b",
        r"(?i)\b(?:slot|appointment|requested time|requested date)\b[^.!?]*\b(?:is|looks|appears|seems)\s+(?:available|free|busy|unavailable)\b",
        r"(?i)\b(?:available|free|busy|unavailable)\b[^.!?]*\b(?:slot|appointment|requested time|requested date)\b",
        r"(?i)\b(?:calendar|schedule)\b[^.!?]*\b(?:shows|says|confirms|has|is|looks|appears)\b",
        r"(?i)\b(?:i(?:'ll| will)? have the therapist|the therapist will)\s+check\b[^.!?]*\bconfirm\b",
        r"(?i)\b(?:we(?:['\u2019]ll| will)|i(?:['\u2019]ll| will))\s+pass\s+your\s+request\s+(?:on\s+)?to\s+veronika\b",
        r"(?i)\b(?:we(?:['\u2019]ll| will)|i(?:['\u2019]ll| will))\s+pass\s+your\s+request\s+on\b",
        r"(?i)\b(?:i(?:['\u2019]ll| will)|i have|i['\u2019]ve)\s+let\s+veronika\s+know\b",
        r"(?i)\b(?:we|i)(?:['\u2019]ll| will)\s+take care of\b",
        r"(?i)\bwe(?:['\u2019]ll| will)\s+get\b[^.!?]{0,60}\bset up\b",
        r"(?i)\bwe(?:['\u2019]ll| will)\s+make sure\b[^.!?]{0,60}\b(?:ready|set for you)\b",
        r"(?i)\b(?:we|i)(?:['\u2019]ll| will)\s+arrange (?:the )?(?:remaining )?details\b",
        r"(?i)\bi(?:['\u2019]ve| have) noted (?:your|the)\s+(?:request|preference|details?)[^.!?]{0,80}\b(?:and\s+)?(?:i|we)?\s*(?:will|['\u2019]ll)\s+take care of\b",
        r"(?i)\bi(?:['\u2019]ll| will)\s+look into suitable times and get back to you\b",
        r"(?i)\bwe(?:['\u2019]ll| will)\s+(?:be in touch|get back to you)\b",
        r"(?i)\bveronika will confirm(?: the appointment| it| your request)?(?: with you)?(?: shortly)?\b",
        r"(?i)\bveronika will (?:review|be in touch|contact you|get back to you|follow up)\b",
        r"(?i)\b(?:she(?:['\u2019]ll| will)|the therapist will) (?:be in touch|contact you|get back to you|follow up|confirm)\b",
        r"(?i)\bi(?:['\u2019]ve| have) noted your booking\b",
        r"(?i)\byour booking request has been sent\b",
        r"(?i)\blooking forward to welcoming you\b",
        r"(?i)\bif there(?:'s| is) anything else you(?:'d| would) like to know\b",
    ]
    return any(re.search(pattern, cleaned) for pattern in forbidden_patterns)


def reply_is_meaningful(reply: str) -> bool:
    cleaned = str(reply or "").strip()
    return bool(
        cleaned
        and re.search(r"[A-Za-z0-9£]", cleaned)
    )


def reply_is_hard_handoff_claim(reply: str) -> bool:
    cleaned = str(reply or "").lower()
    return any(
        phrase in cleaned
        for phrase in [
            "veronika",
            "therapist will",
            "let veronika know",
            "pass your request",
            "booking request has been sent",
            "noted your booking",
            "looking forward to welcoming",
            "will confirm",
            "contact you",
            "follow up",
        ]
    )


def reply_claims_service_recorded_without_state(
    reply: str,
    state: dict,
) -> bool:
    if state.get("treatment") not in [None, ""]:
        return False

    cleaned = str(reply or "").strip().lower()

    if not cleaned:
        return False

    if not re.search(
        r"\b(?:recorded|noted|saved|logged)\b|\bgreat choice\b",
        cleaned,
    ):
        return False

    return bool(
        resolve_service_metadata_from_message(reply)
        or structured_service_identity(reply)
    )


def customer_requests_information(message: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(message or "").strip().lower())

    if not cleaned:
        return False

    if "?" in cleaned:
        return True

    return bool(re.match(
        r"^(?:where|what|which|who|when|why|how|is|are|do|does|can|could|"
        r"tell me|please tell me)\b",
        cleaned,
    ))


def customer_asks_business_faq(message: str) -> bool:
    cleaned = normalise_service_match_text(message)
    faq_phrases = [
        "where",
        "based",
        "address",
        "location",
        "city centre",
        "how much",
        "price",
        "cost",
        "do you offer",
        "do you have",
        "opening",
        "open",
        "close",
        "phone",
        "number",
        "parking",
    ]
    return any(phrase in cleaned for phrase in faq_phrases)


def customer_asks_opening_hours(message: str) -> bool:
    cleaned = normalise_service_match_text(message)
    return any(
        phrase in cleaned
        for phrase in [
            "opening hours",
            "working hours",
            "business hours",
            "when are you open",
            "what days are you open",
            "what time do you open",
            "what time do you close",
        ]
    )


def format_business_clock_time(value) -> str:
    if not isinstance(value, tuple) or len(value) < 2:
        return ""

    hour, minute = value[:2]
    return f"{int(hour):02d}:{int(minute):02d}"


def build_business_hours_reply() -> str:
    """Render the dashboard timetable without asking the model to rewrite it."""
    hours, loaded = load_business_hours_with_status()

    if not loaded:
        return "I'll need to check the current opening hours for you."

    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    lines = ["Our current opening hours are:"]

    for day, label in enumerate(day_names):
        configured = hours.get(day)
        detail = "Closed"

        if configured:
            detail = (
                f"{format_business_clock_time(configured[0])}"
                f"-{format_business_clock_time(configured[1])}"
            )

        lines.append(f"• {label}: {detail}")

    return "\n".join(lines)


def authoritative_information_fallback(latest_message: str) -> str:
    """Answer only stable built-in business facts; otherwise admit the gap."""
    cleaned = normalise_service_match_text(latest_message)

    if any(
        phrase in cleaned
        for phrase in [
            "where",
            "based",
            "address",
            "location",
            "city centre",
        ]
    ):
        return f"We're based at {BUSINESS_ADDRESS}."

    return (
        "I'm sorry, I do not have that information available. "
        "Veronika can confirm it for you."
    )


def latest_assistant_reply(history: list[ChatMessage]) -> str:
    for message in reversed(history):
        if message.role not in {"user", "customer"}:
            return str(message.content or "")

    return ""


def newest_contact_acknowledgement(
    state: dict,
    extractor_result: dict,
    latest_message: str,
) -> str:
    """Acknowledge contact details supplied in the newest customer turn."""
    patch = extractor_result.get("state_patch") or {}
    parsed_phone = parse_phone_from_text(latest_message)
    supplied_phone = bool(
        parsed_phone
        and is_valid_phone_number(parsed_phone)
        and state.get("phone") not in [None, ""]
    )
    name_candidate = str(patch.get("name") or "").strip()
    if (
        not name_candidate
        and state.get("name") not in [None, ""]
        and str(latest_message or "").strip().lower()
        == str(state.get("name") or "").strip().lower()
    ):
        name_candidate = str(state.get("name") or "").strip()

    supplied_name = bool(
        name_candidate
        and is_valid_customer_name(name_candidate)
        and state.get("name") not in [None, ""]
    )

    if supplied_name and supplied_phone:
        return "Thanks, I've got your name and phone number."

    if supplied_name:
        return "Thanks, I've got your name."

    if supplied_phone:
        return "Thanks, I've got your phone number."

    return ""


def customer_explicitly_asks_availability(message: str) -> bool:
    cleaned = str(message or "").lower()
    return (
        asks_for_next_available_time(cleaned)
        or any(
            phrase in cleaned
            for phrase in [
                "availability",
                "available",
                "is it free",
                "is that free",
                "is the slot free",
            ]
        )
    )


def customer_asks_if_booked(message: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(message or "").strip().lower())
    return any(
        phrase in cleaned
        for phrase in [
            "am i booked",
            "is it booked",
            "is that booked",
            "is my appointment confirmed",
            "is it confirmed",
            "so i'm booked",
            "so i am booked",
        ]
    )


def generate_natural_reply_body(
    state: dict,
    extractor_result: dict,
    business_context: str,
    services_context: str,
    latest_message: str,
    history: list[ChatMessage],
) -> str:
    prompt = build_responder_context(
        state,
        extractor_result,
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
        reply = sanitise_natural_responder_body(reply)
        reply = sanitise_booking_claims(reply)
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
    del calendar_result
    safe_sentences = []

    for sentence in split_customer_sentences(reply):
        if sentence and not sentence_contains_forbidden_booking_claim(sentence):
            safe_sentences.append(sentence)

    return collapse_repeated_reply_content(" ".join(safe_sentences))


def strip_model_schedule_wording(reply: str) -> str:
    """Remove model-authored schedule wording after Python parsed the input."""
    schedule_pattern = re.compile(
        r"(?i)\b(?:"
        + "|".join(SCHEDULE_WEEKDAYS)
        + "|"
        + SCHEDULE_MONTH_NAMES
        + r"|today|tomorrow|noon|midday|midnight)\b"
        r"|\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b"
        r"|\b\d{1,2}(?:st|nd|rd|th)\b"
    )

    return collapse_repeated_reply_content(" ".join(
        sentence
        for sentence in split_customer_sentences(reply)
        if not schedule_pattern.search(sentence)
    ))


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
    full_next_question = render_next_question(
        state,
        next_required_detail,
    )
    next_question = full_next_question
    schedule_clarification = str(
        extractor_result.get("schedule_clarification") or ""
    ).strip()
    fixed_duration_correction = (
        extractor_result.get("fixed_duration_correction")
        or state.get("_fixed_duration_correction")
        or {}
    )

    if schedule_clarification:
        return schedule_clarification

    if fixed_duration_correction:
        service_name = fixed_duration_correction.get("service_name") or "That service"
        fixed_minutes = fixed_duration_correction.get("fixed_minutes")
        price = format_price_pence(
            fixed_duration_correction.get("price_pence")
        )
        details = (
            f"{fixed_minutes}-minute treatment"
            if fixed_minutes
            else "fixed-duration treatment"
        )
        price_text = f" for {price}" if price else ""
        return (
            f"{service_name} is only offered as a {details}{price_text}. "
            f"Would you like to go ahead with that?"
        )

    if state.get("_duration_ambiguous"):
        return render_next_question(state, "duration")

    booking_flow_active = (
        state.get("conversation_mode") == "booking_request"
        or next_required_detail is not None
    )
    handoff_ready = handoff_is_allowed(
        state,
        calendar_result,
    )
    policy_body = (
        build_children_pets_policy_reply(latest_message)
        if customer_asks_children_pets_policy(latest_message)
        else ""
    )
    specific_price_body = build_specific_price_reply(
        latest_message
    )
    structured_menu_body = build_customer_service_menu_reply(
        latest_message
    )
    structured_hours_body = (
        build_business_hours_reply()
        if customer_asks_opening_hours(latest_message)
        else ""
    )
    info_only_without_booking_intent = bool(
        (policy_body or specific_price_body)
        and not customer_has_explicit_booking_intent(latest_message)
    )

    # Suggested slots are fully deterministic because the model must never
    # invent alternatives or rewrite date labels. The newest useful message
    # still gets handled first so pending alternatives cannot swallow a side
    # question or newly supplied contact detail.
    if calendar_result.get("suggestions"):
        if info_only_without_booking_intent:
            return enforce_response_duration_consistency(
                policy_body or specific_price_body,
                state,
            )

        parts = []
        contact_acknowledgement = newest_contact_acknowledgement(
            state,
            extractor_result,
            latest_message,
        )

        if contact_acknowledgement:
            parts.append(contact_acknowledgement)

        if customer_requests_information(latest_message):
            information_body = structured_menu_body or structured_hours_body

            if not information_body:
                information_body = generate_natural_reply_body(
                    state,
                    extractor_result,
                    business_context,
                    services_context,
                    latest_message,
                    history,
                )
                information_body = sanitise_natural_responder_body(
                    information_body
                )
                information_body = strip_unverified_calendar_claims(
                    information_body,
                    calendar_result,
                )
                information_body = strip_questions_from_response(
                    information_body
                )

                if extractor_result.get("schedule_input_detected"):
                    information_body = strip_model_schedule_wording(
                        information_body
                    )

            if reply_is_meaningful(information_body):
                parts.append(information_body)

        alternative_reply = collapse_repeated_reply_content(
            build_calendar_alternative_reply(
                calendar_result
            )
        )
        parts.append(alternative_reply)

        return enforce_response_duration_consistency(
            collapse_repeated_reply_content(
                "\n\n".join(parts)
            ),
            state,
        )

    if info_only_without_booking_intent:
        next_question = ""

    if customer_asks_if_booked(latest_message):
        body = (
            "Your request has been noted, but the appointment is not "
            "confirmed yet."
        )
    elif extractor_result.get("verified_alternative_selected"):
        body = ""
    elif policy_body:
        body = policy_body
    elif specific_price_body:
        body = specific_price_body
    elif structured_menu_body:
        body = structured_menu_body
    elif structured_hours_body:
        body = structured_hours_body
    else:
        body = generate_natural_reply_body(
            state,
            extractor_result,
            business_context,
            services_context,
            latest_message,
            history,
        )
    calendar_text = safe_calendar_customer_text(
        state,
        calendar_result,
    )
    if handoff_ready:
        calendar_text = ""
    if info_only_without_booking_intent:
        calendar_text = ""
    previous_reply = latest_assistant_reply(history)

    if (
        calendar_text
        and calendar_text in previous_reply
        and not customer_explicitly_asks_availability(latest_message)
    ):
        calendar_text = ""

    parts = []

    body_before_sanitise = str(body or "")
    had_body_before_sanitise = bool(body_before_sanitise.strip())
    body_is_structured_menu = bool(
        structured_menu_body
        and body == structured_menu_body
    )
    body_is_structured_policy = bool(
        policy_body
        and body == policy_body
    )
    body_is_structured_price = bool(
        specific_price_body
        and body == specific_price_body
    )
    body_is_structured_hours = bool(
        structured_hours_body
        and body == structured_hours_body
    )

    if (
        not body_is_structured_menu
        and not body_is_structured_policy
        and not body_is_structured_price
        and not body_is_structured_hours
    ):
        body = sanitise_natural_responder_body(body)
        body = strip_unverified_calendar_claims(
            body,
            calendar_result,
        )

    if extractor_result.get("schedule_input_detected"):
        body = strip_model_schedule_wording(body)

    if reply_claims_service_recorded_without_state(body, state):
        body = ""

    if next_question and not body_is_structured_menu:
        body = strip_questions_from_response(body)

        if response_already_contains_options(
            body,
            state,
            next_required_detail,
            full_next_question,
        ):
            next_question = build_short_next_question(
                state,
                next_required_detail,
            )

    if not reply_is_meaningful(body):
        body = ""

    if (
        not body
        and had_body_before_sanitise
        and next_question
        and not reply_is_hard_handoff_claim(body_before_sanitise)
    ):
        body = "I've noted that."

    if (
        not body
        and customer_requests_information(latest_message)
        and (
            not next_question
            or customer_asks_business_faq(latest_message)
        )
    ):
        body = authoritative_information_fallback(
            latest_message
        )

    if body.strip().lower() == "thanks.":
        if booking_flow_active:
            body = ""
        elif customer_requests_information(latest_message):
            body = authoritative_information_fallback(
                latest_message
            )

    if body:
        parts.append(body)

    if calendar_text and calendar_text.lower() not in body.lower():
        parts.append(calendar_text)

    if handoff_ready:
        handoff = (
            f"Thank you, {state.get('name')}. Your requested slot currently "
            "appears free. Veronika will confirm the appointment with you "
            "shortly."
        )

        if (
            handoff.lower() not in "\n\n".join(parts).lower()
            and handoff.lower() not in previous_reply.lower()
        ):
            parts.append(handoff)

    elif next_question:
        combined = "\n\n".join(parts).lower()

        if next_question.lower() not in combined:
            parts.append(next_question)

    if not parts:
        if next_question:
            parts.append(next_question)
        elif booking_flow_active:
            parts.append("Your request details are saved.")
        elif customer_requests_information(latest_message):
            parts.append(authoritative_information_fallback(latest_message))
        else:
            parts.append("Thanks.")

    return enforce_response_duration_consistency(
        strip_customer_markdown(
            collapse_repeated_reply_content(
                "\n\n".join(parts)
            )
        ),
        state,
    )


def enforce_response_duration_consistency(reply: str, state: dict) -> str:
    """Remove model wording that contradicts canonical duration-linked state."""
    canonical_minutes = parse_duration_minutes(state.get("duration"))

    if not canonical_minutes:
        return reply

    metadata = (
        authoritative_service_metadata(state.get("treatment"))
        or structured_service_identity(state.get("treatment"))
    )
    canonical_price_pence = simple_request_price_pence(
        metadata,
        canonical_minutes,
    )
    safe_sentences = []
    conflict_found = False

    for sentence in split_customer_sentences(reply):
        mentioned = duration_candidates_from_text(sentence)
        mentioned_prices = response_price_pence_values(sentence)

        if (
            len(mentioned) == 1
            and canonical_minutes not in mentioned
        ) or (
            canonical_price_pence is not None
            and len(mentioned) <= 1
            and mentioned_prices
            and mentioned_prices != {canonical_price_pence}
        ):
            conflict_found = True
            continue

        safe_sentences.append(sentence)

    if not conflict_found:
        return reply

    safe_sentences.insert(
        0,
        f"I've noted the {format_minutes_as_duration(canonical_minutes)} duration.",
    )
    return collapse_repeated_reply_content(" ".join(safe_sentences))


def response_price_pence_values(value: str) -> set[int]:
    """Extract explicit customer-visible pound prices from response text."""
    prices = set()

    for match in re.finditer(
        r"(?:£|Â£)\s*(\d+(?:\.\d{1,2})?)"
        r"|\b(\d+(?:\.\d{1,2})?)\s+pounds?\b",
        str(value or ""),
        flags=re.IGNORECASE,
    ):
        amount = match.group(1) or match.group(2)
        prices.add(round(float(amount) * 100))

    return prices


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

    prices = metadata.get("price_by_duration") or {}

    if duration_minutes is not None:
        value = prices.get(
            str(duration_minutes)
        )

        if value not in [None, ""]:
            return int(value)

    if metadata.get("booking_mode") == "choose_duration":
        return None

    fixed_price = metadata.get("price_pence")

    if fixed_price not in [None, ""]:
        return int(fixed_price)

    return None


def load_active_simple_request(
    session_id: str,
) -> dict | None:
    try:
        query = (
            supabase.table("booking_requests")
            .select("*")
            .eq("session_id", session_id)
        )

        if hasattr(query, "in_"):
            query = query.in_(
                "request_status",
                ["draft", "ready_for_review"],
            )

        response = query.order("request_number").limit(1).execute()

        if response.data:
            return response.data[0]

    except Exception:
        print("projection_draft_load_failed")

    return None


def load_simple_draft_request(
    session_id: str,
) -> dict | None:
    """Backward-compatible alias for older projection tests."""
    return load_active_simple_request(session_id)


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

    except Exception:
        print("projection_request_number_failed")

    return 1


def sync_simple_single_request_projection(
    session_id: str,
    state: dict,
    booking_flow_active: bool,
) -> bool:
    """
    Mirror the canonical single-treatment lead into the relational dashboard.

    This is intentionally a projection only. It does not implement carts,
    separate appointment drafts, or multi-booking workflow logic.
    """
    if not booking_flow_active:
        print("projection_skipped_inactive_booking_flow")
        return False

    if state.get("_duration_ambiguous"):
        print("projection_skipped_ambiguous_duration")
        return False

    metadata = active_service_metadata_from_state(state)

    if not metadata:
        print("projection_skipped_missing_metadata")
        return False

    locked_service = state.get("active_service_name") or state.get("treatment")
    if (
        locked_service
        and metadata.get("service_name")
        and normalise_treatment_name(locked_service)
        != normalise_treatment_name(metadata.get("service_name"))
    ):
        print(
            "projection_skipped_service_lock_conflict "
            f"locked={locked_service} "
            f"metadata={metadata.get('service_name')}"
        )
        return False

    duration_minutes = parse_duration_minutes(
        state.get("duration")
    )

    if duration_minutes is None:
        print("projection_skipped_missing_duration")
        return False

    price_pence = simple_request_price_pence(
        metadata,
        duration_minutes,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    request_row = load_simple_draft_request(
        session_id
    )
    request_status = (
        "ready_for_review"
        if is_ready_for_owner_review(state)
        else "draft"
    )

    request_payload = {
        "session_id": session_id,
        "request_status": request_status,
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
            response = (
                supabase.table("booking_requests")
                .update(request_payload)
                .eq("id", request_id)
                .execute()
            )

            if not response.data:
                print("projection_request_update_failed")
                return False
        else:
            request_payload["request_number"] = (
                next_simple_request_number(
                    session_id
                )
            )
            try:
                response = (
                    supabase.table("booking_requests")
                    .insert(request_payload)
                    .execute()
                )
            except Exception:
                print("projection_request_insert_failed")
                return False

            if not response.data:
                print("projection_request_insert_failed")
                return False

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
            .select("id, service_name")
            .eq("booking_request_id", request_id)
            .eq("item_order", 1)
            .limit(1)
            .execute()
        )

        try:
            previous_service = None
            if existing_item_response.data:
                previous_service = existing_item_response.data[0].get(
                    "service_name"
                )

            if (
                previous_service
                and item_payload.get("service_name")
                and normalise_treatment_name(previous_service)
                != normalise_treatment_name(item_payload.get("service_name"))
            ):
                print(
                    "booking_item_service_changed "
                    f"session_id={session_id} "
                    f"previous_service={previous_service} "
                    f"new_service={item_payload.get('service_name')} "
                    f"source_message={state.get('_latest_source_message')} "
                    f"extraction_output={state.get('_service_validation')} "
                    f"deterministic_match={locked_service} "
                    "reason=active_service_lock"
                )

            if existing_item_response.data:
                item_response = (
                    supabase.table("booking_items")
                    .update(item_payload)
                    .eq(
                        "id",
                        existing_item_response.data[0]["id"],
                    )
                    .execute()
                )
            else:
                item_response = (
                    supabase.table("booking_items")
                    .insert(item_payload)
                    .execute()
                )

            if not item_response.data:
                print("projection_item_update_failed")
                return False
        except Exception:
            print("projection_item_update_failed")
            return False

        try:
            link_response = (
                supabase.table("conversations")
                .update({
                    "active_booking_request_id": request_id,
                })
                .eq("session_id", session_id)
                .execute()
            )

            if not link_response.data:
                print("projection_conversation_link_failed")
                return False
        except Exception:
            print("projection_conversation_link_failed")
            return False

        return True

    except Exception:
        print("projection_request_update_failed")
        return False


CANONICAL_SAVE_FAILURE_REPLY = (
    "Sorry, I could not safely save your request just now. "
    "Please try again in a moment."
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

    existing_conversation = get_existing_conversation(
        request.session_id
    )

    if not auto_reply_enabled():
        paused_state = {
            **prepare_conversation_for_resume(existing_conversation),
            "source": request_source,
        }
        save_conversation_overview(
            session_id=request.session_id,
            lead_data=paused_state,
            source=request_source,
        )
        return {
            "reply": (
                "Thanks - your message has been saved for Veronika."
                if request_source == "website"
                else ""
            ),
            "auto_reply_enabled": False,
        }

    if (
        customer_explicitly_cancels_request(request.message)
        and cancel_active_booking_request(existing_conversation)
    ):
        reply = "Your appointment request has been cancelled."
        save_message(
            session_id=request.session_id,
            role="assistant",
            content=reply,
            source=request_source,
        )
        return {"reply": reply}

    resume_was_reset = (
        active_booking_request_status(existing_conversation)
        in INACTIVE_BOOKING_REQUEST_STATUSES
    )
    current_state = prepare_conversation_for_resume(existing_conversation)
    current_state = {
        **current_state,
        "source": request_source,
    }
    effective_history = (
        []
        if resume_was_reset
        else merge_recent_history(
            request.history,
            load_recent_messages(request.session_id),
        )
    )
    services_context = build_authoritative_services_context()
    business_context = load_business_documents_context()

    extractor_result = extract_hidden_state_patch(
        current_state=current_state,
        latest_message=request.message,
        history=effective_history,
        services_context=services_context,
    )
    alternative_resolution = resolve_verified_alternative_reply(
        current_state,
        request.message,
    )

    if alternative_resolution.get("status") == "selected":
        extractor_result = {
            **extractor_result,
            "verified_alternative_selected": True,
        }

    merged_state = apply_validated_state_patch(
        existing_state=current_state,
        extractor_result=extractor_result,
        latest_message=request.message,
        history=effective_history,
    )
    merged_state = apply_verified_alternative_resolution(
        merged_state,
        current_state,
        alternative_resolution,
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

    if not save_conversation_overview(
        session_id=request.session_id,
        lead_data=merged_state,
        source=request_source,
    ):
        return {"reply": CANONICAL_SAVE_FAILURE_REPLY}

    if alternative_resolution.get("status") in {"ambiguous", "no_match"}:
        calendar_result = {
            "status": (
                "alternative_ambiguous"
                if alternative_resolution.get("status") == "ambiguous"
                else "alternative_no_match"
            ),
            "suggestions": [
                option["label"]
                for option in active_verified_alternatives(merged_state)
            ],
        }
    elif active_verified_alternatives(merged_state):
        calendar_result = {
            "status": "pending_alternatives",
            "suggestions": [
                option["label"]
                for option in active_verified_alternatives(merged_state)
            ],
        }
    else:
        calendar_result = verified_calendar_result_for_state(
            merged_state,
            booking_flow_active,
            request.message,
        )

    merged_state = apply_canonical_controller_state(
        merged_state,
        booking_flow_active,
        calendar_result,
    )

    if not save_conversation_overview(
        session_id=request.session_id,
        lead_data=merged_state,
        source=request_source,
    ):
        return {"reply": CANONICAL_SAVE_FAILURE_REPLY}

    merged_state["_canonical_save_succeeded"] = True

    try:
        projection_saved = sync_simple_single_request_projection(
            session_id=request.session_id,
            state=merged_state,
            booking_flow_active=booking_flow_active,
        )
        if not projection_saved and booking_flow_active:
            print(
                "booking_projection_non_blocking_failure "
                f"session_id={request.session_id} reason=projection_returned_false"
            )
    except Exception as projection_error:
        print(
            "booking_projection_non_blocking_failure "
            f"session_id={request.session_id} "
            f"reason={type(projection_error).__name__}"
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
        history=effective_history,
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
