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
        if re.search(rf"\bnext\s+week\s+{weekday_name}\b", lower_message):
            start_of_next_week = today + timedelta(days=(7 - today.weekday()))
            return (start_of_next_week + timedelta(days=weekday_number)).isoformat()

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

        if 10 <= len(digits) <= 15:
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


SUPPORTED_MASSAGE_DURATIONS = {
    "relaxing massage": {30, 60, 90, 120},
    "swedish massage": {30, 60, 90, 120},
    "deep tissue massage": {30, 60, 90, 120},
    "hot stone massage": {60, 90, 120},
}


def normalise_treatment_name(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def allowed_durations_for_treatment(
    treatment: str | None,
) -> set[int] | None:
    cleaned = normalise_treatment_name(treatment)

    for treatment_name, durations in SUPPORTED_MASSAGE_DURATIONS.items():
        if treatment_name in cleaned or cleaned in treatment_name:
            return durations

    return None


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
        "name": "B12 Vitamin Shot",
        "duration": "15 minutes",
        "aliases": ["b12 vitamin shot", "b12 shot"],
    },
    {
        "name": "Vitamin C Shot",
        "duration": "15 minutes",
        "aliases": ["vitamin c shot", "c vitamin shot"],
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


def validate_slot_against_working_hours(
    start_time: datetime,
    end_time: datetime,
    duration_minutes: int | None,
) -> dict | None:
    opening_window = working_hours_for_date(start_time.date())

    if not opening_window:
        return {
            "status": "outside_hours",
            "message": (
                f"{format_customer_date(start_time.date().isoformat())} is outside working hours. "
                "Ask the customer for another preferred time."
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

    explicit_date = parse_relative_date_from_text(latest_message)
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

    questions = {
        "treatment": "Which treatment would you like?",
        "name": "What's your name?",
        "phone": "What's your phone number?",
        "preferred_date": "Which day would you prefer?",
        "preferred_time": "What time would you prefer?",
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


def ensure_next_booking_question(
    reply: str,
    missing_fields: list[str],
    lead_data: dict,
) -> str:
    """
    When a customer asks a side question while booking, answer it and still
    continue the lead flow. Example: "10:30, is there parking?"
    """
    if not missing_fields:
        return reply

    next_field = missing_fields[0]

    if reply_already_asks_for_field(reply, next_field):
        return reply

    return reply.rstrip() + "\n\n" + canonical_missing_question(
        next_field,
        lead_data,
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

    next_field = missing_fields[0]

    questions = {
        "treatment": "Ask only which treatment they would like.",
        "duration": (
            "Ask only about the valid treatment option or duration. "
            "Never suggest massage durations for a non-massage treatment. "
            + canonical_missing_question("duration", lead_data)
        ),
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


def load_business_hours() -> dict:
    """
    Read editable working hours from Supabase.
    Fall back to safe defaults if the table cannot be read.
    """
    hours = dict(DEFAULT_BUSINESS_HOURS)

    try:
        response = (
            supabase.table("business_hours")
            .select("day_of_week, open_time, close_time, is_closed")
            .eq("business_slug", BUSINESS_SLUG)
            .order("day_of_week")
            .execute()
        )

        if not response.data:
            return hours

        for row in response.data:
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
        print(f"Could not load business hours: {error}")

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

    working_hours_error = validate_slot_against_working_hours(
        start_time,
        end_time,
        duration_minutes,
    )

    if working_hours_error:
        return working_hours_error

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

            session_id = f"instagram:{sender_id}"
            history = load_recent_messages(session_id)

            request = ChatRequest(
                session_id=session_id,
                message=message_text,
                history=history,
                source="instagram",
            )

            try:
                response = asyncio.run(
                    chat(request, BackgroundTasks())
                )

                reply = response.get("reply")

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
        existing_before_processing.get("status") == "booking_request_complete"
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
            for msg in reversed(request.history)
            if msg.role == "assistant"
        ),
        "",
    )

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

    if calendar_result.get("status") in {
        "outside_hours",
        "past",
        "busy",
    }:
        merged_lead["preferred_time"] = None
        merged_lead["status"] = "details_incomplete"
        merged_lead["notification_sent_at"] = None

        save_conversation_overview(
            session_id=request.session_id,
            lead_data=merged_lead,
            source=request_source,
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

    next_question_hint = build_next_question_hint(
        missing_fields,
        merged_lead,
    )
    customer_date_label = format_customer_date(
        merged_lead.get("preferred_date")
    )

    if is_simple_acknowledgement(request.message) and not missing_fields:
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
- If duration is the next missing detail, use NEXT QUESTION TARGET. Never offer massage durations for a non-massage treatment.
- If a treatment has one fixed duration listed under CONFIRMED CUSTOMER DETAILS, do not ask the customer to choose a session length.
- Do not repeat hello twice.
- Avoid filler such as "I've noted that down", "just to confirm", "we offer", "to proceed", and "the requested time currently looks available" unless genuinely needed.
- Use the conversation history to understand short replies.
- Do not ask the customer to repeat information they already gave.
- Never ask again for any detail listed under CONFIRMED CUSTOMER DETAILS.
- Ask naturally for only the missing details. Ask for no more than two missing details in one reply.
- Follow the CALENDAR AVAILABILITY RESULT exactly.
- If CALENDAR AVAILABILITY RESULT says Visibility: silent, do not mention calendar availability.
- When speaking to the customer, use CUSTOMER-FACING DATE LABEL. Say "today" or "tomorrow" where appropriate. Never show an ISO date such as 2026-06-01 to the customer unless they explicitly ask for the exact date.
- If a requested slot is busy, outside working hours, or in the past, say it once briefly and ask for another preferred time. Never treat that time as accepted.
- If a requested slot appears free, say it once briefly and clearly state that the therapist still needs to confirm.
- Never accept an unsupported session duration. Massage durations must match the allowed options.
- Never suggest or invent a specific date or time unless it comes from the customer's message or from calendar alternatives.
- If the calendar cannot be checked, do not guess. Say the therapist will check and confirm.
- Never reveal private calendar event details.
- Never say a booking is confirmed, reserved, or completed.
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
    reply = ensure_next_booking_question(
        reply,
        missing_fields,
        merged_lead,
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
