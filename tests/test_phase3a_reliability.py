import asyncio
from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest.mock import patch
from urllib import error as urllib_error

from test_smoke import load_chatbot_module


class Response:
    def __init__(self, data):
        self.data = data


class UrlResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class InMemorySupabase:
    def __init__(self):
        self.rows = {
            "booking_requests": [],
            "booking_items": [],
            "conversations": [{"session_id": "session-1"}],
        }
        self.next_ids = {"booking_requests": 1, "booking_items": 1}

    def table(self, name):
        return Query(self, name)


class Query:
    def __init__(self, database, table):
        self.database = database
        self.table = table
        self.filters = []
        self.operation = "select"
        self.payload = None
        self.limit_count = None
        self.descending = False
        self.order_field = None

    def select(self, fields):
        self.operation = "select"
        return self

    def eq(self, field, value):
        self.filters.append((field, value))
        return self

    def order(self, field, desc=False):
        self.order_field = field
        self.descending = desc
        return self

    def limit(self, count):
        self.limit_count = count
        return self

    def insert(self, payload):
        self.operation = "insert"
        self.payload = dict(payload)
        return self

    def update(self, payload):
        self.operation = "update"
        self.payload = dict(payload)
        return self

    def execute(self):
        rows = self.database.rows[self.table]

        if self.operation == "insert":
            row = dict(self.payload)
            row.setdefault("id", self.database.next_ids[self.table])
            self.database.next_ids[self.table] += 1
            rows.append(row)
            return Response([dict(row)])

        matches = [
            row
            for row in rows
            if all(row.get(field) == value for field, value in self.filters)
        ]

        if self.operation == "update":
            for row in matches:
                row.update(self.payload)
            return Response([dict(row) for row in matches])

        if self.order_field:
            matches.sort(
                key=lambda row: row.get(self.order_field) or 0,
                reverse=self.descending,
            )

        if self.limit_count is not None:
            matches = matches[:self.limit_count]

        return Response([dict(row) for row in matches])


class CanonicalStatePreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def apply(self, existing, message, history=None, patch_result=None):
        extractor = self.chatbot.empty_extractor_result()
        extractor["state_patch"] = patch_result or {}
        return self.chatbot.apply_validated_state_patch(
            existing,
            extractor,
            message,
            history or [],
        )

    def test_website_customer_history_is_recognised(self):
        history = [
            self.chatbot.ChatMessage(
                role="customer",
                content="lip filler tomorrow at 9 am",
            )
        ]

        compact = self.chatbot.compact_recent_history(history)

        self.assertIn("Customer: lip filler tomorrow at 9 am", compact)

    def test_persisted_history_recovers_schedule_after_variant_choice(self):
        tomorrow = (
            datetime.now(self.chatbot.BUSINESS_TIMEZONE).date()
            + timedelta(days=1)
        ).isoformat()
        history = [
            self.chatbot.ChatMessage(
                role="user",
                content="can i get a lip filler tomorrow at 9 am",
            ),
            self.chatbot.ChatMessage(
                role="assistant",
                content="For lip filler, would you like 0.5 ml or 1 ml?",
            ),
            self.chatbot.ChatMessage(role="user", content="1 ml"),
        ]
        existing = {"treatment": "Lip Filler"}

        def resolve_variant(state, message):
            result = dict(state)
            result["treatment"] = "Lip Filler 1 ml"
            result["duration"] = "1 hour"
            return result

        with patch.object(
            self.chatbot,
            "apply_structured_service_resolution",
            side_effect=resolve_variant,
        ):
            with patch.object(
                self.chatbot,
                "hydrate_configured_service_defaults",
                side_effect=lambda state: state,
            ):
                result = self.apply(existing, "1 ml", history)

        self.assertEqual(result["treatment"], "Lip Filler 1 ml")
        self.assertEqual(result["duration"], "1 hour")
        self.assertEqual(result["preferred_date"], tomorrow)
        self.assertEqual(result["preferred_time"], "09:00")

    def test_omitted_fields_and_single_schedule_changes_are_preserved(self):
        existing = {
            "treatment": "Relaxing Massage",
            "duration": "30 minutes",
            "preferred_date": "2099-01-01",
            "preferred_time": "15:00",
            "name": "Ilko",
            "phone": "0782318283",
        }

        with patch.object(
            self.chatbot,
            "apply_structured_service_resolution",
            side_effect=lambda state, message: state,
        ):
            with patch.object(
                self.chatbot,
                "hydrate_configured_service_defaults",
                side_effect=lambda state: state,
            ):
                changed_time = self.apply(existing, "18:00")
                changed_date = self.apply(existing, "2099-01-02")
                duration_reply = self.apply(existing, "30 minutes")

        self.assertEqual(changed_time["preferred_date"], "2099-01-01")
        self.assertEqual(changed_time["preferred_time"], "18:00")
        self.assertEqual(changed_date["preferred_date"], "2099-01-02")
        self.assertEqual(changed_date["preferred_time"], "15:00")

        for field, value in existing.items():
            self.assertEqual(duration_reply[field], value)

    def test_date_before_time_and_time_before_date_are_preserved(self):
        with patch.object(
            self.chatbot,
            "apply_structured_service_resolution",
            side_effect=lambda state, message: state,
        ):
            with patch.object(
                self.chatbot,
                "hydrate_configured_service_defaults",
                side_effect=lambda state: state,
            ):
                date_then_time = self.apply(
                    {
                        "treatment": "Relaxing Massage",
                        "duration": "30 minutes",
                        "preferred_date": "2099-01-01",
                    },
                    "15:00",
                )
                time_then_date = self.apply(
                    {
                        "treatment": "Relaxing Massage",
                        "duration": "30 minutes",
                        "preferred_time": "15:00",
                    },
                    "2099-01-01",
                )

        self.assertEqual(date_then_time["preferred_date"], "2099-01-01")
        self.assertEqual(date_then_time["preferred_time"], "15:00")
        self.assertEqual(time_then_date["preferred_date"], "2099-01-01")
        self.assertEqual(time_then_date["preferred_time"], "15:00")

    def test_merge_recent_history_prefers_persisted_customer_context(self):
        persisted = [
            self.chatbot.ChatMessage(role="user", content="tomorrow at 9 am"),
            self.chatbot.ChatMessage(role="assistant", content="Which option?"),
        ]
        client = [
            self.chatbot.ChatMessage(role="customer", content="tomorrow at 9 am"),
            self.chatbot.ChatMessage(role="customer", content="1 ml"),
        ]

        merged = self.chatbot.merge_recent_history(client, persisted)

        self.assertEqual(
            [(item.role, item.content) for item in merged],
            [
                ("user", "tomorrow at 9 am"),
                ("assistant", "Which option?"),
                ("user", "1 ml"),
            ],
        )

    def test_unavailable_calendar_result_does_not_clear_requested_slot(self):
        state = {
            "treatment": "Relaxing Massage",
            "duration": "30 minutes",
            "preferred_date": "2099-01-01",
            "preferred_time": "15:00",
        }

        controlled = self.chatbot.apply_canonical_controller_state(
            state,
            booking_flow_active=True,
            calendar_result={"status": "busy"},
        )

        self.assertEqual(controlled["preferred_date"], "2099-01-01")
        self.assertEqual(controlled["preferred_time"], "15:00")


class ProjectionReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def setUp(self):
        self.database = InMemorySupabase()
        self.metadata = {
            "service_id": 7,
            "service_name": "Lip Filler 0.5 ml",
            "category": "dermal_fillers",
            "price_pence": 7000,
            "price_by_duration": None,
        }

    def sync(self, state):
        with patch.object(self.chatbot, "supabase", self.database):
            with patch.object(
                self.chatbot,
                "authoritative_service_metadata",
                return_value=self.metadata,
            ):
                return self.chatbot.sync_simple_single_request_projection(
                    "session-1",
                    state,
                    True,
                )

    def test_projection_creates_then_updates_one_request_and_item(self):
        state = {
            "treatment": "Lip Filler 0.5 ml",
            "duration": "45 minutes",
            "preferred_date": None,
            "preferred_time": None,
            "slot_status": "not_checked",
            "next_required_detail": "preferred_date",
        }
        self.assertTrue(self.sync(state))

        state.update({
            "preferred_date": "2099-01-01",
            "preferred_time": "18:00",
            "slot_status": "provisional_free",
            "next_required_detail": "handoff",
            "name": "Ilko",
            "phone": "0812838123",
        })
        self.metadata["service_name"] = "Lip Filler 1 ml"
        self.metadata["price_pence"] = 10000
        state["treatment"] = "Lip Filler 1 ml"
        self.assertTrue(self.sync(state))

        self.assertEqual(len(self.database.rows["booking_requests"]), 1)
        self.assertEqual(len(self.database.rows["booking_items"]), 1)
        request = self.database.rows["booking_requests"][0]
        item = self.database.rows["booking_items"][0]
        conversation = self.database.rows["conversations"][0]
        self.assertEqual(request["preferred_date"], "2099-01-01")
        self.assertEqual(request["preferred_time"], "18:00")
        self.assertEqual(request["missing_detail"], "handoff")
        self.assertEqual(item["service_name"], "Lip Filler 1 ml")
        self.assertEqual(item["price_pence"], 10000)
        self.assertEqual(conversation["active_booking_request_id"], request["id"])

    def test_projection_logs_safe_skip_and_insert_failure_codes(self):
        with patch.object(
            self.chatbot,
            "authoritative_service_metadata",
            return_value=None,
        ):
            with patch("builtins.print") as output:
                result = self.chatbot.sync_simple_single_request_projection(
                    "session-1",
                    {"treatment": "Unknown", "duration": "45 minutes"},
                    True,
                )

        self.assertFalse(result)
        output.assert_called_with("projection_skipped_missing_metadata")

        with patch.object(self.chatbot, "supabase", self.database):
            with patch.object(
                self.chatbot,
                "authoritative_service_metadata",
                return_value=self.metadata,
            ):
                with patch.object(
                    self.chatbot,
                    "next_simple_request_number",
                    return_value=1,
                ):
                    with patch.object(
                        self.chatbot,
                        "load_simple_draft_request",
                        return_value=None,
                    ):
                        with patch.object(
                            self.database,
                            "table",
                            side_effect=RuntimeError("hidden database detail"),
                        ):
                            with patch("builtins.print") as output:
                                result = self.chatbot.sync_simple_single_request_projection(
                                    "session-1",
                                    {
                                        "treatment": "Lip Filler 0.5 ml",
                                        "duration": "45 minutes",
                                    },
                                    True,
                                )

        self.assertFalse(result)
        output.assert_called_with("projection_request_insert_failed")


class RepeatedWordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def compose(self, state, calendar, detail, message, history):
        with patch.object(
            self.chatbot,
            "generate_natural_reply_body",
            return_value=(
                "Your treatment details are updated. "
                "Veronika will review the details and get back to you shortly."
            ),
        ):
            return self.chatbot.compose_verified_customer_reply(
                state,
                self.chatbot.empty_extractor_result(),
                calendar,
                detail,
                "",
                "",
                message,
                history,
            )

    def test_unchanged_calendar_and_handoff_are_not_repeated(self):
        state = {
            "preferred_date": "2099-01-01",
            "preferred_time": "10:00",
        }
        calendar = {"status": "free"}
        calendar_text = self.chatbot.safe_calendar_customer_text(state, calendar)
        handoff = "Veronika will confirm the appointment with you shortly."
        history = [
            self.chatbot.ChatMessage(
                role="assistant",
                content=f"{calendar_text}\n\n{handoff}",
            )
        ]

        reply = self.compose(state, calendar, "handoff", "thanks", history)

        self.assertNotIn(calendar_text, reply)
        self.assertNotIn(handoff, reply)
        self.assertNotIn("Veronika will review", reply)
        self.assertIn("Your treatment details are updated.", reply)

    def test_explicit_availability_question_can_repeat_calendar_text(self):
        state = {
            "preferred_date": "2099-01-01",
            "preferred_time": "10:00",
        }
        calendar = {"status": "free"}
        calendar_text = self.chatbot.safe_calendar_customer_text(state, calendar)
        history = [
            self.chatbot.ChatMessage(role="assistant", content=calendar_text)
        ]

        reply = self.compose(
            state,
            calendar,
            "name_and_phone",
            "is that slot still available?",
            history,
        )

        self.assertEqual(reply.count(calendar_text), 1)
        self.assertEqual(reply.count("?"), 1)


class GoogleCalendarDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def setUp(self):
        self.start = datetime(2099, 1, 1, 10, tzinfo=timezone.utc)
        self.end = self.start + timedelta(hours=1)
        self.chatbot.GOOGLE_CALENDAR_ID = "primary"

    def test_successful_free_and_busy_responses(self):
        for busy in [[], [{"start": self.start.isoformat(), "end": self.end.isoformat()}]]:
            with self.subTest(busy=busy):
                with patch.object(
                    self.chatbot,
                    "refresh_google_access_token",
                    return_value="placeholder",
                ):
                    with patch.object(
                        self.chatbot.urllib_request,
                        "urlopen",
                        return_value=UrlResponse({
                            "calendars": {"primary": {"busy": busy}}
                        }),
                    ):
                        result = self.chatbot.query_google_freebusy(
                            self.start,
                            self.end,
                        )

                self.assertEqual(result, busy)
                self.assertEqual(
                    self.chatbot.GOOGLE_CALENDAR_LAST_DIAGNOSTIC,
                    "freebusy_success",
                )

    def test_missing_token_refresh_failure_api_failure_malformed_and_timeout(self):
        with patch.object(
            self.chatbot,
            "get_google_refresh_token",
            return_value=None,
        ):
            self.assertIsNone(self.chatbot.refresh_google_access_token())
            self.assertEqual(
                self.chatbot.GOOGLE_CALENDAR_LAST_DIAGNOSTIC,
                "missing_stored_refresh_token",
            )

        with patch.object(
            self.chatbot,
            "get_google_refresh_token",
            return_value="placeholder",
        ):
            with patch.object(self.chatbot, "GOOGLE_CLIENT_ID", "placeholder"):
                with patch.object(
                    self.chatbot,
                    "GOOGLE_CLIENT_SECRET",
                    "placeholder",
                ):
                    with patch.object(
                        self.chatbot.urllib_request,
                        "urlopen",
                        side_effect=urllib_error.HTTPError(
                            "https://example.invalid",
                            400,
                            "bad request",
                            {},
                            None,
                        ),
                    ):
                        self.assertIsNone(self.chatbot.refresh_google_access_token())
                        self.assertEqual(
                            self.chatbot.GOOGLE_CALENDAR_LAST_DIAGNOSTIC,
                            "refresh_token_http_400",
                        )

                    with patch.object(
                        self.chatbot.urllib_request,
                        "urlopen",
                        return_value=UrlResponse({}),
                    ):
                        self.assertIsNone(self.chatbot.refresh_google_access_token())
                        self.assertEqual(
                            self.chatbot.GOOGLE_CALENDAR_LAST_DIAGNOSTIC,
                            "refresh_token_missing_access_token",
                        )

        with patch.object(
            self.chatbot,
            "refresh_google_access_token",
            return_value="placeholder",
        ):
            with patch.object(
                self.chatbot.urllib_request,
                "urlopen",
                side_effect=urllib_error.HTTPError(
                    "https://example.invalid",
                    400,
                    "bad request",
                    {},
                    None,
                ),
            ):
                self.assertIsNone(self.chatbot.query_google_freebusy(self.start, self.end))
                self.assertEqual(
                    self.chatbot.GOOGLE_CALENDAR_LAST_DIAGNOSTIC,
                    "freebusy_http_error",
                )

            with patch.object(
                self.chatbot.urllib_request,
                "urlopen",
                return_value=UrlResponse({"unexpected": "shape"}),
            ):
                self.assertIsNone(self.chatbot.query_google_freebusy(self.start, self.end))
                self.assertEqual(
                    self.chatbot.GOOGLE_CALENDAR_LAST_DIAGNOSTIC,
                    "malformed_api_response",
                )

            with patch.object(
                self.chatbot.urllib_request,
                "urlopen",
                side_effect=TimeoutError(),
            ):
                self.assertIsNone(self.chatbot.query_google_freebusy(self.start, self.end))
                self.assertEqual(
                    self.chatbot.GOOGLE_CALENDAR_LAST_DIAGNOSTIC,
                    "network_timeout",
                )

    def test_status_contains_only_safe_connection_diagnostic(self):
        self.chatbot.GOOGLE_CALENDAR_LAST_DIAGNOSTIC = "freebusy_http_error"

        with patch.object(
            self.chatbot,
            "get_google_refresh_token",
            return_value="hidden-token",
        ):
            status = asyncio.run(self.chatbot.google_calendar_status())

        self.assertEqual(status, {
            "connected": True,
            "last_diagnostic_code": "freebusy_http_error",
        })


if __name__ == "__main__":
    unittest.main()
