from datetime import datetime, timedelta
import unittest
from unittest.mock import patch

from fastapi import BackgroundTasks

from test_smoke import load_chatbot_module


class LiveBugRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def apply_state(self, existing, message, history=None):
        with patch.object(
            self.chatbot,
            "hydrate_configured_service_defaults",
            side_effect=lambda state: state,
        ):
            return self.chatbot.apply_validated_state_patch(
                existing_state=existing,
                extractor_result=self.chatbot.empty_extractor_result(),
                latest_message=message,
                history=history or [],
            )

    def compose(self, body, state, message, detail=None, calendar=None):
        with patch.object(
            self.chatbot,
            "generate_natural_reply_body",
            return_value=body,
        ):
            return self.chatbot.compose_verified_customer_reply(
                state=state,
                extractor_result=self.chatbot.empty_extractor_result(),
                calendar_result=calendar or {"status": "not_checked"},
                next_required_detail=detail,
                business_context="",
                services_context="",
                latest_message=message,
                history=[],
            )

    def test_relaxing_massage_next_friday_60_minutes_keeps_duration_and_date(self):
        message = "Relaxing massage next Friday at 15:00, 60 minutes"
        state = self.apply_state({}, message)
        expected_date = self.chatbot.parse_validated_date_from_text("next Friday")

        self.assertEqual(state["treatment"], "Relaxing Massage")
        self.assertEqual(state["duration"], "1 hour")
        self.assertEqual(state["preferred_date"], expected_date)
        self.assertEqual(state["preferred_time"], "15:00")

    def test_soft_60_min_reply_overwrites_stale_90_min_duration(self):
        history = [
            self.chatbot.ChatMessage(
                role="assistant",
                content="How long would you like the session for: 30, 60, 90, or 120 minutes?",
            )
        ]
        state = self.apply_state(
            {"treatment": "Relaxing Massage", "duration": "1 hour 30 minutes"},
            "60 minutes possibly",
            history,
        )

        self.assertEqual(state["duration"], "1 hour")

    def test_response_and_projection_use_same_resolved_duration_and_date(self):
        projected = {}

        class Query:
            def __init__(self, table):
                self.table = table

            def select(self, *args, **kwargs):
                return self

            def eq(self, *args, **kwargs):
                return self

            def in_(self, *args, **kwargs):
                return self

            def order(self, *args, **kwargs):
                return self

            def limit(self, *args, **kwargs):
                return self

            def update(self, payload):
                projected[self.table] = payload
                return self

            def insert(self, payload):
                projected[self.table] = payload
                return self

            def execute(self):
                if self.table == "booking_requests" and "request_status" not in projected.get(self.table, {}):
                    return type("Response", (), {"data": []})()
                if self.table == "booking_items" and "service_name" not in projected.get(self.table, {}):
                    return type("Response", (), {"data": []})()
                if self.table == "booking_requests":
                    return type("Response", (), {"data": [{"id": 10}]})()
                return type("Response", (), {"data": [{"id": 20}]})()

        class Supabase:
            def table(self, name):
                return Query(name)

        state = {
            "conversation_mode": "booking_request",
            "treatment": "Relaxing Massage",
            "duration": "1 hour",
            "preferred_date": "2026-06-19",
            "preferred_time": "15:00",
            "slot_status": "not_checked",
            "next_required_detail": "name_and_phone",
        }
        metadata = {
            "service_id": 1,
            "service_name": "Relaxing Massage",
            "category": "massage",
            "price_by_duration": {"60": 5000, "90": 7500},
            "price_pence": None,
        }

        with patch.object(self.chatbot, "supabase", Supabase()), patch.object(
            self.chatbot, "authoritative_service_metadata", return_value=metadata
        ):
            self.assertTrue(
                self.chatbot.sync_simple_single_request_projection(
                    "duration-projection",
                    state,
                    True,
                )
            )

        reply = self.compose(
            "A 90-minute Relaxing Massage is \u00a375.",
            state,
            "60 minutes",
            "name_and_phone",
        )

        self.assertEqual(projected["booking_requests"]["total_duration_minutes"], 60)
        self.assertEqual(projected["booking_items"]["duration_minutes"], 60)
        self.assertNotIn("90-minute", reply)
        self.assertIn("1 hour", reply)

    def test_hydrafacial_and_hydraface_resolve_to_distinct_services(self):
        hydrafacial = self.apply_state({}, "I want Hydrafacial Saturday 10:00")
        hydraface = self.apply_state({}, "Hydraface facial")

        self.assertEqual(hydrafacial["treatment"], "Hydrafacial")
        self.assertEqual(hydrafacial["duration"], "1 hour 30 minutes")
        self.assertEqual(
            self.chatbot.simple_request_price_pence(
                self.chatbot.structured_service_identity("Hydrafacial"),
                90,
            ),
            8000,
        )
        self.assertEqual(hydraface["treatment"], "Hydraface Facial Treatment")
        self.assertEqual(hydraface["duration"], "1 hour")
        self.assertEqual(
            self.chatbot.simple_request_price_pence(
                self.chatbot.structured_service_identity("Hydraface Facial Treatment"),
                60,
            ),
            6000,
        )

    def test_next_required_detail_uses_existing_name_and_phone(self):
        base = {
            "conversation_mode": "booking_request",
            "treatment": "Hydrafacial",
            "duration": "1 hour 30 minutes",
            "preferred_date": "2026-06-20",
            "preferred_time": "10:00",
        }

        self.assertEqual(
            self.chatbot.compute_next_required_detail(
                {**base, "name": "Maya", "phone": "07700900123"},
                True,
            ),
            "handoff",
        )
        self.assertEqual(
            self.chatbot.compute_next_required_detail(
                {**base, "name": "Maya"},
                True,
            ),
            "phone",
        )
        self.assertEqual(
            self.chatbot.compute_next_required_detail(
                {**base, "phone": "07700900123"},
                True,
            ),
            "name",
        )

    def test_dermal_filler_variants_and_generic_filler(self):
        lip = self.apply_state({}, "lip filler")
        nasolabial = self.apply_state({}, "nasolabial filler")
        generic = self.apply_state({}, "filler")

        self.assertIn("Lip Filler", lip["treatment"])
        self.assertIn("Nasolabial", nasolabial["treatment"])
        self.assertEqual(generic["treatment"], "Dermal Filler")
        self.assertEqual(
            self.chatbot.compute_next_required_detail(generic, True),
            "service_variant",
        )

    def test_price_question_returns_price_only_and_does_not_start_booking(self):
        reply = self.compose(
            "This model answer should not be used.",
            {},
            "How much is a 2-hour deep tissue massage?",
            None,
        )

        self.assertEqual(
            reply,
            "Deep Tissue Massage for 120 minutes is \u00a395. Would you like me to check availability?",
        )
        self.assertFalse(
            self.chatbot.has_booking_intent_state(
                {},
                {"intent": "booking_request"},
                "How much is a 2-hour deep tissue massage?",
            )
        )
        self.assertNotIn("Which day", reply)

    def test_booking_and_availability_phrases_enter_flow(self):
        self.assertTrue(
            self.chatbot.has_booking_intent_state(
                {},
                {"intent": "unclear"},
                "Can I book 2-hour deep tissue massage Friday?",
            )
        )
        self.assertTrue(
            self.chatbot.has_booking_intent_state(
                {},
                {"intent": "unclear"},
                "Do you have availability for deep tissue Friday?",
            )
        )

    def test_children_and_pets_policy_is_locked_and_not_booking_prompt(self):
        child = self.compose("", {}, "Can I bring my child?")
        dog = self.compose("", {}, "Can I bring my dog?")
        both = self.compose("", {}, "Can I bring children and dogs?")

        self.assertIn("children are okay", child.lower())
        self.assertIn("not allowed", dog.lower())
        self.assertIn("children are okay", both.lower())
        self.assertIn("not allowed", both.lower())
        for reply in [child, dog, both]:
            self.assertNotIn("waiting area", reply.lower())
            self.assertNotIn("Which day", reply)

    def test_ready_for_owner_review_requires_verified_free_slot(self):
        state = {
            "_canonical_save_succeeded": True,
            "conversation_mode": "booking_request",
            "treatment": "Relaxing Massage",
            "duration": "1 hour",
            "preferred_date": "2026-06-19",
            "preferred_time": "15:00",
            "name": "Maya",
            "phone": "07700900123",
            "next_required_detail": "handoff",
            "verified_alternatives": [],
        }

        self.assertFalse(
            self.chatbot.is_ready_for_owner_review(
                {**state, "slot_status": "not_checked"}
            )
        )
        self.assertFalse(
            self.chatbot.is_ready_for_owner_review(
                {**state, "slot_status": "unavailable"}
            )
        )
        self.assertTrue(
            self.chatbot.is_ready_for_owner_review(
                {**state, "slot_status": "provisional_free"}
            )
        )


class ServiceLockProjectionRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def metadata(self, treatment):
        name = self.chatbot.normalise_treatment_name(treatment)
        if name == "swedish massage":
            return {
                "service_id": 101,
                "service_name": "Swedish Massage",
                "category": "massage",
                "booking_mode": "choose_duration",
                "fixed_duration_minutes": None,
                "allowed_durations_minutes": [30, 60, 90, 120],
                "price_pence": None,
                "price_by_duration": {
                    "30": 3500,
                    "60": 5000,
                    "90": 7500,
                    "120": 9500,
                },
                "source": "test_services",
            }
        if name == "hydrafacial":
            return {
                "service_id": 202,
                "service_name": "Hydrafacial",
                "category": "facials",
                "booking_mode": "fixed_duration",
                "fixed_duration_minutes": 90,
                "allowed_durations_minutes": [90],
                "price_pence": 8000,
                "price_by_duration": None,
                "source": "test_services",
            }
        return None

    def apply_state(self, existing, message, history=None, extractor=None):
        with patch.object(
            self.chatbot,
            "authoritative_service_metadata",
            side_effect=self.metadata,
        ):
            return self.chatbot.apply_validated_state_patch(
                existing_state=existing,
                extractor_result=extractor or self.chatbot.empty_extractor_result(),
                latest_message=message,
                history=history or [],
            )

    def project(self, state, existing_item=None):
        writes = {
            "booking_requests": [],
            "booking_items": [],
            "conversations": [],
        }

        class Response:
            def __init__(self, data):
                self.data = data

        class Query:
            def __init__(self, table):
                self.table = table
                self.operation = "select"
                self.payload = None
                self.filters = []

            def select(self, *args, **kwargs):
                return self

            def eq(self, field, value):
                self.filters.append((field, value))
                return self

            def in_(self, *args, **kwargs):
                return self

            def order(self, *args, **kwargs):
                return self

            def limit(self, *args, **kwargs):
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
                if self.table == "booking_requests" and self.operation == "select":
                    return Response([{"id": 55, "request_number": 1}])
                if self.table == "booking_items" and self.operation == "select":
                    return Response([existing_item] if existing_item else [])
                if self.operation in {"insert", "update"}:
                    row = {**self.payload, "id": 77}
                    writes[self.table].append(row)
                    return Response([row])
                return Response([])

        class Supabase:
            def table(self, name):
                return Query(name)

        with patch.object(self.chatbot, "supabase", Supabase()), patch.object(
            self.chatbot,
            "authoritative_service_metadata",
            side_effect=self.metadata,
        ):
            ok = self.chatbot.sync_simple_single_request_projection(
                "service-lock-session",
                state,
                True,
            )

        self.assertTrue(ok)
        return writes

    def complete_swedish_state(self, **overrides):
        state = {
            "conversation_mode": "booking_request",
            "treatment": "Swedish Massage",
            "active_service_id": 101,
            "active_service_name": "Swedish Massage",
            "active_service_source": "latest_explicit_customer_intent",
            "active_category": "massage",
            "duration": "1 hour",
            "preferred_date": "2026-06-19",
            "preferred_time": "15:00",
            "slot_status": "not_checked",
            "next_required_detail": "name_and_phone",
        }
        state.update(overrides)
        return state

    def assert_projected_swedish_60(self, state, existing_item=None):
        writes = self.project(state, existing_item=existing_item)
        item = writes["booking_items"][0]
        request = writes["booking_requests"][0]

        self.assertEqual(item["service_id"], 101)
        self.assertEqual(item["service_name"], "Swedish Massage")
        self.assertEqual(item["duration_minutes"], 60)
        self.assertEqual(item["price_pence"], 5000)
        self.assertEqual(request["total_duration_minutes"], 60)
        self.assertEqual(request["total_price_pence"], 5000)

    def test_bot_hydrafacial_suggestion_cannot_pollute_swedish_booking(self):
        history = [
            self.chatbot.ChatMessage(
                role="assistant",
                content="For skin concerns, Hydrafacial is 1 hr 30 min and costs \u00a380.",
            )
        ]
        state = self.apply_state(
            {},
            "Can I book Swedish massage today",
            history,
        )
        state = self.apply_state(
            state,
            "60 minutes possibly",
            [
                *history,
                self.chatbot.ChatMessage(
                    role="assistant",
                    content="How long would you like the session for?",
                ),
            ],
        )

        self.assertEqual(state["active_service_name"], "Swedish Massage")
        self.assertEqual(state["duration"], "1 hour")
        self.assert_projected_swedish_60(self.complete_swedish_state(**state))

    def test_previous_session_hydrafacial_history_cannot_select_current_service(self):
        history = [
            self.chatbot.ChatMessage(
                role="user",
                content="I had Hydrafacial before.",
            ),
            self.chatbot.ChatMessage(
                role="assistant",
                content="Hydrafacial is 90 minutes.",
            ),
        ]
        state = self.apply_state(
            {},
            "Can I book Swedish massage today 60 minutes",
            history,
        )

        self.assertEqual(state["active_service_name"], "Swedish Massage")
        self.assertEqual(state["duration"], "1 hour")

    def test_switch_from_hydrafacial_removes_stale_item_and_saves_swedish(self):
        state = self.apply_state(
            {
                "treatment": "Hydrafacial",
                "active_service_id": 202,
                "active_service_name": "Hydrafacial",
                "active_service_source": "canonical_state",
                "duration": "1 hour 30 minutes",
            },
            "Actually Swedish massage 60 minutes",
        )

        self.assertEqual(state["active_service_name"], "Swedish Massage")
        self.assertEqual(state["duration"], "1 hour")
        self.assert_projected_swedish_60(
            self.complete_swedish_state(**state),
            existing_item={"id": 9, "service_name": "Hydrafacial"},
        )

    def test_generic_facial_question_then_swedish_booking_saves_swedish(self):
        state = self.apply_state({}, "What facials do you do?")
        state = self.apply_state(
            state,
            "Can I book Swedish massage 60 minutes?",
        )

        self.assertEqual(state["active_service_name"], "Swedish Massage")
        self.assertEqual(state["duration"], "1 hour")

    def test_wrong_ai_extractor_treatment_is_rejected_by_latest_service(self):
        extractor = self.chatbot.empty_extractor_result()
        extractor["state_patch"] = {
            "treatment": "Hydrafacial",
            "duration": "1 hour 30 minutes",
        }
        state = self.apply_state(
            {},
            "Can I book Swedish massage 60 minutes?",
            extractor=extractor,
        )

        self.assertEqual(state["active_service_name"], "Swedish Massage")
        self.assertEqual(state["duration"], "1 hour")
        self.assertEqual(
            state["_service_validation"]["reason"],
            "latest_explicit_customer_service_wins",
        )

    def test_soft_60_minutes_attaches_to_active_swedish_only(self):
        state = self.apply_state(
            {
                "treatment": "Swedish Massage",
                "active_service_id": 101,
                "active_service_name": "Swedish Massage",
                "duration": None,
            },
            "60 minutes possibly",
            [
                self.chatbot.ChatMessage(
                    role="assistant",
                    content="How long would you like the session for?",
                )
            ],
        )

        self.assertEqual(state["active_service_name"], "Swedish Massage")
        self.assertEqual(state["duration"], "1 hour")

    def test_dashboard_projection_matches_booking_item(self):
        state = self.complete_swedish_state()
        writes = self.project(state)
        item = writes["booking_items"][0]

        dashboard_line = (
            f"{item['service_name']} - "
            f"{self.chatbot.format_customer_menu_duration(item['duration_minutes'])}"
        )

        self.assertIn("Swedish Massage", dashboard_line)
        self.assertIn("1 hr", dashboard_line)


class SafetyReplyRegressionTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def test_conversation_save_retries_without_optional_active_service_fields(self):
        writes = []

        class Query:
            def __init__(self):
                self.payload = None

            def upsert(self, payload, **kwargs):
                self.payload = dict(payload)
                return self

            def execute(self):
                writes.append(self.payload)
                if "active_service_name" in self.payload:
                    raise RuntimeError("unknown column active_service_name")
                return type("Response", (), {"data": [self.payload]})()

        class Supabase:
            def table(self, name):
                return Query()

        with patch.object(self.chatbot, "supabase", Supabase()):
            self.assertTrue(
                self.chatbot.save_conversation_overview(
                    "save-fallback-session",
                    {
                        "treatment": "Swedish Massage",
                        "active_service_id": 101,
                        "active_service_name": "Swedish Massage",
                        "active_service_source": "latest_explicit_customer_intent",
                    },
                )
            )

        self.assertEqual(len(writes), 2)
        self.assertIn("active_service_name", writes[0])
        self.assertNotIn("active_service_name", writes[1])

    async def test_projection_failure_does_not_show_safety_error_for_swedish_flow(self):
        messages = [
            "Can I book Swedish massage today",
            "60 minutes",
            "16:30",
        ]

        with patch.object(self.chatbot, "save_message"), patch.object(
            self.chatbot, "get_existing_conversation", return_value={}
        ), patch.object(
            self.chatbot, "active_booking_request_status", return_value=None
        ), patch.object(
            self.chatbot, "load_recent_messages", return_value=[]
        ), patch.object(
            self.chatbot, "build_authoritative_services_context", return_value=""
        ), patch.object(
            self.chatbot, "load_business_documents_context", return_value=""
        ), patch.object(
            self.chatbot,
            "extract_hidden_state_patch",
            return_value=self.chatbot.empty_extractor_result(),
        ), patch.object(
            self.chatbot, "save_conversation_overview", return_value=True
        ), patch.object(
            self.chatbot,
            "sync_simple_single_request_projection",
            side_effect=RuntimeError("booking_items unavailable"),
        ), patch.object(
            self.chatbot,
            "compose_verified_customer_reply",
            return_value="Normal assistant reply.",
        ):
            for message in messages:
                response = await self.chatbot.chat(
                    self.chatbot.ChatRequest(
                        session_id="swedish-no-safety",
                        message=message,
                        history=[],
                    ),
                    BackgroundTasks(),
                )
                self.assertNotIn(
                    "could not safely save",
                    response["reply"].lower(),
                )


if __name__ == "__main__":
    unittest.main()
