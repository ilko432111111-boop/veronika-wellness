from datetime import datetime, timedelta
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
