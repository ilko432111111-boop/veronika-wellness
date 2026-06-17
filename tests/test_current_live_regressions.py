import unittest
from unittest.mock import patch

from test_smoke import load_chatbot_module


class ProjectionSupabase:
    def __init__(self, existing_item=None):
        self.existing_item = existing_item
        self.writes = {
            "booking_requests": [],
            "booking_items": [],
            "conversations": [],
        }

    def table(self, name):
        return ProjectionQuery(self, name)


class ProjectionQuery:
    def __init__(self, database, table):
        self.database = database
        self.table = table
        self.operation = "select"
        self.payload = None

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

    def insert(self, payload):
        self.operation = "insert"
        self.payload = dict(payload)
        return self

    def update(self, payload):
        self.operation = "update"
        self.payload = dict(payload)
        return self

    def execute(self):
        response = type("Response", (), {})()

        if self.table == "booking_requests" and self.operation == "select":
            response.data = [{"id": 10, "request_number": 1}]
            return response

        if self.table == "booking_items" and self.operation == "select":
            response.data = [self.database.existing_item] if self.database.existing_item else []
            return response

        if self.operation in {"insert", "update"}:
            row = {**self.payload, "id": 20}
            self.database.writes[self.table].append(row)
            response.data = [row]
            return response

        response.data = []
        return response


class CurrentLiveRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def apply_state(self, existing, message, history=None, extractor=None):
        return self.chatbot.apply_validated_state_patch(
            existing_state=existing,
            extractor_result=extractor or self.chatbot.empty_extractor_result(),
            latest_message=message,
            history=history or [],
        )

    def controlled(self, state, calendar=None):
        return self.chatbot.apply_canonical_controller_state(
            state,
            True,
            calendar or {"status": "not_checked"},
        )

    def compose(self, state, message, detail=None, body="", calendar=None, extractor=None):
        with patch.object(
            self.chatbot,
            "generate_natural_reply_body",
            return_value=body,
        ):
            return self.chatbot.compose_verified_customer_reply(
                state=state,
                extractor_result=extractor or self.chatbot.empty_extractor_result(),
                calendar_result=calendar or {"status": "not_checked"},
                next_required_detail=detail,
                business_context="",
                services_context="",
                latest_message=message,
                history=[],
            )

    def project(self, state):
        database = ProjectionSupabase()

        with patch.object(self.chatbot, "supabase", database):
            self.assertTrue(
                self.chatbot.sync_simple_single_request_projection(
                    "current-live-regression",
                    state,
                    True,
                )
            )

        return database.writes

    def assert_projected_item(self, state, service_name, duration_minutes, price_pence):
        writes = self.project(state)
        request = writes["booking_requests"][0]
        item = writes["booking_items"][0]

        self.assertEqual(request["total_duration_minutes"], duration_minutes)
        self.assertEqual(request["total_price_pence"], price_pence)
        self.assertEqual(item["service_name"], service_name)
        self.assertEqual(item["duration_minutes"], duration_minutes)
        self.assertEqual(item["price_pence"], price_pence)

    def test_price_only_deep_tissue_does_not_start_booking_flow(self):
        message = "How much is a 2-hour deep tissue massage?"
        reply = self.compose({}, message)

        self.assertIn("\u00a395", reply)
        self.assertIn("Deep Tissue Massage", reply)
        self.assertFalse(
            self.chatbot.has_booking_intent_state(
                {},
                {"intent": "booking_request"},
                message,
            )
        )
        self.assertNotIn("which day", reply.lower())
        self.assertNotIn("what time", reply.lower())
        self.assertNotIn("name", reply.lower())
        self.assertNotIn("phone", reply.lower())

    def test_relaxing_massage_booking_saves_60_minutes_and_next_friday(self):
        message = "Can I book a relaxing massage next Friday at 15:00 for 60 minutes?"
        state = self.apply_state({}, message)
        controlled = self.controlled(state)
        expected_date = self.chatbot.parse_validated_date_from_text("next Friday")

        self.assertEqual(controlled["treatment"], "Relaxing Massage")
        self.assertEqual(controlled["active_service_name"], "Relaxing Massage")
        self.assertEqual(self.chatbot.parse_duration_minutes(controlled["duration"]), 60)
        self.assertEqual(controlled["preferred_date"], expected_date)
        self.assertEqual(controlled["preferred_time"], "15:00")
        self.assertEqual(controlled["next_required_detail"], "name_and_phone")
        self.assert_projected_item(controlled, "Relaxing Massage", 60, 5000)

    def test_duration_only_text_cannot_match_hydrafacial_alias(self):
        self.assertEqual(
            self.chatbot.service_metadata_match_score(
                "60 minutes please",
                "Hydrafacial",
                ["hydrafacial", "hydrafacial 90 minutes", "90 minute hydrafacial"],
            ),
            0,
        )

    def test_swedish_duration_reply_preserves_active_service(self):
        state = self.apply_state({}, "hey i wanna book swedish massage")
        extractor = self.chatbot.empty_extractor_result()
        extractor["state_patch"] = {
            "treatment": "Hydrafacial",
            "duration": "1 hour 30 minutes",
        }
        state = self.apply_state(
            state,
            "60 minutes please",
            [
                self.chatbot.ChatMessage(
                    role="assistant",
                    content="Which duration would you prefer?",
                )
            ],
            extractor=extractor,
        )
        controlled = self.controlled(state)
        reply = self.compose(
            controlled,
            "60 minutes please",
            controlled["next_required_detail"],
            body="Hydrafacial is only offered as a 90-minute treatment for \u00a380.",
        )

        self.assertEqual(controlled["treatment"], "Swedish Massage")
        self.assertEqual(controlled["active_service_name"], "Swedish Massage")
        self.assertEqual(controlled["active_category"], "massage")
        self.assertEqual(self.chatbot.parse_duration_minutes(controlled["duration"]), 60)
        self.assertEqual(controlled["_service_validation"]["accepted"], False)
        self.assertEqual(
            controlled["_service_validation"]["reason"],
            "slot_filling_message_preserved_active_service",
        )
        self.assertNotIn("Hydrafacial", reply)
        self.assertIn(
            controlled["next_required_detail"],
            {"preferred_date", "preferred_time"},
        )
        self.assert_projected_item(controlled, "Swedish Massage", 60, 5000)

    def test_swedish_direct_booking_uses_swedish_sixty_minutes(self):
        state = self.controlled(
            self.apply_state({}, "can i book swedish massage 60 minutes please")
        )
        reply = self.compose(
            state,
            "can i book swedish massage 60 minutes please",
            state["next_required_detail"],
        )

        self.assertEqual(state["treatment"], "Swedish Massage")
        self.assertEqual(state["active_service_name"], "Swedish Massage")
        self.assertEqual(self.chatbot.parse_duration_minutes(state["duration"]), 60)
        self.assertIn(state["next_required_detail"], {"preferred_date", "preferred_time"})
        self.assertNotIn("Hydrafacial", reply)
        self.assert_projected_item(
            {
                **state,
                "preferred_date": state.get("preferred_date") or "2026-06-20",
                "preferred_time": state.get("preferred_time") or "10:00",
            },
            "Swedish Massage",
            60,
            5000,
        )

    def test_swedish_today_then_soft_60_minutes_asks_for_time_next(self):
        state = self.apply_state({}, "I want a Swedish massage today")
        state = self.apply_state(
            state,
            "60 minutes possibly",
            [
                self.chatbot.ChatMessage(
                    role="assistant",
                    content="How long would you like the session for?",
                )
            ],
        )
        controlled = self.controlled(state)

        self.assertEqual(controlled["treatment"], "Swedish Massage")
        self.assertEqual(controlled["active_service_name"], "Swedish Massage")
        self.assertEqual(self.chatbot.parse_duration_minutes(controlled["duration"]), 60)
        self.assertNotEqual(self.chatbot.parse_duration_minutes(controlled["duration"]), 90)
        self.assertEqual(controlled["next_required_detail"], "preferred_time")
        self.assert_projected_item(controlled, "Swedish Massage", 60, 5000)

    def test_hydrafacial_flow_uses_hydrafacial_not_hydraface_and_blocks_handoff(self):
        state = self.apply_state({}, "I have acne scars, what facial do you recommend?")
        state = self.apply_state(state, "Hydrafacial")
        state = self.apply_state(state, "Saturday at 10:00")
        controlled = self.controlled(state)
        reply = self.compose(
            controlled,
            "Saturday at 10:00",
            controlled["next_required_detail"],
            body="Veronika will confirm shortly.",
        )

        self.assertEqual(controlled["treatment"], "Hydrafacial")
        self.assertEqual(controlled["active_service_name"], "Hydrafacial")
        self.assertEqual(self.chatbot.parse_duration_minutes(controlled["duration"]), 90)
        self.assertEqual(controlled["preferred_time"], "10:00")
        self.assertEqual(
            self.chatbot.simple_request_price_pence(
                self.chatbot.active_service_metadata_from_state(controlled),
                90,
            ),
            8000,
        )
        self.assertNotEqual(controlled["treatment"], "Hydraface Facial Treatment")
        self.assertEqual(controlled["next_required_detail"], "name_and_phone")
        self.assertFalse(
            self.chatbot.handoff_is_allowed(
                controlled,
                {"status": "not_checked"},
            )
        )
        self.assertNotIn("Veronika will confirm", reply)
        self.assert_projected_item(controlled, "Hydrafacial", 90, 8000)

    def test_duration_only_reply_after_hydrafacial_keeps_hydrafacial_fixed_duration(self):
        extractor = self.chatbot.empty_extractor_result()
        state = self.apply_state({}, "I want Hydrafacial")
        state = self.apply_state(
            state,
            "60 minutes please",
            [
                self.chatbot.ChatMessage(
                    role="assistant",
                    content="Which day would you prefer?",
                )
            ],
            extractor=extractor,
        )
        controlled = self.controlled(state)
        reply = self.compose(
            controlled,
            "60 minutes please",
            controlled["next_required_detail"],
            extractor=extractor,
        )

        self.assertEqual(controlled["treatment"], "Hydrafacial")
        self.assertEqual(controlled["active_service_name"], "Hydrafacial")
        self.assertIsNone(controlled["duration"])
        self.assertIn("Hydrafacial is only offered as a 90-minute treatment", reply)
        self.assertIn("\u00a380", reply)
        self.assertNotIn("Swedish", reply)
        self.assertNotIn("Hydraface Facial Treatment", reply)

    def test_explicit_switch_from_swedish_to_hydrafacial_changes_service(self):
        state = self.apply_state({}, "I want Swedish massage")
        state = self.controlled(
            self.apply_state(state, "Actually I want Hydrafacial")
        )
        metadata = self.chatbot.active_service_metadata_from_state(state)

        self.assertEqual(state["treatment"], "Hydrafacial")
        self.assertEqual(state["active_service_name"], "Hydrafacial")
        self.assertEqual(self.chatbot.parse_duration_minutes(state["duration"]), 90)
        self.assertEqual(self.chatbot.simple_request_price_pence(metadata, 90), 8000)

    def test_hydraface_treatment_is_separate_from_hydrafacial(self):
        state = self.controlled(
            self.apply_state({}, "Hydraface Facial Treatment")
        )
        metadata = self.chatbot.active_service_metadata_from_state(state)

        self.assertEqual(state["treatment"], "Hydraface Facial Treatment")
        self.assertEqual(state["active_service_name"], "Hydraface Facial Treatment")
        self.assertEqual(self.chatbot.parse_duration_minutes(state["duration"]), 60)
        self.assertEqual(self.chatbot.simple_request_price_pence(metadata, 60), 6000)
        self.assertNotEqual(state["treatment"], "Hydrafacial")

    def test_filler_variant_selects_nasolabial_half_ml_only(self):
        generic = self.controlled(
            self.apply_state({}, "I want filler")
        )
        selected = self.controlled(
            self.apply_state(generic, "Nasolabial Folds 0.5 ml")
        )
        metadata = self.chatbot.active_service_metadata_from_state(selected)

        self.assertEqual(generic["treatment"], "Dermal Filler")
        self.assertEqual(generic["next_required_detail"], "service_variant")
        self.assertIn("Nasolabial Folds 0.5 ml", selected["treatment"])
        self.assertNotIn("Lip Filler", selected["treatment"])
        self.assertNotIn("Marionette Lines", selected["treatment"])
        self.assertEqual(self.chatbot.parse_duration_minutes(selected["duration"]), 45)
        self.assertEqual(self.chatbot.simple_request_price_pence(metadata, 45), 7000)
        self.assert_projected_item(
            {
                **selected,
                "preferred_date": "2026-06-20",
                "preferred_time": "10:00",
                "next_required_detail": "name_and_phone",
            },
            selected["active_service_name"],
            45,
            7000,
        )

    def test_children_and_pets_policy_does_not_force_booking_flow(self):
        message = "Can I bring my child and dog?"
        reply = self.compose({}, message)

        self.assertIn("children are okay", reply.lower())
        self.assertIn("not allowed", reply.lower())
        self.assertNotIn("waiting area", reply.lower())
        self.assertFalse(
            self.chatbot.has_booking_intent_state(
                {},
                {"intent": "booking_request"},
                message,
            )
        )
        self.assertNotIn("which day", reply.lower())

    def test_services_list_is_grouped_and_mobile_readable(self):
        reply = self.compose({}, "What services do you offer?")

        self.assertIn("\n\nMassage\n", reply)
        self.assertIn("\n\nFacials\n", reply)
        self.assertIn("\u2022 ", reply)
        self.assertGreaterEqual(reply.count("\n\n"), 3)
        self.assertNotIn("Dermal Fillers\n", reply[:80])
        self.assertLess(max(len(line) for line in reply.splitlines()), 180)

    def test_ready_for_review_requires_contact_and_checked_availability(self):
        base = {
            "_canonical_save_succeeded": True,
            "conversation_mode": "booking_request",
            "treatment": "Relaxing Massage",
            "duration": "1 hour",
            "preferred_date": "2026-06-19",
            "preferred_time": "15:00",
            "verified_alternatives": [],
            "next_required_detail": "handoff",
        }

        self.assertFalse(
            self.chatbot.is_ready_for_owner_review(
                {
                    **base,
                    "slot_status": "free",
                    "phone": "07700900123",
                }
            )
        )
        self.assertFalse(
            self.chatbot.is_ready_for_owner_review(
                {
                    **base,
                    "slot_status": "free",
                    "name": "Maya",
                }
            )
        )
        self.assertFalse(
            self.chatbot.is_ready_for_owner_review(
                {
                    **base,
                    "slot_status": "not_checked",
                    "name": "Maya",
                    "phone": "07700900123",
                }
            )
        )
        self.assertTrue(
            self.chatbot.is_ready_for_owner_review(
                {
                    **base,
                    "slot_status": "free",
                    "name": "Maya",
                    "phone": "07700900123",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
