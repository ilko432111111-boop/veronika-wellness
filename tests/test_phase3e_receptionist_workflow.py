import unittest
from unittest.mock import patch

from test_smoke import load_chatbot_module


class Phase3ERemainingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()
        cls.alternatives = [
            "Thursday 11 June at 09:00",
            "Thursday 11 June at 11:00",
            "Thursday 11 June at 11:30",
        ]

    def compose(
        self,
        body,
        state,
        message,
        extractor=None,
        calendar=None,
        detail="verified_alternative",
    ):
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

    def test_soft_premature_handoff_language_is_removed(self):
        phrases = [
            "We'll take care of the rest.",
            "We'll take care of the next steps.",
            "We'll get everything set up.",
            "We'll make sure everything is ready.",
            "We'll make sure that's set for you.",
            "We'll arrange the remaining details shortly.",
            "I'll look into suitable times and get back to you shortly.",
            "We'll be in touch.",
        ]

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                reply = self.compose(
                    phrase,
                    {
                        "conversation_mode": "booking_request",
                        "treatment": "Relaxing Massage",
                    },
                    "I'd like a relaxing massage",
                    detail="duration",
                )
                self.assertNotIn(phrase.lower(), reply.lower())
                self.assertIn("I've noted that.", reply)
                self.assertIn("How long would you like", reply)

    def test_contact_details_are_acknowledged_before_verified_alternatives(self):
        extractor = self.chatbot.empty_extractor_result()
        extractor["state_patch"] = {
            "name": "Ilko",
            "phone": "07700900114",
        }
        reply = self.compose(
            "",
            {
                "conversation_mode": "booking_request",
                "treatment": "EMS",
                "duration": "1 hour",
                "preferred_date": "2026-06-10",
                "preferred_time": "14:00",
                "name": "Ilko",
                "phone": "07700900114",
            },
            "Ilko, 07700900114",
            extractor=extractor,
            calendar={
                "status": "pending_alternatives",
                "suggestions": self.alternatives,
            },
        )

        self.assertTrue(reply.startswith("Thanks, I've got your name and phone number."))
        self.assertIn("The current verified options are", reply)
        self.assertEqual(reply.count("?"), 1)

    def test_side_question_is_answered_before_verified_alternatives(self):
        reply = self.compose(
            "EMS costs £200 for a one-hour session.",
            {
                "conversation_mode": "booking_request",
                "treatment": "EMS",
                "duration": "1 hour",
                "preferred_date": "2026-06-10",
                "preferred_time": "14:00",
            },
            "How much is EMS?",
            calendar={
                "status": "pending_alternatives",
                "suggestions": self.alternatives,
            },
        )

        self.assertTrue(reply.startswith("EMS costs £200 for a one-hour session."))
        self.assertIn("The current verified options are", reply)
        self.assertEqual(reply.count("?"), 1)

    def test_extractor_cannot_invent_multi_duration_service_length(self):
        extractor = self.chatbot.empty_extractor_result()
        extractor["state_patch"] = {
            "treatment": "Relaxing Massage",
            "duration": "30 minutes",
        }

        with patch.object(
            self.chatbot,
            "apply_structured_service_resolution",
            side_effect=lambda state, message: state,
        ), patch.object(
            self.chatbot,
            "hydrate_configured_service_defaults",
            side_effect=lambda state: state,
        ):
            state = self.chatbot.apply_validated_state_patch(
                existing_state={},
                extractor_result=extractor,
                latest_message="I'd like a relaxing massage.",
                history=[],
            )

        self.assertEqual(state["treatment"], "Relaxing Massage")
        self.assertIsNone(state["duration"])
        self.assertEqual(
            self.chatbot.compute_next_required_detail(state, True),
            "duration",
        )

    def test_vague_booking_intent_cannot_create_extractor_treatment(self):
        extractor = self.chatbot.empty_extractor_result()
        extractor["intent"] = "booking_request"
        extractor["state_patch"] = {
            "treatment": "Swedish Massage",
        }

        with patch.object(
            self.chatbot,
            "grounded_treatment_from_latest_message",
            return_value=None,
        ), patch.object(
            self.chatbot,
            "apply_structured_service_resolution",
            side_effect=lambda state, message: {
                **state,
                "treatment": "Swedish Massage",
            },
        ), patch.object(
            self.chatbot,
            "hydrate_configured_service_defaults",
            side_effect=lambda state: state,
        ):
            state = self.chatbot.apply_validated_state_patch(
                existing_state={},
                extractor_result=extractor,
                latest_message="I'd like to book an appointment.",
                history=[],
            )

        self.assertIsNone(state["treatment"])
        self.assertEqual(
            self.chatbot.compute_next_required_detail(state, True),
            "treatment",
        )
        self.assertEqual(
            self.chatbot.render_next_question(state, "treatment"),
            "Which treatment are you interested in?",
        )

    def test_existing_active_treatment_survives_vague_booking_follow_up(self):
        extractor = self.chatbot.empty_extractor_result()
        extractor["state_patch"] = {
            "treatment": "Deep Tissue Massage",
        }

        with patch.object(
            self.chatbot,
            "grounded_treatment_from_latest_message",
            return_value=None,
        ), patch.object(
            self.chatbot,
            "apply_structured_service_resolution",
            side_effect=lambda state, message: state,
        ), patch.object(
            self.chatbot,
            "hydrate_configured_service_defaults",
            side_effect=lambda state: state,
        ):
            state = self.chatbot.apply_validated_state_patch(
                existing_state={"treatment": "Swedish Massage"},
                extractor_result=extractor,
                latest_message="Yes, I'd like to book it.",
                history=[],
            )

        self.assertEqual(state["treatment"], "Swedish Massage")

    def test_hot_stone_short_duration_replies_move_to_date(self):
        history = [
            self.chatbot.ChatMessage(
                role="assistant",
                content=(
                    "How long would you like the session for: "
                    "60, 90, or 120 minutes?"
                ),
            ),
        ]

        for message in ["60", "60 then", "60 minutes", "one hour", "1 hour"]:
            with self.subTest(message=message), patch.object(
                self.chatbot,
                "apply_structured_service_resolution",
                side_effect=lambda state, value: state,
            ), patch.object(
                self.chatbot,
                "hydrate_configured_service_defaults",
                side_effect=lambda state: state,
            ):
                state = self.chatbot.apply_validated_state_patch(
                    existing_state={"treatment": "Hot Stone Massage"},
                    extractor_result=self.chatbot.empty_extractor_result(),
                    latest_message=message,
                    history=history,
                )

            self.assertEqual(state["duration"], "1 hour")
            self.assertEqual(
                self.chatbot.compute_next_required_detail(state, True),
                "preferred_date",
            )

    def test_common_duration_formats_are_parsed(self):
        cases = {
            "30mins": 30,
            "30 mins": 30,
            "30min": 30,
            "30 min": 30,
            "30minutes": 30,
            "30 minutes": 30,
            "1hr": 60,
            "1 hr": 60,
            "1hour": 60,
            "1 hour": 60,
            "60mins": 60,
            "90mins": 90,
            "90 mins": 90,
            "2hrs": 120,
            "2 hours": 120,
            "one hour": 60,
            "half an hour": 30,
            "1.5 hours": 90,
            "1 hour 30": 90,
            "1h30": 90,
            "1 hr 30 min": 90,
            "hour and a half": 90,
            "hour": 60,
        }

        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(
                    self.chatbot.parse_duration_minutes(value),
                    expected,
                )

    def test_soft_duration_reply_replaces_stale_massage_duration(self):
        history = [
            self.chatbot.ChatMessage(
                role="assistant",
                content="How long would you like the session for: 30, 60, 90, or 120 minutes?",
            ),
        ]

        with patch.object(
            self.chatbot,
            "apply_structured_service_resolution",
            side_effect=lambda state, message: state,
        ), patch.object(
            self.chatbot,
            "hydrate_configured_service_defaults",
            side_effect=lambda state: state,
        ), patch.object(
            self.chatbot,
            "allowed_durations_for_treatment",
            return_value={30, 60, 90, 120},
        ):
            for message in [
                "60 minutes possibly",
                "60mins",
                "1 hour",
                "one hour",
            ]:
                with self.subTest(message=message):
                    state = self.chatbot.apply_validated_state_patch(
                        existing_state={
                            "treatment": "Swedish Massage",
                            "duration": "1 hour 30 minutes",
                        },
                        extractor_result=self.chatbot.empty_extractor_result(),
                        latest_message=message,
                        history=history,
                    )
                    self.assertEqual(state["duration"], "1 hour")

    def test_conflicting_duration_reply_requires_clarification(self):
        history = [
            self.chatbot.ChatMessage(
                role="assistant",
                content="How long would you like the session for?",
            ),
        ]
        extractor = self.chatbot.empty_extractor_result()

        with patch.object(
            self.chatbot,
            "apply_structured_service_resolution",
            side_effect=lambda state, message: state,
        ), patch.object(
            self.chatbot,
            "hydrate_configured_service_defaults",
            side_effect=lambda state: state,
        ), patch.object(
            self.chatbot,
            "allowed_durations_for_treatment",
            return_value={30, 60, 90, 120},
        ):
            state = self.chatbot.apply_validated_state_patch(
                existing_state={
                    "treatment": "Swedish Massage",
                    "duration": "1 hour 30 minutes",
                    "preferred_date": "2099-01-01",
                    "preferred_time": "12:00",
                },
                extractor_result=extractor,
                latest_message="maybe 60 or 90",
                history=history,
            )

        self.assertEqual(state["duration"], "1 hour 30 minutes")
        self.assertTrue(state["_duration_ambiguous"])
        self.assertEqual(
            self.chatbot.compute_next_required_detail(state, True),
            "duration",
        )
        self.assertEqual(
            self.chatbot.verified_calendar_result_for_state(
                state,
                True,
                "maybe 60 or 90",
            )["status"],
            "not_checked",
        )
        reply = self.compose(
            "A 90-minute massage sounds good.",
            state,
            "maybe 60 or 90",
            extractor=extractor,
            detail="duration",
        )
        self.assertIn("How long would you like", reply)
        self.assertNotIn("90-minute massage sounds good", reply)

    def test_response_duration_is_forced_to_canonical_state(self):
        with patch.object(
            self.chatbot,
            "authoritative_service_metadata",
            return_value={
                "price_pence": None,
                "price_by_duration": {"60": 5500, "90": 7500},
            },
        ):
            reply = self.compose(
                "A 90-minute Swedish massage sounds perfect. It costs £75.",
                {
                    "conversation_mode": "booking_request",
                    "treatment": "Swedish Massage",
                    "duration": "1 hour",
                },
                "60 minutes possibly",
                detail="preferred_date",
            )

        self.assertIn("I've noted the 1 hour duration.", reply)
        self.assertNotIn("90-minute", reply)
        self.assertNotIn("£75", reply)

    def test_massage_price_uses_resolved_duration(self):
        metadata = {
            "price_by_duration": {
                "30": 3500,
                "60": 5500,
                "90": 7500,
                "120": 9500,
            },
            "price_pence": None,
        }

        for minutes, price in [(30, 3500), (60, 5500), (90, 7500), (120, 9500)]:
            with self.subTest(minutes=minutes):
                self.assertEqual(
                    self.chatbot.simple_request_price_pence(metadata, minutes),
                    price,
                )

    def test_owner_service_editor_validation_rejects_bad_input(self):
        valid = {
            "category_key": "massage",
            "service_name": "Swedish Massage",
            "duration_minutes": 60,
            "price_pence": 5500,
            "is_active": True,
            "aliases": ["swedish"],
            "requires_duration_choice": False,
            "sort_order": 10,
        }
        invalid_cases = [
            {**valid, "price_pence": -1},
            {**valid, "service_name": ""},
            {**valid, "duration_minutes": 999999},
            {**valid, "service_name": "<script>alert(1)</script>"},
            {**valid, "category_key": "Not Safe"},
        ]

        for payload in invalid_cases:
            with self.subTest(payload=payload), self.assertRaises(
                self.chatbot.HTTPException
            ):
                self.chatbot.clean_admin_service_payload(payload)

    def test_owner_service_editor_maps_selectable_duration_prices(self):
        cleaned = self.chatbot.clean_admin_service_payload({
            "category_key": "massage",
            "service_name": "Swedish Massage",
            "duration_minutes": None,
            "price_pence": 3500,
            "is_active": True,
            "aliases": ["swedish"],
            "requires_duration_choice": True,
            "sort_order": 10,
            "duration_prices": {"30": 3500, "60": 5500, "90": 7500},
        })

        self.assertEqual(cleaned["booking_mode"], "choose_duration")
        self.assertEqual(cleaned["allowed_durations_minutes"], [30, 60, 90])
        self.assertEqual(cleaned["price_by_duration"]["60"], 5500)

    def test_successful_empty_active_service_query_does_not_restore_fallbacks(self):
        class EmptyResponse:
            data = []

        class EmptyQuery:
            def select(self, *args, **kwargs):
                return self

            def eq(self, *args, **kwargs):
                return self

            def order(self, *args, **kwargs):
                return self

            def execute(self):
                return EmptyResponse()

        class EmptySupabase:
            def table(self, name):
                return EmptyQuery()

        self.chatbot._MASSAGE_SERVICE_CACHE["rows"] = None
        self.chatbot._MASSAGE_SERVICE_CACHE["loaded_at"] = 0

        with patch.object(self.chatbot, "supabase", EmptySupabase()):
            services, from_supabase = self.chatbot.load_massage_services()

        self.assertTrue(from_supabase)
        self.assertEqual(services, [])

    def test_duration_validation_uses_generic_service_metadata(self):
        with patch.object(
            self.chatbot,
            "find_massage_service",
            return_value=None,
        ), patch.object(
            self.chatbot,
            "authoritative_service_metadata",
            return_value={
                "allowed_durations_minutes": [45, 75],
                "fixed_duration_minutes": None,
            },
        ):
            self.assertEqual(
                self.chatbot.allowed_durations_for_treatment(
                    "Configured Service"
                ),
                {45, 75},
            )

    def test_pre_treatment_duration_is_preserved_after_variant_selection(self):
        history = [
            self.chatbot.ChatMessage(
                role="user",
                content="Can I book a 30mins back massage?",
            ),
            self.chatbot.ChatMessage(
                role="assistant",
                content="Which massage would you prefer?",
            ),
            self.chatbot.ChatMessage(
                role="user",
                content="Deep tissue please",
            ),
        ]

        def resolve_deep_tissue(state, message):
            result = dict(state)
            if "deep tissue" in message.lower():
                result["treatment"] = "Deep Tissue Massage"
            return result

        with patch.object(
            self.chatbot,
            "apply_structured_service_resolution",
            side_effect=resolve_deep_tissue,
        ), patch.object(
            self.chatbot,
            "hydrate_configured_service_defaults",
            side_effect=lambda state: state,
        ), patch.object(
            self.chatbot,
            "allowed_durations_for_treatment",
            return_value={30, 60, 90, 120},
        ):
            state = self.chatbot.apply_validated_state_patch(
                existing_state={"treatment": "Massage"},
                extractor_result=self.chatbot.empty_extractor_result(),
                latest_message="Deep tissue please",
                history=history,
            )

        self.assertEqual(state["treatment"], "Deep Tissue Massage")
        self.assertEqual(state["duration"], "30 minutes")
        self.assertEqual(
            self.chatbot.compute_next_required_detail(state, True),
            "preferred_date",
        )

    def test_explicit_invalid_duration_clears_stale_duration(self):
        with patch.object(
            self.chatbot,
            "apply_structured_service_resolution",
            side_effect=lambda state, message: {
                **state,
                "treatment": "Hot Stone Massage",
            },
        ), patch.object(
            self.chatbot,
            "hydrate_configured_service_defaults",
            side_effect=lambda state: state,
        ), patch.object(
            self.chatbot,
            "allowed_durations_for_treatment",
            return_value={60, 90, 120},
        ):
            state = self.chatbot.apply_validated_state_patch(
                existing_state={
                    "treatment": "Relaxing Massage",
                    "duration": "60 minutes",
                },
                extractor_result=self.chatbot.empty_extractor_result(),
                latest_message="Hot stone 30mins",
                history=[],
            )

        self.assertEqual(state["treatment"], "Hot Stone Massage")
        self.assertIsNone(state["duration"])
        self.assertEqual(
            self.chatbot.compute_next_required_detail(state, True),
            "duration",
        )

    def test_customer_reply_markdown_is_rendered_as_plain_text(self):
        reply = self.compose(
            (
                "### Massage options\n"
                "- **Relaxing Massage** - 30 min (£35)\n"
                "- `Hot Stone Massage` - 60 min (£55)"
            ),
            {},
            "Tell me about Relaxing Massage.",
            detail=None,
        )

        self.assertIn("Relaxing Massage - 30 min (£35)", reply)
        self.assertIn("Hot Stone Massage - 60 min (£55)", reply)
        self.assertNotIn("**", reply)
        self.assertNotIn("###", reply)
        self.assertNotIn("`", reply)


if __name__ == "__main__":
    unittest.main()
