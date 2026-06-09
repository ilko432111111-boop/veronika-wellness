import unittest
from contextlib import ExitStack
from unittest.mock import patch

from fastapi import BackgroundTasks

from test_smoke import load_chatbot_module


class VerifiedAlternativeSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def setUp(self):
        self.alternatives = [
            {
                "date": "2026-06-11",
                "time": "09:00",
                "label": "Thursday 11 June at 09:00",
                "duration_minutes": 60,
            },
            {
                "date": "2026-06-11",
                "time": "09:30",
                "label": "Thursday 11 June at 09:30",
                "duration_minutes": 60,
            },
            {
                "date": "2026-06-11",
                "time": "10:00",
                "label": "Thursday 11 June at 10:00",
                "duration_minutes": 60,
            },
        ]
        self.state = {
            "treatment": "Relaxing Massage",
            "duration": "1 hour",
            "preferred_date": "2026-06-10",
            "preferred_time": "14:00",
            "name": "Test Customer",
            "phone": "07000000000",
            "notes": "Keep this note",
            "verified_alternatives": self.alternatives,
        }

    def selected(self, message):
        resolution = self.chatbot.resolve_verified_alternative_reply(
            self.state,
            message,
        )
        self.assertEqual(resolution["status"], "selected")
        return resolution["matches"][0]

    def test_yes_time_resolves_unique_verified_alternative(self):
        for message in ["Yes, 10am", "10 am", "10:00", "10 works"]:
            with self.subTest(message=message):
                selected = self.selected(message)
                self.assertEqual(selected["date"], "2026-06-11")
                self.assertEqual(selected["time"], "10:00")

        self.assertEqual(self.selected("the 9:30 slot please")["time"], "09:30")

    def test_last_one_resolves_final_alternative(self):
        self.assertEqual(self.selected("The last one")["time"], "10:00")

    def test_second_option_resolves_correctly(self):
        self.assertEqual(self.selected("Second option")["time"], "09:30")

    def test_weekday_and_time_resolve_correctly(self):
        self.assertEqual(self.selected("Thursday 10am")["time"], "10:00")

    def test_date_and_time_resolve_correctly(self):
        self.assertEqual(self.selected("11 June 10am")["time"], "10:00")

    def test_ambiguous_time_does_not_guess(self):
        state = dict(self.state)
        state["verified_alternatives"] = [
            self.alternatives[-1],
            {
                "date": "2026-06-12",
                "time": "10:00",
                "label": "Friday 12 June at 10:00",
                "duration_minutes": 60,
            },
        ]

        resolution = self.chatbot.resolve_verified_alternative_reply(
            state,
            "10am",
        )
        retained = self.chatbot.apply_verified_alternative_resolution(
            state,
            state,
            resolution,
        )
        reply = self.chatbot.build_calendar_alternative_reply({
            "status": "alternative_ambiguous",
            "suggestions": [
                option["label"]
                for option in retained["verified_alternatives"]
            ],
        })

        self.assertEqual(resolution["status"], "ambiguous")
        self.assertEqual(
            retained["verified_alternatives"],
            state["verified_alternatives"],
        )
        self.assertEqual(reply.count("?"), 1)
        self.assertIn("more than one verified option", reply)

    def test_unknown_time_is_not_accepted_and_alternatives_remain(self):
        resolution = self.chatbot.resolve_verified_alternative_reply(
            self.state,
            "4pm",
        )
        retained = self.chatbot.apply_verified_alternative_resolution(
            self.state,
            self.state,
            resolution,
        )
        reply = self.chatbot.build_calendar_alternative_reply({
            "status": "alternative_no_match",
            "suggestions": [
                option["label"]
                for option in retained["verified_alternatives"]
            ],
        })

        self.assertEqual(resolution["status"], "no_match")
        self.assertEqual(retained["preferred_time"], "14:00")
        self.assertEqual(retained["verified_alternatives"], self.alternatives)
        self.assertEqual(reply.count("?"), 1)
        self.assertIn("not one of the current verified options", reply)

    def test_treatment_or_duration_change_clears_stale_alternatives(self):
        for changed in [
            {**self.state, "treatment": "Deep Facial Cleanse"},
            {**self.state, "duration": "90 minutes"},
        ]:
            with self.subTest(changed=changed):
                result = self.chatbot.apply_verified_alternative_resolution(
                    changed,
                    self.state,
                    {"status": "not_applicable", "matches": []},
                )
                self.assertEqual(result["verified_alternatives"], [])

    def test_explicit_new_requested_slot_clears_stale_alternatives(self):
        resolution = self.chatbot.resolve_verified_alternative_reply(
            self.state,
            "Actually, I would like Friday at 4pm instead",
        )
        changed = {
            **self.state,
            "preferred_date": "2026-06-12",
            "preferred_time": "16:00",
        }
        result = self.chatbot.apply_verified_alternative_resolution(
            changed,
            self.state,
            resolution,
        )

        self.assertEqual(resolution["status"], "explicit_replacement")
        self.assertEqual(result["verified_alternatives"], [])

    def test_backend_suggestions_are_persisted_as_structured_alternatives(self):
        controlled = self.chatbot.apply_canonical_controller_state(
            {
                **self.state,
                "name": None,
                "phone": None,
                "verified_alternatives": [],
            },
            booking_flow_active=True,
            calendar_result={
                "status": "busy",
                "suggestions": [
                    "Thursday 11 June at 09:00",
                    "Thursday 11 June at 10:00",
                ],
            },
        )

        self.assertEqual(
            controlled["verified_alternatives"][1],
            self.alternatives[2],
        )

    def test_relative_suggestions_remain_pending_until_customer_selects(self):
        structured = self.chatbot.structured_verified_alternatives(
            ["tomorrow at 10:00"],
            60,
        )
        self.assertEqual(len(structured), 1)
        self.assertEqual(structured[0]["time"], "10:00")
        self.assertEqual(structured[0]["duration_minutes"], 60)

        controlled = self.chatbot.apply_canonical_controller_state(
            {
                **self.state,
                "verified_alternatives": structured,
            },
            booking_flow_active=True,
            calendar_result={"status": "not_checked"},
        )
        self.assertEqual(
            controlled["next_required_detail"],
            "verified_alternative",
        )
        self.assertEqual(
            controlled["verified_alternatives"],
            structured,
        )

    def test_successful_selection_clears_alternatives_and_preserves_contact(self):
        resolution = self.chatbot.resolve_verified_alternative_reply(
            self.state,
            "Yes, 10am",
        )
        result = self.chatbot.apply_verified_alternative_resolution(
            self.state,
            self.state,
            resolution,
        )

        self.assertEqual(result["preferred_date"], "2026-06-11")
        self.assertEqual(result["preferred_time"], "10:00")
        self.assertEqual(result["verified_alternatives"], [])
        self.assertEqual(result["name"], "Test Customer")
        self.assertEqual(result["phone"], "07000000000")
        self.assertEqual(result["notes"], "Keep this note")

    def test_so_am_i_booked_uses_manual_confirmation_wording(self):
        with patch.object(
            self.chatbot,
            "generate_natural_reply_body",
        ) as responder:
            reply = self.chatbot.compose_verified_customer_reply(
                state={**self.state, "name": None, "phone": None},
                extractor_result=self.chatbot.empty_extractor_result(),
                calendar_result={"status": "not_checked"},
                next_required_detail="name_and_phone",
                business_context="",
                services_context="",
                latest_message="So am I booked?",
                history=[],
            )

        responder.assert_not_called()
        self.assertIn("appointment is not confirmed yet", reply)
        self.assertNotIn("you are booked", reply.lower())
        self.assertEqual(reply.count("?"), 1)

        complete_reply = self.chatbot.compose_verified_customer_reply(
            state=self.state,
            extractor_result=self.chatbot.empty_extractor_result(),
            calendar_result={"status": "not_checked"},
            next_required_detail="handoff",
            business_context="",
            services_context="",
            latest_message="So am I booked?",
            history=[],
        )
        self.assertEqual(
            complete_reply,
            "Your request has been noted, but the appointment is not "
            "confirmed yet.",
        )
        self.assertEqual(complete_reply.count("?"), 0)


class AlternativeSelectionControllerTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    async def test_yes_time_saves_selected_slot_rechecks_and_replies_once(self):
        alternatives = [
            {
                "date": "2026-06-11",
                "time": "09:00",
                "label": "Thursday 11 June at 09:00",
                "duration_minutes": 60,
            },
            {
                "date": "2026-06-11",
                "time": "09:30",
                "label": "Thursday 11 June at 09:30",
                "duration_minutes": 60,
            },
            {
                "date": "2026-06-11",
                "time": "10:00",
                "label": "Thursday 11 June at 10:00",
                "duration_minutes": 60,
            },
        ]
        current = {
            "treatment": "Relaxing Massage",
            "duration": "1 hour",
            "preferred_date": "2026-06-10",
            "preferred_time": "14:00",
            "name": None,
            "phone": None,
            "verified_alternatives": alternatives,
            "conversation_mode": "booking_request",
        }
        saved_states = []
        request = self.chatbot.ChatRequest(
            session_id="phase-3b-selection",
            message="Yes, 10am",
            history=[],
        )

        def save_state(session_id, lead_data, source):
            saved_states.append(dict(lead_data))
            return True

        with ExitStack() as stack:
            stack.enter_context(patch.object(self.chatbot, "save_message"))
            stack.enter_context(patch.object(
                self.chatbot,
                "get_existing_conversation",
                return_value=current,
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "load_recent_messages",
                return_value=[],
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "build_authoritative_services_context",
                return_value="",
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "load_business_documents_context",
                return_value="",
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "extract_hidden_state_patch",
                return_value=self.chatbot.empty_extractor_result(),
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "apply_structured_service_resolution",
                side_effect=lambda state, message: state,
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "hydrate_configured_service_defaults",
                side_effect=lambda state: state,
            ))
            calendar_check = stack.enter_context(patch.object(
                self.chatbot,
                "verified_calendar_result_for_state",
                return_value={"status": "free"},
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "save_conversation_overview",
                side_effect=save_state,
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "sync_simple_single_request_projection",
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "generate_natural_reply_body",
                return_value="Thanks.",
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "send_booking_notification",
            ))

            response = await self.chatbot.chat(request, BackgroundTasks())

        calendar_state = calendar_check.call_args.args[0]
        self.assertEqual(calendar_state["preferred_date"], "2026-06-11")
        self.assertEqual(calendar_state["preferred_time"], "10:00")
        self.assertEqual(saved_states[-1]["verified_alternatives"], [])
        self.assertIn(
            "Thursday 11 June at 10:00 currently appears free.",
            response["reply"],
        )
        self.assertTrue(
            response["reply"].startswith(
                "Thursday 11 June at 10:00 currently appears free."
            )
        )
        self.assertIn("Could I take your name and phone number", response["reply"])
        self.assertNotIn("booked", response["reply"].lower())
        self.assertEqual(response["reply"].count("?"), 1)


if __name__ == "__main__":
    unittest.main()
