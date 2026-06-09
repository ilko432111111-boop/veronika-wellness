from datetime import datetime
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from fastapi import BackgroundTasks

from test_smoke import load_chatbot_module


class Phase3CFaqSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def compose(self, body, message, detail=None, state=None):
        with patch.object(
            self.chatbot,
            "generate_natural_reply_body",
            return_value=body,
        ):
            return self.chatbot.compose_verified_customer_reply(
                state=state or {},
                extractor_result=self.chatbot.empty_extractor_result(),
                calendar_result={"status": "not_checked"},
                next_required_detail=detail,
                business_context="",
                services_context="",
                latest_message=message,
                history=[],
            )

    def test_location_answers_survive_sanitization(self):
        cases = [
            (
                "Where are you based?",
                "We're based in Leeds city centre at 25 Albion Place, LS1 6JS.",
            ),
            (
                "Where is your place?",
                "Our place is at 25 Albion Place, Leeds, LS1 6JS.",
            ),
            (
                "Are you in Leeds city centre?",
                "Yes, we are based in Leeds city centre.",
            ),
        ]

        for message, answer in cases:
            with self.subTest(message=message):
                self.assertEqual(self.compose(answer, message), answer)

    def test_authoritative_numeric_faq_answers_survive_sanitization(self):
        answers = [
            "Our address is 25 Albion Place, LS1 6JS.",
            "EMS costs £200 for a one-hour session.",
            "Ultrasound lasts 45 minutes.",
            "We are open Monday from 09:00 to 17:00.",
            "You can call us on 07943319617.",
        ]

        for answer in answers:
            with self.subTest(answer=answer):
                self.assertEqual(
                    self.chatbot.sanitise_natural_responder_body(answer),
                    answer,
                )
                self.assertEqual(
                    self.chatbot.strip_unverified_calendar_claims(
                        answer,
                        {"status": "not_checked"},
                    ),
                    answer,
                )

    def test_invented_availability_and_confirmation_claims_are_removed(self):
        unsafe_answers = [
            "I checked the calendar and Tuesday at 1pm is available.",
            "The next available slot is Wednesday at 2pm.",
            "Your appointment is confirmed for tomorrow at 1pm.",
            "I have booked you in for today at 1pm.",
        ]

        for answer in unsafe_answers:
            with self.subTest(answer=answer):
                self.assertEqual(
                    self.chatbot.sanitise_natural_responder_body(answer),
                    "",
                )

    def test_empty_and_punctuation_only_bodies_use_meaningful_fallback(self):
        expected = (
            "I'm sorry, I do not have that information available. "
            "Veronika can confirm it for you."
        )

        for body in ["", ")", "...", "Thanks."]:
            with self.subTest(body=body):
                self.assertEqual(
                    self.compose(body, "Do you have parking?"),
                    expected,
                )

    def test_empty_location_body_uses_authoritative_business_fallback(self):
        reply = self.compose("", "Where are you based?")

        self.assertIn("25 Albion Place", reply)
        self.assertIn("LS1 6JS", reply)
        self.assertNotEqual(reply, "Thanks.")

    def test_active_booking_flow_keeps_side_answer_and_one_workflow_question(self):
        reply = self.compose(
            (
                "EMS costs £200 for a one-hour session. "
                "Which treatment are you interested in? "
                "What time would suit you?"
            ),
            "How much is EMS?",
            detail="treatment",
        )

        self.assertIn("EMS costs £200 for a one-hour session.", reply)
        self.assertEqual(
            reply.count("Which treatment are you interested in?"),
            1,
        )
        self.assertNotIn("What time would suit you?", reply)
        self.assertEqual(reply.count("?"), 1)


class Phase3CTreatmentBeforeCalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def test_date_and_time_are_saved_before_treatment_without_calendar_call(self):
        extractor = self.chatbot.empty_extractor_result()
        extractor["intent"] = "booking_request"
        extractor["state_patch"] = {
            "preferred_date_expression": "today",
            "preferred_time": "13:00",
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
                state = self.chatbot.apply_validated_state_patch(
                    existing_state={},
                    extractor_result=extractor,
                    latest_message="Can I come today at 1pm?",
                    history=[],
                )
                controlled = self.chatbot.apply_canonical_controller_state(
                    state,
                    booking_flow_active=True,
                    calendar_result={"status": "not_checked"},
                )

        with patch.object(
            self.chatbot,
            "check_requested_calendar_slot",
        ) as calendar_check:
            result = self.chatbot.verified_calendar_result_for_state(
                {
                    **controlled,
                    "duration": "1 hour",
                },
                booking_flow_active=True,
                latest_message="Can I come today at 1pm?",
            )

        self.assertEqual(
            state["preferred_date"],
            datetime.now(self.chatbot.BUSINESS_TIMEZONE).date().isoformat(),
        )
        self.assertEqual(state["preferred_time"], "13:00")
        self.assertEqual(controlled["next_required_detail"], "treatment")
        self.assertEqual(
            self.chatbot.render_next_question(
                controlled,
                controlled["next_required_detail"],
            ),
            "Which treatment are you interested in?",
        )
        with patch.object(
            self.chatbot,
            "generate_natural_reply_body",
            return_value="",
        ):
            reply = self.chatbot.compose_verified_customer_reply(
                state=controlled,
                extractor_result=extractor,
                calendar_result=result,
                next_required_detail=controlled["next_required_detail"],
                business_context="",
                services_context="",
                latest_message="Can I come today at 1pm?",
                history=[],
            )

        self.assertEqual(reply, "Which treatment are you interested in?")
        self.assertEqual(result["status"], "not_checked")
        calendar_check.assert_not_called()


class Phase3CAlternativeLifecycleTests(unittest.TestCase):
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
                "time": "11:00",
                "label": "Thursday 11 June at 11:00",
                "duration_minutes": 60,
            },
            {
                "date": "2026-06-11",
                "time": "11:30",
                "label": "Thursday 11 June at 11:30",
                "duration_minutes": 60,
            },
        ]
        self.state = {
            "treatment": "Relaxing Massage",
            "duration": "1 hour",
            "preferred_date": "2026-06-10",
            "preferred_time": "09:00",
            "name": None,
            "phone": None,
            "notes": "Keep this note",
            "slot_status": "unavailable",
            "verified_alternatives": self.alternatives,
            "conversation_mode": "booking_request",
        }

    def test_short_bare_hour_replies_resolve_unique_verified_alternative(self):
        for message in ["9?", "9", "yes 9", "the 9 one"]:
            with self.subTest(message=message):
                resolution = self.chatbot.resolve_verified_alternative_reply(
                    self.state,
                    message,
                )
                self.assertEqual(resolution["status"], "selected")
                self.assertEqual(resolution["matches"][0]["time"], "09:00")

    def test_ambiguous_bare_hour_retains_verified_alternatives(self):
        state = {
            **self.state,
            "verified_alternatives": [
                self.alternatives[0],
                {
                    **self.alternatives[0],
                    "date": "2026-06-12",
                    "label": "Friday 12 June at 09:00",
                },
            ],
        }
        resolution = self.chatbot.resolve_verified_alternative_reply(
            state,
            "9?",
        )
        retained = self.chatbot.apply_verified_alternative_resolution(
            state,
            state,
            resolution,
        )

        self.assertEqual(resolution["status"], "ambiguous")
        self.assertEqual(retained["verified_alternatives"], state["verified_alternatives"])

    def test_selection_clears_stale_slot_state_and_preserves_other_fields(self):
        resolution = self.chatbot.resolve_verified_alternative_reply(
            self.state,
            "9?",
        )
        selected = self.chatbot.apply_verified_alternative_resolution(
            self.state,
            self.state,
            resolution,
        )

        self.assertEqual(selected["preferred_date"], "2026-06-11")
        self.assertEqual(selected["preferred_time"], "09:00")
        self.assertEqual(selected["verified_alternatives"], [])
        self.assertEqual(selected["slot_status"], "not_checked")
        self.assertEqual(selected["notes"], "Keep this note")

    def test_pending_alternatives_remain_the_next_required_workflow_action(self):
        complete_original_fields = {
            **self.state,
            "name": "Test Customer",
            "phone": "07000000000",
        }
        controlled = self.chatbot.apply_canonical_controller_state(
            complete_original_fields,
            booking_flow_active=True,
            calendar_result={"status": "not_checked"},
        )

        self.assertEqual(controlled["next_required_detail"], "verified_alternative")
        self.assertEqual(controlled["verified_alternatives"], self.alternatives)

    def test_contact_details_after_selection_keep_selected_slot(self):
        resolution = self.chatbot.resolve_verified_alternative_reply(
            self.state,
            "9?",
        )
        selected = self.chatbot.apply_verified_alternative_resolution(
            self.state,
            self.state,
            resolution,
        )
        selected.update({"name": "Test Customer", "phone": "07000000000"})
        after_contact = self.chatbot.apply_verified_alternative_resolution(
            selected,
            selected,
            {"status": "not_applicable", "matches": []},
        )

        self.assertEqual(after_contact["preferred_date"], "2026-06-11")
        self.assertEqual(after_contact["preferred_time"], "09:00")
        self.assertEqual(after_contact["verified_alternatives"], [])


class Phase3CActiveFlowReplyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def compose(
        self,
        body,
        state,
        detail,
        message="thanks",
        history=None,
        calendar=None,
    ):
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
                history=history or [],
            )

    def test_active_flow_never_returns_only_thanks(self):
        state = {"conversation_mode": "booking_request", "treatment": "Massage"}

        for body in ["", "Thanks."]:
            with self.subTest(body=body):
                reply = self.compose(body, state, "duration")
                self.assertNotEqual(reply, "Thanks.")
                self.assertEqual(reply.count("?"), 1)

    def test_premature_handoff_boilerplate_is_suppressed(self):
        body = (
            "We'll pass your request on to Veronika. "
            "Veronika will confirm the appointment shortly. "
            "Relaxing massage sounds good."
        )
        reply = self.compose(
            body,
            {"conversation_mode": "booking_request", "treatment": "Relaxing Massage"},
            "duration",
        )

        self.assertNotIn("pass your request", reply.lower())
        self.assertNotIn("confirm the appointment", reply.lower())
        self.assertIn("Relaxing massage sounds good.", reply)
        self.assertEqual(reply.count("?"), 1)

    def test_final_handoff_appears_once_and_is_not_repeated(self):
        state = {
            "conversation_mode": "booking_request",
            "status": "booking_request_complete",
            "treatment": "Relaxing Massage",
            "duration": "1 hour",
            "preferred_date": "2026-06-11",
            "preferred_time": "09:00",
            "name": "Test Customer",
            "phone": "07000000000",
            "verified_alternatives": [],
            "slot_status": "provisional_free",
            "_canonical_save_succeeded": True,
        }
        handoff = (
            "Thank you, Test Customer. Your requested slot currently appears "
            "free. Veronika will confirm the appointment with you shortly."
        )
        first = self.compose(
            "",
            state,
            "handoff",
            calendar={"status": "free"},
        )
        second = self.compose(
            "",
            state,
            "handoff",
            history=[self.chatbot.ChatMessage(role="assistant", content=first)],
            calendar={"status": "free"},
        )

        self.assertEqual(first, handoff)
        self.assertNotIn(handoff, second)
        self.assertNotEqual(second, "Thanks.")


class Phase3CSelectedSlotControllerTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    async def test_selected_slot_reaches_projection_and_old_alternatives_do_not_return(self):
        alternatives = [
            {
                "date": "2026-06-11",
                "time": "09:00",
                "label": "Thursday 11 June at 09:00",
                "duration_minutes": 60,
            },
        ]
        current = {
            "treatment": "Relaxing Massage",
            "duration": "1 hour",
            "preferred_date": "2026-06-10",
            "preferred_time": "09:00",
            "name": None,
            "phone": None,
            "slot_status": "unavailable",
            "verified_alternatives": alternatives,
            "conversation_mode": "booking_request",
        }
        projected = []
        request = self.chatbot.ChatRequest(
            session_id="phase-3c-short-selection",
            message="9?",
            history=[],
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(self.chatbot, "save_message"))
            stack.enter_context(patch.object(
                self.chatbot, "get_existing_conversation", return_value=current
            ))
            stack.enter_context(patch.object(
                self.chatbot, "load_recent_messages", return_value=[]
            ))
            stack.enter_context(patch.object(
                self.chatbot, "build_authoritative_services_context", return_value=""
            ))
            stack.enter_context(patch.object(
                self.chatbot, "load_business_documents_context", return_value=""
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
                self.chatbot, "save_conversation_overview", return_value=True
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "sync_simple_single_request_projection",
                side_effect=lambda session_id, state, booking_flow_active: projected.append(
                    dict(state)
                ),
            ))
            stack.enter_context(patch.object(
                self.chatbot, "generate_natural_reply_body", return_value=""
            ))
            stack.enter_context(patch.object(self.chatbot, "send_booking_notification"))

            response = await self.chatbot.chat(request, BackgroundTasks())

        checked_state = calendar_check.call_args.args[0]
        self.assertEqual(checked_state["preferred_date"], "2026-06-11")
        self.assertEqual(checked_state["preferred_time"], "09:00")
        self.assertEqual(checked_state["verified_alternatives"], [])
        self.assertEqual(projected[-1]["preferred_date"], "2026-06-11")
        self.assertEqual(projected[-1]["preferred_time"], "09:00")
        self.assertNotIn("next available options", response["reply"].lower())
        self.assertIn("Thursday 11 June at 09:00 currently appears free.", response["reply"])
        self.assertEqual(response["reply"].count("?"), 1)


if __name__ == "__main__":
    unittest.main()
