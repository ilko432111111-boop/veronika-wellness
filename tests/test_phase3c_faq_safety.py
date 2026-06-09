from datetime import datetime
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
