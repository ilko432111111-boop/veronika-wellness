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

    def test_customer_reply_markdown_is_rendered_as_plain_text(self):
        reply = self.compose(
            (
                "### Massage options\n"
                "- **Relaxing Massage** - 30 min (£35)\n"
                "- `Hot Stone Massage` - 60 min (£55)"
            ),
            {},
            "Do you offer massages?",
            detail=None,
        )

        self.assertIn("Relaxing Massage - 30 min (£35)", reply)
        self.assertIn("Hot Stone Massage - 60 min (£55)", reply)
        self.assertNotIn("**", reply)
        self.assertNotIn("###", reply)
        self.assertNotIn("`", reply)


if __name__ == "__main__":
    unittest.main()
