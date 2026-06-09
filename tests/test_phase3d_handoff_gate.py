import unittest
from contextlib import ExitStack
from unittest.mock import patch

from fastapi import BackgroundTasks

from test_smoke import load_chatbot_module


HANDOFF = (
    "Thank you, Ilko. Your requested slot currently appears free. "
    "Veronika will confirm the appointment with you shortly."
)


class Phase3DHandoffGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def setUp(self):
        self.state = {
            "conversation_mode": "booking_request",
            "status": "booking_request_complete",
            "treatment": "Relaxing Massage",
            "duration": "1 hour",
            "preferred_date": "2026-06-11",
            "preferred_time": "15:00",
            "slot_status": "provisional_free",
            "name": "Ilko",
            "phone": "07000000000",
            "verified_alternatives": [],
            "_canonical_save_succeeded": True,
        }
        self.calendar = {"status": "free"}

    def allowed(self, state=None, calendar=None):
        return self.chatbot.handoff_is_allowed(
            state or self.state,
            calendar or self.calendar,
        )

    def test_fully_eligible_state_is_allowed(self):
        self.assertTrue(self.allowed())

    def test_missing_required_fields_block_handoff(self):
        cases = [
            ("treatment", None),
            ("duration", None),
            ("preferred_date", None),
            ("preferred_time", None),
            ("name", None),
            ("phone", None),
        ]

        for field, value in cases:
            with self.subTest(field=field):
                self.assertFalse(self.allowed({**self.state, field: value}))

    def test_missing_variant_blocks_handoff(self):
        for treatment in [
            "Massage",
            "Ultrasound",
            "Microneedling",
            "Facial",
            "Dermal Filler",
        ]:
            with self.subTest(treatment=treatment):
                self.assertFalse(self.allowed({
                    **self.state,
                    "treatment": treatment,
                }))

    def test_noncanonical_treatment_blocks_handoff(self):
        self.assertFalse(self.allowed({
            **self.state,
            "treatment": "Made Up Service",
        }))

    def test_unverified_or_unavailable_slot_blocks_handoff(self):
        for slot_status, calendar_status in [
            ("not_checked", "not_checked"),
            ("unknown", "unknown"),
            ("unavailable", "busy"),
            ("provisional_free", "unknown"),
        ]:
            with self.subTest(
                slot_status=slot_status,
                calendar_status=calendar_status,
            ):
                self.assertFalse(self.allowed(
                    {**self.state, "slot_status": slot_status},
                    {"status": calendar_status},
                ))

    def test_pending_alternatives_and_failed_save_block_handoff(self):
        alternative = {
            "date": "2026-06-12",
            "time": "09:00",
            "label": "Friday 12 June at 09:00",
            "duration_minutes": 60,
        }
        self.assertFalse(self.allowed({
            **self.state,
            "verified_alternatives": [alternative],
        }))
        self.assertFalse(self.allowed({
            **self.state,
            "_canonical_save_succeeded": False,
        }))


class Phase3DReplyCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def setUp(self):
        self.complete_state = {
            "conversation_mode": "booking_request",
            "status": "booking_request_complete",
            "treatment": "Relaxing Massage",
            "duration": "1 hour",
            "preferred_date": "2026-06-11",
            "preferred_time": "15:00",
            "slot_status": "provisional_free",
            "name": "Ilko",
            "phone": "07000000000",
            "verified_alternatives": [],
            "_canonical_save_succeeded": True,
        }

    def compose(
        self,
        body,
        state,
        calendar,
        detail,
        message="Thanks",
        history=None,
    ):
        with patch.object(
            self.chatbot,
            "generate_natural_reply_body",
            return_value=body,
        ):
            return self.chatbot.compose_verified_customer_reply(
                state=state,
                extractor_result=self.chatbot.empty_extractor_result(),
                calendar_result=calendar,
                next_required_detail=detail,
                business_context="",
                services_context="",
                latest_message=message,
                history=history or [],
            )

    def test_booking_intent_only_has_no_handoff_wording(self):
        reply = self.compose(
            (
                "Of course. I've let Veronika know and she'll be in touch "
                "shortly."
            ),
            {"conversation_mode": "booking_request"},
            {"status": "not_checked"},
            "treatment",
            "hey id like to book",
        )

        self.assertEqual(reply, "Of course.\n\nWhich treatment are you interested in?")
        self.assertNotIn("Veronika", reply)
        self.assertEqual(reply.count("?"), 1)

    def test_verified_slot_missing_contacts_asks_only_for_contacts(self):
        reply = self.compose(
            "",
            {
                **self.complete_state,
                "name": None,
                "phone": None,
                "_canonical_save_succeeded": True,
            },
            {"status": "free"},
            "name_and_phone",
        )

        self.assertEqual(
            reply,
            "Thursday 11 June at 15:00 currently appears free.\n\n"
            "Could I take your name and phone number, please?",
        )
        self.assertNotIn("Veronika", reply)
        self.assertEqual(reply.count("?"), 1)

    def test_fully_eligible_state_appends_controlled_handoff_once(self):
        first = self.compose(
            "",
            self.complete_state,
            {"status": "free"},
            "handoff",
        )
        second = self.compose(
            "",
            self.complete_state,
            {"status": "free"},
            "handoff",
            history=[self.chatbot.ChatMessage(role="assistant", content=first)],
        )

        self.assertEqual(first, HANDOFF)
        self.assertEqual(first.count("Veronika will confirm"), 1)
        self.assertNotIn("Veronika will confirm", second)

    def test_model_written_handoff_concepts_are_stripped(self):
        phrases = [
            "I'll let Veronika know.",
            "I’ll let Veronika know.",
            "I have let Veronika know.",
            "We'll pass your request on.",
            "We’ll pass your request on.",
            "Veronika will be in touch shortly.",
            "Veronika will contact you.",
            "Veronika will review the details.",
            "Veronika will get back to you.",
            "The therapist will confirm.",
            "I've noted your booking.",
            "Your booking request has been sent.",
            "Looking forward to welcoming you.",
        ]

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                reply = self.compose(
                    phrase,
                    {"conversation_mode": "booking_request"},
                    {"status": "not_checked"},
                    "treatment",
                    "hey id like to book",
                )
                self.assertEqual(reply, "Which treatment are you interested in?")

    def test_unknown_or_unavailable_slot_never_adds_handoff(self):
        for slot_status, calendar_status in [
            ("unknown", "unknown"),
            ("unavailable", "busy"),
        ]:
            with self.subTest(slot_status=slot_status):
                reply = self.compose(
                    "",
                    {**self.complete_state, "slot_status": slot_status},
                    {"status": calendar_status},
                    "handoff",
                )
                self.assertNotIn("Veronika", reply)
                self.assertNotIn("confirm", reply.lower())

    def test_empty_active_body_uses_next_question_and_never_thanks(self):
        reply = self.compose(
            "",
            {"conversation_mode": "booking_request"},
            {"status": "not_checked"},
            "treatment",
        )

        self.assertEqual(reply, "Which treatment are you interested in?")

    def test_informational_replies_remain_natural(self):
        cases = [
            ("Where are you based?", "We're based at 25 Albion Place, Leeds, LS1 6JS."),
            ("How much is EMS?", "EMS costs £200 for a one-hour session."),
            (
                "What treatments do you have?",
                "We offer massage, EMS, ultrasound, facials, and fillers.",
            ),
        ]

        for message, body in cases:
            with self.subTest(message=message):
                self.assertEqual(
                    self.compose(body, {}, {"status": "not_requested"}, None, message),
                    body,
                )

    def test_final_replies_have_at_most_one_workflow_question(self):
        for detail, state in [
            ("treatment", {"conversation_mode": "booking_request"}),
            (
                "duration",
                {
                    "conversation_mode": "booking_request",
                    "treatment": "Relaxing Massage",
                },
            ),
            (
                "name_and_phone",
                {
                    **self.complete_state,
                    "name": None,
                    "phone": None,
                },
            ),
        ]:
            with self.subTest(detail=detail):
                reply = self.compose(
                    "Which treatment? What time?",
                    state,
                    {"status": "not_checked"},
                    detail,
                )
                self.assertLessEqual(reply.count("?"), 1)

    def test_prompt_explicitly_forbids_handoff_concepts(self):
        prompt = self.chatbot.build_responder_context(
            state={},
            extractor_result=self.chatbot.empty_extractor_result(),
            business_context="",
            services_context="",
            latest_message="hey id like to book",
            history=[],
        )

        self.assertIn("Never mention handoff", prompt)
        self.assertIn("Python will append final handoff wording if eligible", prompt)


class Phase3DSaveFailureTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    async def test_canonical_save_failure_reply_has_no_handoff_wording(self):
        request = self.chatbot.ChatRequest(
            session_id="phase-3d-save-failure",
            message="hey id like to book",
            history=[],
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(self.chatbot, "save_message"))
            stack.enter_context(patch.object(
                self.chatbot, "get_existing_conversation", return_value={}
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
                self.chatbot, "save_conversation_overview", return_value=False
            ))

            response = await self.chatbot.chat(request, BackgroundTasks())

        self.assertNotIn("Veronika", response["reply"])
        self.assertNotIn("confirm", response["reply"].lower())


if __name__ == "__main__":
    unittest.main()
