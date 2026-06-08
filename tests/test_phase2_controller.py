import unittest
from contextlib import ExitStack
from unittest.mock import patch

from fastapi import BackgroundTasks

from test_smoke import load_chatbot_module


class CalendarGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def test_verified_calendar_requires_positive_duration(self):
        state = {
            "preferred_date": "2099-01-01",
            "preferred_time": "10:00",
            "duration": None,
        }

        with patch.object(
            self.chatbot,
            "check_requested_calendar_slot",
        ) as calendar_check:
            result = self.chatbot.verified_calendar_result_for_state(
                state,
                booking_flow_active=True,
                latest_message="Is that available?",
            )

        self.assertEqual(result["status"], "not_checked")
        calendar_check.assert_not_called()

    def test_direct_calendar_check_skips_hours_and_google_without_duration(self):
        state = {
            "preferred_date": "2099-01-01",
            "preferred_time": "10:00",
            "duration": None,
        }

        with patch.object(
            self.chatbot,
            "validate_slot_against_working_hours",
        ) as hours_check:
            with patch.object(
                self.chatbot,
                "query_google_freebusy",
            ) as google_check:
                result = self.chatbot.check_requested_calendar_slot(
                    state,
                    latest_message="Please book that time",
                )

        self.assertEqual(result["status"], "not_checked")
        self.assertIsNone(self.chatbot.build_requested_slot(state))
        hours_check.assert_not_called()
        google_check.assert_not_called()


class CanonicalSaveFailureTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    async def test_chat_stops_after_initial_canonical_save_failure(self):
        request = self.chatbot.ChatRequest(
            session_id="phase-2-save-failure",
            message="I would like to book",
            history=[],
        )
        background_tasks = BackgroundTasks()
        merged_state = {
            "treatment": "Relaxing Massage",
            "duration": "60 minutes",
            "preferred_date": "2099-01-01",
            "preferred_time": "10:00",
            "name": "Test Customer",
            "phone": "07000000000",
        }

        with patch.object(self.chatbot, "save_message"):
            with patch.object(
                self.chatbot,
                "get_existing_conversation",
                return_value={},
            ):
                with patch.object(
                    self.chatbot,
                    "build_authoritative_services_context",
                    return_value="",
                ):
                    with patch.object(
                        self.chatbot,
                        "load_business_documents_context",
                        return_value="",
                    ):
                        with patch.object(
                            self.chatbot,
                            "extract_hidden_state_patch",
                            return_value=self.chatbot.empty_extractor_result(),
                        ):
                            with patch.object(
                                self.chatbot,
                                "apply_validated_state_patch",
                                return_value=merged_state,
                            ):
                                with patch.object(
                                    self.chatbot,
                                    "add_manual_multi_treatment_note",
                                    return_value=merged_state,
                                ):
                                    with patch.object(
                                        self.chatbot,
                                        "has_booking_intent_state",
                                        return_value=True,
                                    ):
                                        with patch.object(
                                            self.chatbot,
                                            "apply_canonical_controller_state",
                                            return_value=merged_state,
                                        ):
                                            with patch.object(
                                                self.chatbot,
                                                "save_conversation_overview",
                                                return_value=False,
                                            ):
                                                with patch.object(
                                                    self.chatbot,
                                                    "verified_calendar_result_for_state",
                                                ) as calendar_check:
                                                    with patch.object(
                                                        self.chatbot,
                                                        "sync_simple_single_request_projection",
                                                    ) as projection:
                                                        with patch.object(
                                                            self.chatbot,
                                                            "compose_verified_customer_reply",
                                                        ) as responder:
                                                            response = await self.chatbot.chat(
                                                                request,
                                                                background_tasks,
                                                            )

        self.assertEqual(
            response["reply"],
            self.chatbot.CANONICAL_SAVE_FAILURE_REPLY,
        )
        calendar_check.assert_not_called()
        projection.assert_not_called()
        responder.assert_not_called()
        self.assertEqual(background_tasks.tasks, [])

    async def test_chat_stops_after_post_calendar_canonical_save_failure(self):
        request = self.chatbot.ChatRequest(
            session_id="phase-2-second-save-failure",
            message="I would like to book",
            history=[],
        )
        background_tasks = BackgroundTasks()
        merged_state = {
            "treatment": "Relaxing Massage",
            "duration": "60 minutes",
            "preferred_date": "2099-01-01",
            "preferred_time": "10:00",
        }

        with ExitStack() as stack:
            stack.enter_context(patch.object(self.chatbot, "save_message"))
            stack.enter_context(patch.object(
                self.chatbot,
                "get_existing_conversation",
                return_value={},
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
                "apply_validated_state_patch",
                return_value=merged_state,
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "add_manual_multi_treatment_note",
                return_value=merged_state,
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "has_booking_intent_state",
                return_value=True,
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "apply_canonical_controller_state",
                return_value=merged_state,
            ))
            save = stack.enter_context(patch.object(
                self.chatbot,
                "save_conversation_overview",
                side_effect=[True, False],
            ))
            stack.enter_context(patch.object(
                self.chatbot,
                "verified_calendar_result_for_state",
                return_value={"status": "not_checked"},
            ))
            projection = stack.enter_context(patch.object(
                self.chatbot,
                "sync_simple_single_request_projection",
            ))
            responder = stack.enter_context(patch.object(
                self.chatbot,
                "compose_verified_customer_reply",
            ))

            response = await self.chatbot.chat(request, background_tasks)

        self.assertEqual(
            response["reply"],
            self.chatbot.CANONICAL_SAVE_FAILURE_REPLY,
        )
        self.assertEqual(save.call_count, 2)
        projection.assert_not_called()
        responder.assert_not_called()
        self.assertEqual(background_tasks.tasks, [])


class ResponderSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def compose(self, state, calendar_result, next_required_detail):
        return self.chatbot.compose_verified_customer_reply(
            state=state,
            extractor_result=self.chatbot.empty_extractor_result(),
            calendar_result=calendar_result,
            next_required_detail=next_required_detail,
            business_context="",
            services_context="",
            latest_message="Hello",
            history=[],
        )

    def test_responder_prompt_contains_no_calendar_text_or_next_question(self):
        prompt = self.chatbot.build_responder_context(
            state={},
            extractor_result=self.chatbot.empty_extractor_result(),
            business_context="",
            services_context="",
            latest_message="Hello",
            history=[],
        )

        self.assertNotIn("VERIFIED CALENDAR CUSTOMER TEXT", prompt)
        self.assertNotIn("ONE ALLOWED NEXT QUESTION", prompt)

    def test_invented_calendar_wording_and_alternatives_cannot_reach_reply(self):
        state = {
            "preferred_date": "2099-01-01",
            "preferred_time": "10:00",
        }
        malicious_body = (
            "We have availability tomorrow. "
            "I checked the calendar. "
            "The slot is free. "
            "Try 15:30 instead. "
            "Around noon could also work. "
            "Your treatment request is noted. "
            "Which day works?"
        )
        calendar_result = {"status": "free"}
        expected_calendar_text = self.chatbot.safe_calendar_customer_text(
            state,
            calendar_result,
        )

        with patch.object(
            self.chatbot,
            "generate_natural_reply_body",
            return_value=malicious_body,
        ):
            reply = self.compose(
                state,
                calendar_result,
                "name_and_phone",
            )

        for forbidden in [
            "We have availability tomorrow",
            "I checked the calendar",
            "The slot is free",
            "15:30",
            "noon",
            "Which day works",
        ]:
            self.assertNotIn(forbidden, reply)

        self.assertIn("Your treatment request is noted.", reply)
        self.assertEqual(reply.count(expected_calendar_text), 1)
        self.assertEqual(reply.count("?"), 1)

    def test_final_reply_has_at_most_one_workflow_question_for_each_state(self):
        cases = [
            ("treatment", {}, None),
            ("duration", {"treatment": "Relaxing Massage"}, None),
            (
                "preferred_date",
                {"treatment": "Relaxing Massage", "duration": "60 minutes"},
                None,
            ),
            (
                "preferred_time",
                {
                    "treatment": "Relaxing Massage",
                    "duration": "60 minutes",
                    "preferred_date": "2099-01-01",
                },
                None,
            ),
            (
                "name_and_phone",
                {
                    "treatment": "Relaxing Massage",
                    "duration": "60 minutes",
                    "preferred_date": "2099-01-01",
                    "preferred_time": "10:00",
                },
                None,
            ),
            ("handoff", {}, None),
            (
                "service_variant",
                {"treatment": "Massage"},
                "Which massage would you like?",
            ),
        ]

        for next_detail, state, variant_question in cases:
            with self.subTest(next_detail=next_detail):
                with ExitStack() as stack:
                    stack.enter_context(patch.object(
                        self.chatbot,
                        "generate_natural_reply_body",
                        return_value="Noted. What time works? Which day works?",
                    ))

                    if variant_question:
                        stack.enter_context(patch.object(
                            self.chatbot,
                            "build_schedule_first_question",
                            return_value=variant_question,
                        ))

                    if next_detail == "duration":
                        stack.enter_context(patch.object(
                            self.chatbot,
                            "allowed_durations_for_treatment",
                            return_value={30, 60},
                        ))

                    reply = self.compose(
                        state,
                        {"status": "not_checked"},
                        next_detail,
                    )

                expected_count = 0 if next_detail == "handoff" else 1
                self.assertEqual(reply.count("?"), expected_count)

    def test_unavailable_alternatives_have_one_backend_question_only(self):
        calendar_result = {
            "status": "busy",
            "suggestions": [
                "Friday 1 January at 10:00",
                "Friday 1 January at 11:00",
            ],
        }

        with patch.object(
            self.chatbot,
            "generate_natural_reply_body",
        ) as responder:
            reply = self.compose(
                {},
                calendar_result,
                "name_and_phone",
            )

        responder.assert_not_called()
        self.assertEqual(reply.count("?"), 1)
        self.assertIn("Friday 1 January at 10:00", reply)
        self.assertIn("Friday 1 January at 11:00", reply)


if __name__ == "__main__":
    unittest.main()
