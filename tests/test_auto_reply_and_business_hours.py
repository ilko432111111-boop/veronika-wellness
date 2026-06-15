import unittest
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, patch

from fastapi import BackgroundTasks

from test_smoke import load_chatbot_module


class AutoReplyPauseTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    async def test_paused_website_chat_saves_inbound_without_running_responder(self):
        request = self.chatbot.ChatRequest(
            session_id="paused-test",
            message="Can I book a massage?",
            source="website",
        )

        with patch.object(self.chatbot, "auto_reply_enabled", return_value=False):
            with patch.object(self.chatbot, "save_message") as save_message:
                with patch.object(
                    self.chatbot,
                    "get_existing_conversation",
                    return_value={},
                ):
                    with patch.object(
                        self.chatbot,
                        "prepare_conversation_for_resume",
                        return_value={"status": "new_chat"},
                    ):
                        with patch.object(
                            self.chatbot,
                            "save_conversation_overview",
                            return_value=True,
                        ) as save_overview:
                            with patch.object(
                                self.chatbot,
                                "extract_hidden_state_patch",
                                new=AsyncMock(),
                            ) as extractor:
                                response = await self.chatbot.chat(
                                    request,
                                    BackgroundTasks(),
                                )

        self.assertFalse(response["auto_reply_enabled"])
        self.assertIn("message has been saved", response["reply"])
        save_message.assert_called_once()
        save_overview.assert_called_once()
        extractor.assert_not_called()

    async def test_paused_instagram_chat_returns_no_reply(self):
        request = self.chatbot.ChatRequest(
            session_id="paused-instagram-test",
            message="Hello",
            source="instagram",
        )

        with patch.object(self.chatbot, "auto_reply_enabled", return_value=False):
            with patch.object(self.chatbot, "save_message"):
                with patch.object(self.chatbot, "get_existing_conversation", return_value={}):
                    with patch.object(
                        self.chatbot,
                        "prepare_conversation_for_resume",
                        return_value={"status": "new_chat"},
                    ):
                        with patch.object(
                            self.chatbot,
                            "save_conversation_overview",
                            return_value=True,
                        ):
                            response = await self.chatbot.chat(
                                request,
                                BackgroundTasks(),
                            )

        self.assertEqual(response["reply"], "")


class BusinessHoursSourceOfTruthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def test_closed_day_is_rejected(self):
        monday = date(2026, 6, 15)
        hours = {day: None for day in range(7)}

        with patch.object(
            self.chatbot,
            "load_business_hours_with_status",
            return_value=(hours, True),
        ):
            start = datetime(2026, 6, 15, 11, tzinfo=self.chatbot.BUSINESS_TIMEZONE)
            result = self.chatbot.validate_slot_against_working_hours(
                start,
                start + timedelta(hours=1),
                60,
            )

        self.assertEqual(monday.weekday(), 0)
        self.assertEqual(result["status"], "closed_day")

    def test_time_after_admin_closing_time_is_rejected(self):
        hours = {day: None for day in range(7)}
        hours[0] = ((10, 0), (16, 0))

        with patch.object(
            self.chatbot,
            "load_business_hours_with_status",
            return_value=(hours, True),
        ):
            start = datetime(2026, 6, 15, 17, tzinfo=self.chatbot.BUSINESS_TIMEZONE)
            result = self.chatbot.validate_slot_against_working_hours(
                start,
                start + timedelta(hours=1),
                60,
            )

        self.assertEqual(result["status"], "outside_hours")
        self.assertIn("10:00", result["message"])
        self.assertIn("16:00", result["message"])

    def test_hours_load_failure_never_confidently_offers_slot(self):
        with patch.object(
            self.chatbot,
            "load_business_hours_with_status",
            return_value=({day: None for day in range(7)}, False),
        ):
            start = datetime(2026, 6, 15, 11, tzinfo=self.chatbot.BUSINESS_TIMEZONE)
            result = self.chatbot.validate_slot_against_working_hours(
                start,
                start + timedelta(hours=1),
                60,
            )

        self.assertEqual(result["status"], "hours_unknown")
        self.assertIn("Do not guess", result["message"])

    def test_opening_hours_reply_uses_live_dashboard_values(self):
        hours = {day: None for day in range(7)}
        hours[0] = ((10, 0), (16, 0))

        with patch.object(
            self.chatbot,
            "load_business_hours_with_status",
            return_value=(hours, True),
        ):
            reply = self.chatbot.build_business_hours_reply()

        self.assertIn("• Monday: 10:00-16:00", reply)
        self.assertIn("• Tuesday: Closed", reply)


if __name__ == "__main__":
    unittest.main()
