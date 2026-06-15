import unittest
from unittest.mock import patch

from test_smoke import load_chatbot_module


class ServiceMenuFormattingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()
        cls.services = [
            {
                "category": "massage",
                "service_name": "Relaxing Massage",
                "allowed_durations_minutes": [30, 60],
                "price_by_duration": {"30": 3500, "60": 5000},
            },
            {
                "category": "massage",
                "service_name": "Swedish Massage",
                "allowed_durations_minutes": [30, 60],
                "price_by_duration": {"30": 3500, "60": 5000},
            },
            {
                "category": "massage",
                "service_name": "Deep Tissue Massage",
                "allowed_durations_minutes": [30, 60],
                "price_by_duration": {"30": 4000, "60": 5500},
            },
            {
                "category": "vitamin_shots",
                "service_name": "B12 Vitamin Shot",
                "fixed_duration_minutes": 15,
                "price_pence": 2000,
            },
            {
                "category": "facials",
                "service_name": "Deep Facial Cleanse",
                "fixed_duration_minutes": 60,
                "price_pence": 6000,
            },
            {
                "category": "facials",
                "service_name": "Hydrafacial",
                "fixed_duration_minutes": 90,
                "price_pence": 8000,
            },
        ]

    def compose(self, message):
        with patch.object(
            self.chatbot,
            "customer_service_catalogue",
            return_value=self.services,
        ):
            with patch.object(
                self.chatbot,
                "generate_natural_reply_body",
                return_value="One huge dense paragraph that should not be used.",
            ):
                return self.chatbot.compose_verified_customer_reply(
                    state={},
                    extractor_result=self.chatbot.empty_extractor_result(),
                    calendar_result={"status": "not_checked"},
                    next_required_detail=None,
                    business_context="",
                    services_context="",
                    latest_message=message,
                    history=[],
                )

    def test_full_service_request_is_grouped_and_readable(self):
        reply = self.compose("What services do you offer?")

        self.assertIn("\n\nMassage\n", reply)
        self.assertIn("\n\nVitamin Shots\n", reply)
        self.assertIn("\n\nFacials\n", reply)
        self.assertGreaterEqual(reply.count("\u2022 "), 4)
        self.assertIn("Relaxing Massage / Swedish Massage", reply)
        self.assertNotIn("One huge dense paragraph", reply)

    def test_massage_prices_only_show_massage(self):
        reply = self.compose("Massage prices?")

        self.assertIn("\n\nMassage\n", reply)
        self.assertNotIn("Facials\n", reply)
        self.assertNotIn("Vitamin Shots\n", reply)

    def test_facial_prices_only_show_facials(self):
        reply = self.compose("How much are facials?")

        self.assertIn("\n\nFacials\n", reply)
        self.assertIn("\u2022 Hydrafacial:", reply)
        self.assertNotIn("Massage\n", reply)

    def test_prices_request_returns_compact_grouped_overview(self):
        reply = self.compose("Prices?")

        self.assertIn("here's a quick overview", reply)
        self.assertGreaterEqual(reply.count("\n\n"), 4)
        self.assertNotIn("One huge dense paragraph", reply)

    def test_vitamin_shot_question_only_shows_vitamin_shots(self):
        reply = self.compose("Do you do vitamin shots?")

        self.assertIn("\n\nVitamin Shots\n", reply)
        self.assertIn("\u2022 B12 Vitamin Shot:", reply)
        self.assertNotIn("Massage\n", reply)
        self.assertNotIn("Facials\n", reply)

    def test_specific_service_price_question_does_not_expand_category(self):
        with patch.object(
            self.chatbot,
            "find_facial_service",
            return_value={
                "service_name": "Hydrafacial",
                "booking_mode": "fixed_duration",
            },
        ):
            self.assertEqual(
                self.chatbot.build_customer_service_menu_reply(
                    "How much is Hydrafacial?"
                ),
                "",
            )


if __name__ == "__main__":
    unittest.main()
