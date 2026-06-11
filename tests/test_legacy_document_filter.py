import unittest
from unittest.mock import patch

from test_smoke import load_chatbot_module


class QueryStub:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_args):
        return self

    def execute(self):
        return type("Response", (), {"data": self.rows})()


class SupabaseStub:
    def __init__(self, rows):
        self.rows = rows

    def table(self, name):
        if name != "documents":
            raise AssertionError(f"Unexpected table: {name}")

        return QueryStub(self.rows)


class LegacyDocumentFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()
        cls.catalogue = [
            {
                "category": "massage",
                "service_name": "Relaxing Massage",
                "aliases": ["relaxation massage"],
            },
            {
                "category": "body_treatments",
                "service_name": "Electronic Muscle Stimulation (EMS)",
                "aliases": ["ems"],
            },
            {
                "category": "facials",
                "service_name": "Hydrafacial",
                "aliases": ["hydrafacial 90 minutes"],
            },
            {
                "category": "microneedling",
                "service_name": "Microneedling for 1 Area - Stretch Marks",
                "aliases": ["microneedling for stretch marks"],
            },
            {
                "category": "dermal_fillers",
                "service_name": "Dermal Filler - Lip Filler 0.5 ml",
                "aliases": ["lip filler 0.5 ml"],
            },
        ]

    def test_filters_clear_structured_service_catalogue_rows(self):
        rows = [
            "Relaxing Massage: 60 minutes for 50 pounds",
            "Hydrafacial for 90 Minutes: 1 hour 30 minutes for 80 pounds",
            "Electronic Muscle Stimulation (EMS): 1 hour for 200 pounds",
            "Dermal Filler (Lip Filler, 0.5 ml): 45 minutes for 70 pounds",
            (
                "Microneedling for 1 Area is available for stretch marks or "
                "the face and neck. Prices range from 60 to 70 pounds "
                "depending on the selected area."
            ),
            (
                "Dermal filler treatments are available for lip filler, "
                "marionette lines and nasolabial folds. Each area can be "
                "treated with either 0.5 ml for 70 pounds or 1 ml for "
                "100 pounds."
            ),
        ]

        for content in rows:
            with self.subTest(content=content):
                self.assertTrue(
                    self.chatbot.is_legacy_catalogue_document(
                        content,
                        self.catalogue,
                    )
                )

    def test_retains_general_information_and_non_catalogue_descriptions(self):
        rows = [
            "Business Name: Veronikas Beauty",
            "Address: 25 Albion Place, Leeds, LS1 6JS",
            "Parking Information: There is no free parking nearby.",
            "Please provide 24 hours notice when cancelling.",
            "Microneedling can help improve the appearance of the skin.",
            "Dermal fillers are available after a consultation.",
            "Hydrafacial is a gentle treatment designed to refresh the skin.",
            "An unmigrated treatment costs 40 pounds and takes 30 minutes.",
        ]

        for content in rows:
            with self.subTest(content=content):
                self.assertFalse(
                    self.chatbot.is_legacy_catalogue_document(
                        content,
                        self.catalogue,
                    )
                )

    def test_does_not_filter_category_when_only_fallback_catalogue_exists(self):
        self.assertFalse(
            self.chatbot.is_legacy_catalogue_document(
                "Relaxing Massage: 60 minutes for 50 pounds",
                [],
            )
        )

    def test_migrated_catalogue_contains_only_live_supabase_categories(self):
        live_service = {
            "service_name": "Relaxing Massage",
            "aliases": ["relaxation massage"],
        }
        fallback_service = {
            "service_name": "Hydrafacial",
            "aliases": ["hydrafacial"],
        }

        with (
            patch.object(
                self.chatbot,
                "load_massage_services",
                return_value=([live_service], True),
            ),
            patch.object(
                self.chatbot,
                "load_facial_services",
                return_value=([fallback_service], False),
            ),
            patch.object(
                self.chatbot,
                "load_vitamin_shot_services",
                return_value=([], False),
            ),
            patch.object(
                self.chatbot,
                "load_ultrasound_services",
                return_value=([], False),
            ),
            patch.object(
                self.chatbot,
                "load_body_treatment_services",
                return_value=([], False),
            ),
            patch.object(
                self.chatbot,
                "load_microneedling_services",
                return_value=([], False),
            ),
            patch.object(
                self.chatbot,
                "load_dermal_filler_services",
                return_value=([], False),
            ),
        ):
            catalogue = self.chatbot.migrated_structured_service_catalogue()

        self.assertEqual(
            catalogue,
            [{
                "category": "massage",
                "service_name": "Relaxing Massage",
                "aliases": ["relaxation massage"],
            }],
        )

    def test_document_loader_filters_catalogue_and_opening_hours_only(self):
        documents = [
            {"content": "Business Name: Veronikas Beauty"},
            {"content": "Opening hours: Monday 09:00-17:00"},
            {"content": "Relaxing Massage: 60 minutes for 50 pounds"},
            {"content": "An unmigrated treatment costs 40 pounds and takes 30 minutes."},
            {"content": ""},
        ]

        with (
            patch.object(self.chatbot, "supabase", SupabaseStub(documents)),
            patch.object(
                self.chatbot,
                "migrated_structured_service_catalogue",
                return_value=self.catalogue,
            ),
            self.assertLogs(self.chatbot.logger, level="INFO") as logs,
        ):
            context = self.chatbot.load_business_documents_context()

        self.assertEqual(
            context,
            "Business Name: Veronikas Beauty\n"
            "An unmigrated treatment costs 40 pounds and takes 30 minutes.",
        )
        self.assertIn("total=5", logs.output[0])
        self.assertIn("legacy_catalogue_filtered=1", logs.output[0])
        self.assertIn("retained=2", logs.output[0])


if __name__ == "__main__":
    unittest.main()
