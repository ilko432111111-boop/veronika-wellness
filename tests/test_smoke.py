import ast
from datetime import timezone
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import zoneinfo

from fastapi import FastAPI


ROOT = Path(__file__).resolve().parents[1]
CHATBOT_PATH = ROOT / "chatbot.py"
START_PATH = ROOT / "start.sh"


class StubGroq:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class StubSupabaseClient:
    pass


def load_chatbot_module():
    groq_module = types.ModuleType("groq")
    groq_module.Groq = StubGroq

    supabase_module = types.ModuleType("supabase")
    supabase_module.Client = StubSupabaseClient
    supabase_module.create_client = lambda url, key: StubSupabaseClient()

    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda: None

    controlled_environment = {
        "GROQ_API_KEY": "smoke-test-placeholder",
        "SUPABASE_URL": "https://smoke-test-placeholder.invalid",
        "SUPABASE_KEY": "smoke-test-placeholder",
    }
    stub_modules = {
        "groq": groq_module,
        "supabase": supabase_module,
        "dotenv": dotenv_module,
    }

    spec = importlib.util.spec_from_file_location("chatbot_smoke_test", CHATBOT_PATH)
    module = importlib.util.module_from_spec(spec)

    with patch.dict(os.environ, controlled_environment, clear=True):
        with patch.dict(sys.modules, stub_modules):
            with patch.object(zoneinfo, "ZoneInfo", lambda name: timezone.utc):
                with patch(
                    "urllib.request.urlopen",
                    side_effect=AssertionError("Network access during import"),
                ):
                    spec.loader.exec_module(module)

    return module


class BootstrapSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = CHATBOT_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(CHATBOT_PATH))
        cls.chatbot = load_chatbot_module()

    def test_chatbot_parses_successfully(self):
        self.assertIsInstance(self.tree, ast.Module)

    def test_chatbot_imports_without_external_network_access(self):
        self.assertIsNotNone(self.chatbot)
        self.assertIsInstance(self.chatbot.groq_client, StubGroq)
        self.assertIsInstance(self.chatbot.supabase, StubSupabaseClient)

    def test_fastapi_app_exists(self):
        self.assertIsInstance(self.chatbot.app, FastAPI)

    def test_post_chat_is_registered_exactly_once(self):
        chat_routes = [
            route
            for route in self.chatbot.app.routes
            if getattr(route, "path", None) == "/chat"
            and "POST" in getattr(route, "methods", set())
        ]

        self.assertEqual(len(chat_routes), 1)

    def test_start_script_references_chatbot_app(self):
        self.assertIn("uvicorn chatbot:app", START_PATH.read_text(encoding="utf-8"))

    def test_positive_integer_environment_helper_uses_default_for_invalid_values(self):
        helper = self.chatbot.read_positive_int_environment_variable

        for value in [None, "", "invalid", "0", "-1"]:
            with self.subTest(value=value):
                environment = {} if value is None else {"SMOKE_INT": value}

                with patch.dict(os.environ, environment, clear=True):
                    self.assertEqual(helper("SMOKE_INT", 42), 42)

    def test_positive_integer_environment_helper_accepts_positive_integer(self):
        with patch.dict(os.environ, {"SMOKE_INT": "73"}, clear=True):
            self.assertEqual(
                self.chatbot.read_positive_int_environment_variable(
                    "SMOKE_INT",
                    42,
                ),
                73,
            )


if __name__ == "__main__":
    unittest.main()
