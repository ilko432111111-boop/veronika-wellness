import asyncio
import io
import json
import unittest
from unittest.mock import patch
from urllib import error as urllib_error

from test_smoke import load_chatbot_module


class UrlResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def oauth_http_error(status, payload):
    return urllib_error.HTTPError(
        "https://oauth2.googleapis.com/token",
        status,
        "OAuth request failed",
        {},
        io.BytesIO(json.dumps(payload).encode("utf-8")),
    )


class GoogleOAuthRefreshDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chatbot = load_chatbot_module()

    def run_refresh(self, urlopen_result=None, urlopen_error=None):
        with patch.object(
            self.chatbot,
            "get_google_refresh_token",
            return_value="hidden-placeholder",
        ):
            with patch.object(self.chatbot, "GOOGLE_CLIENT_ID", "placeholder"):
                with patch.object(
                    self.chatbot,
                    "GOOGLE_CLIENT_SECRET",
                    "hidden-placeholder",
                ):
                    with patch.object(
                        self.chatbot.urllib_request,
                        "urlopen",
                        return_value=urlopen_result,
                        side_effect=urlopen_error,
                    ):
                        return self.chatbot.refresh_google_access_token()

    def assert_refresh_diagnostic(
        self,
        expected_code,
        urlopen_result=None,
        urlopen_error=None,
    ):
        self.assertIsNone(
            self.run_refresh(
                urlopen_result=urlopen_result,
                urlopen_error=urlopen_error,
            )
        )
        self.assertEqual(
            self.chatbot.GOOGLE_CALENDAR_LAST_DIAGNOSTIC,
            expected_code,
        )

    def test_invalid_grant_is_classified_from_safe_oauth_error_code(self):
        self.assert_refresh_diagnostic(
            "refresh_token_invalid_grant",
            urlopen_error=oauth_http_error(
                400,
                {
                    "error": "invalid_grant",
                    "error_description": "must not be recorded",
                },
            ),
        )

    def test_invalid_client_is_classified_from_safe_oauth_error_code(self):
        self.assert_refresh_diagnostic(
            "refresh_token_invalid_client",
            urlopen_error=oauth_http_error(401, {"error": "invalid_client"}),
        )

    def test_http_400_without_known_oauth_error_is_classified(self):
        self.assert_refresh_diagnostic(
            "refresh_token_http_400",
            urlopen_error=oauth_http_error(400, {"error": "other"}),
        )

    def test_http_401_without_known_oauth_error_is_classified(self):
        self.assert_refresh_diagnostic(
            "refresh_token_http_401",
            urlopen_error=oauth_http_error(401, {"error": "other"}),
        )

    def test_timeout_is_classified(self):
        self.assert_refresh_diagnostic(
            "refresh_token_timeout",
            urlopen_error=TimeoutError(),
        )

    def test_url_timeout_is_classified(self):
        self.assert_refresh_diagnostic(
            "refresh_token_timeout",
            urlopen_error=urllib_error.URLError(TimeoutError()),
        )

    def test_network_error_is_classified(self):
        self.assert_refresh_diagnostic(
            "refresh_token_network_error",
            urlopen_error=urllib_error.URLError("network unavailable"),
        )

    def test_malformed_json_response_is_classified(self):
        self.assert_refresh_diagnostic(
            "refresh_token_malformed_response",
            urlopen_result=UrlResponse(b"not-json"),
        )

    def test_non_object_json_response_is_classified(self):
        self.assert_refresh_diagnostic(
            "refresh_token_malformed_response",
            urlopen_result=UrlResponse(b"[]"),
        )

    def test_missing_access_token_is_classified(self):
        self.assert_refresh_diagnostic(
            "refresh_token_missing_access_token",
            urlopen_result=UrlResponse(b"{}"),
        )

    def test_unexpected_http_status_is_classified(self):
        self.assert_refresh_diagnostic(
            "refresh_token_unexpected_error",
            urlopen_error=oauth_http_error(500, {"error": "server_error"}),
        )

    def test_unexpected_exception_is_classified(self):
        self.assert_refresh_diagnostic(
            "refresh_token_unexpected_error",
            urlopen_error=ValueError("unexpected"),
        )

    def test_refresh_error_logging_does_not_expose_google_response_fields(self):
        sensitive_marker = "must-not-be-logged"

        with patch("builtins.print") as print_mock:
            self.assert_refresh_diagnostic(
                "refresh_token_invalid_grant",
                urlopen_error=oauth_http_error(
                    400,
                    {
                        "error": "invalid_grant",
                        "error_description": sensitive_marker,
                    },
                ),
            )

        logged_text = " ".join(
            str(argument)
            for call in print_mock.call_args_list
            for argument in call.args
        )
        self.assertNotIn(sensitive_marker, logged_text)
        self.assertEqual(
            logged_text,
            "google_calendar_diagnostic=refresh_token_invalid_grant",
        )

    def test_callback_http_error_logging_contains_status_only(self):
        sensitive_marker = "must-not-be-logged"
        error = oauth_http_error(
            400,
            {
                "error": "invalid_grant",
                "error_description": sensitive_marker,
            },
        )

        with patch.object(
            self.chatbot,
            "verify_google_oauth_state",
            return_value=True,
        ):
            with patch.object(
                self.chatbot,
                "exchange_google_code_for_tokens",
                side_effect=error,
            ):
                with patch("builtins.print") as print_mock:
                    response = asyncio.run(
                        self.chatbot.google_calendar_callback(
                            code="hidden-placeholder",
                            state="hidden-placeholder",
                        )
                    )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            print_mock.call_args.args,
            ("Google OAuth token exchange failed: HTTP 400",),
        )
        self.assertNotIn(sensitive_marker, response.body.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
