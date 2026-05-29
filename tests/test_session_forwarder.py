"""Unit tests for scripts/session_forwarder.py error handling."""

import http.client
import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.session_forwarder import Forwarder


class FakeRequest:
    """A fake request object that satisfies BaseHTTPRequestHandler's constructor."""

    def makefile(self, *args, **kwargs):
        return io.BytesIO()


class TestForwarderErrorHandling(unittest.TestCase):
    """Regression tests for PR #33 / issue #31 — network error handling."""

    def _make_handler(self):
        """Return a Forwarder instance with attrs required by _forward()."""
        handler = Forwarder(FakeRequest(), ("127.0.0.1", 12345), None)
        handler.target = "localhost:8080"
        handler.session = "test-session"
        handler.path = "/v1/messages"
        handler.headers = {}
        handler.command = "GET"
        handler.requestline = "GET /v1/messages HTTP/1.1"
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.send_error = MagicMock()
        return handler

    @patch("scripts.session_forwarder.http.client.HTTPConnection")
    def test_broken_pipe_during_stream_sends_502(self, mock_conn_class):
        """BrokenPipeError during body streaming sends 502 if headers not yet sent."""
        mock_conn = mock_conn_class.return_value
        mock_conn.getresponse.return_value = MagicMock(
            status=200,
            reason="OK",
            getheaders=lambda: [("Content-Type", "application/json")],
            read=MagicMock(side_effect=[b"chunk", b""]),
        )

        handler = self._make_handler()
        handler.wfile = MagicMock()
        handler.wfile.write = MagicMock(side_effect=BrokenPipeError("broken pipe"))

        handler._forward("GET")

        handler.send_error.assert_called_once_with(
            502, "Proxy forwarder upstream connection failed"
        )
        mock_conn.close.assert_called_once()

    @patch("scripts.session_forwarder.http.client.HTTPConnection")
    def test_send_error_swallows_value_error_after_headers_sent(self, mock_conn_class):
        """If headers already sent, ValueError from send_error is swallowed safely."""
        mock_conn = mock_conn_class.return_value
        mock_conn.getresponse.return_value = MagicMock(
            status=200,
            reason="OK",
            getheaders=lambda: [("Content-Length", "5")],
            read=MagicMock(side_effect=[b"hello", b""]),
        )

        handler = self._make_handler()
        handler.wfile = MagicMock()
        # Simulate headers already sent by forcing ValueError on send_error
        handler.send_error = MagicMock(side_effect=ValueError("headers already sent"))

        handler._forward("GET")
        # No unhandled exception should propagate
        mock_conn.close.assert_called_once()

    @patch("scripts.session_forwarder.http.client.HTTPConnection")
    def test_timeout_error_closed_in_finally(self, mock_conn_class):
        """TimeoutError ensures conn.close() still runs via finally."""
        mock_conn = mock_conn_class.return_value
        mock_conn.request.side_effect = TimeoutError("connection timed out")

        handler = self._make_handler()

        handler._forward("POST")

        handler.send_error.assert_called_once_with(
            502, "Proxy forwarder upstream connection failed"
        )
        mock_conn.close.assert_called_once()

    @patch("scripts.session_forwarder.http.client.HTTPConnection")
    def test_oserror_mid_stream(self, mock_conn_class):
        """Generic OSError during streaming is handled."""
        mock_conn = mock_conn_class.return_value
        mock_conn.getresponse.return_value = MagicMock(
            status=200,
            reason="OK",
            getheaders=lambda: [],
            read=MagicMock(side_effect=[b"data", b""]),
        )

        handler = self._make_handler()
        handler.wfile = MagicMock()
        handler.wfile.write = MagicMock(side_effect=OSError(32, "Broken pipe"))
        handler.send_error = MagicMock()

        handler._forward("GET")
        handler.send_error.assert_called_once_with(
            502, "Proxy forwarder upstream connection failed"
        )
        mock_conn.close.assert_called_once()

    @patch("scripts.session_forwarder.http.client.HTTPConnection")
    def test_connection_reset_during_stream(self, mock_conn_class):
        """ConnectionResetError during streaming sends 502."""
        mock_conn = mock_conn_class.return_value
        mock_conn.getresponse.return_value = MagicMock(
            status=200,
            reason="OK",
            getheaders=lambda: [],
            read=MagicMock(side_effect=[b"data", b""]),
        )

        handler = self._make_handler()
        handler.wfile = MagicMock()
        handler.wfile.write = MagicMock(side_effect=ConnectionResetError("Connection reset"))
        handler.send_error = MagicMock()

        handler._forward("GET")

        handler.send_error.assert_called_once_with(
            502, "Proxy forwarder upstream connection failed"
        )
        mock_conn.close.assert_called_once()

    @patch("scripts.session_forwarder.http.client.HTTPConnection")
    def test_remote_disconnected_during_response(self, mock_conn_class):
        """http.client.RemoteDisconnected is caught and 502 is sent."""
        mock_conn = mock_conn_class.return_value
        exc = http.client.RemoteDisconnected("Remote end closed connection")
        mock_conn.getresponse.side_effect = exc

        handler = self._make_handler()
        handler.send_error = MagicMock()

        handler._forward("GET")

        handler.send_error.assert_called_once_with(
            502, "Proxy forwarder upstream connection failed"
        )
        mock_conn.close.assert_called_once()

    @patch("scripts.session_forwarder.http.client.HTTPConnection")
    def test_finally_closes_conn_even_on_success(self, mock_conn_class):
        """Even on successful forwarding, conn.close() is called."""
        mock_conn = mock_conn_class.return_value
        mock_conn.getresponse.return_value = MagicMock(
            status=200,
            reason="OK",
            getheaders=lambda: [("Content-Length", "5")],
            read=MagicMock(side_effect=[b"hello", b""]),
        )

        handler = self._make_handler()
        handler.wfile = MagicMock()

        handler._forward("GET")

        handler.send_error.assert_not_called()
        mock_conn.close.assert_called_once()
