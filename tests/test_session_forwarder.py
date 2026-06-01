import http.client
import importlib.util
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# session_forwarder.py lives in scripts/ which has no __init__.py, so importlib is used.
_spec = importlib.util.spec_from_file_location(
    "session_forwarder",
    str(Path(__file__).resolve().parents[1] / "scripts" / "session_forwarder.py"),
)
_forwarder_mod = importlib.util.module_from_spec(_spec)
sys.modules["session_forwarder"] = _forwarder_mod
_spec.loader.exec_module(_forwarder_mod)

Forwarder = _forwarder_mod.Forwarder

ALL_NETWORK_EXCS = (
    BrokenPipeError,
    ConnectionResetError,
    TimeoutError,
    http.client.RemoteDisconnected,
    http.client.HTTPException,  # covers BadStatusLine, IncompleteRead, etc.
    OSError,  # covers DNS errors and other I/O failures
)


def _make_handler():
    """Return a minimally-initialised Forwarder instance for testing."""
    handler = Forwarder.__new__(Forwarder)
    handler.client_address = ("127.0.0.1", 54321)
    handler.path = "/v1/messages"
    handler.headers = {"Host": "localhost:50001"}
    handler.rfile = MagicMock()
    handler.wfile = MagicMock()
    call_log = []
    handler.send_error = lambda code, msg="": call_log.append(("send_error", code, msg))
    handler._send_error_log = call_log
    return handler


@pytest.mark.parametrize("exc_cls", ALL_NETWORK_EXCS)
def test_forward_logs_and_returns_502_on_network_error(exc_cls):
    handler = _make_handler()
    call_log = handler._send_error_log

    stderr_capture = StringIO()
    with patch("http.client.HTTPConnection") as MockConn:
        mock_conn = MagicMock()
        mock_conn.request.side_effect = exc_cls("simulated")
        MockConn.return_value = mock_conn
        with patch.object(sys, "stderr", stderr_capture):
            Forwarder._forward(handler, "POST")

    assert "forwarder] request forwarding failed" in stderr_capture.getvalue()
    assert ("send_error", 502, "Proxy forwarder upstream connection failed") in call_log


@pytest.mark.parametrize(
    "outer_exc, inner_exc",
    [
        (BrokenPipeError("fwd"), ConnectionResetError("send")),
        (http.client.HTTPException("fwd"), TimeoutError("send")),
        (OSError("fwd"), BrokenPipeError("send")),
    ],
)
def test_forward_logs_nested_send_error_failure(outer_exc, inner_exc):
    """If send_error itself fails, the failure must also be logged to stderr."""
    handler = _make_handler()
    handler.send_error = MagicMock(side_effect=inner_exc)

    stderr_capture = StringIO()
    with patch("http.client.HTTPConnection") as MockConn:
        mock_conn = MagicMock()
        mock_conn.request.side_effect = outer_exc
        MockConn.return_value = mock_conn
        with patch.object(sys, "stderr", stderr_capture):
            Forwarder._forward(handler, "POST")

    assert "forwarder] request forwarding failed" in stderr_capture.getvalue()
    assert "forwarder] send_error also failed" in stderr_capture.getvalue()


def test_conn_closed_on_successful_forward():
    """Connection must be closed even when no exception is raised."""
    handler = _make_handler()
    handler.requestline = "GET /v1/messages HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = "GET"
    handler.protocol_version = "HTTP/1.1"
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.reason = "OK"
    mock_resp.getheaders.return_value = []
    mock_resp.read.side_effect = [b"data", b""]

    with patch("http.client.HTTPConnection") as MockConn:
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp
        MockConn.return_value = mock_conn
        Forwarder._forward(handler, "GET")

        mock_conn.close.assert_called_once()


def test_conn_closed_on_network_error():
    """Connection must be closed even when the forwarder hits an error."""
    handler = _make_handler()

    stderr_capture = StringIO()
    with patch("http.client.HTTPConnection") as MockConn:
        mock_conn = MagicMock()
        mock_conn.request.side_effect = ConnectionResetError("simulated")
        MockConn.return_value = mock_conn
        with patch.object(sys, "stderr", stderr_capture):
            Forwarder._forward(handler, "POST")

        mock_conn.close.assert_called_once()
