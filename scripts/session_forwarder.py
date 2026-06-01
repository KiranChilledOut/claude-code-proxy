#!/usr/bin/env python3
"""Lightweight HTTP forwarder that adds x-session-name to all requests.

Usage: python3 session_forwarder.py <port> <target_host:port> <session_name>

Example: python3 session_forwarder.py 50001 localhost:8083 bugfix-27
"""

import http.client
import http.server
import socketserver
import sys
import urllib.parse


class Forwarder(http.server.BaseHTTPRequestHandler):
    """Forward all requests to target, injecting x-session-name header."""

    target = ""
    session = ""

    def log_message(self, fmt, *args):
        # Suppress default access logging to avoid noise
        pass

    def _forward(self, method):
        forwarded_for = self.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            forwarded_for = f"{forwarded_for}, {self.client_address[0]}"
        else:
            forwarded_for = self.client_address[0]

        # Parse target
        parsed = urllib.parse.urlparse(f"http://{self.target}")
        host, _, port = parsed.netloc.partition(":")
        port = int(port) if port else 80

        # Collect body for POST/PUT/PATCH
        body = None
        content_length = self.headers.get("Content-Length")
        if content_length:
            body = self.rfile.read(int(content_length))

        # Build headers, inject x-session-name
        headers = dict(self.headers)
        headers["x-session-name"] = self.session
        headers["X-Forwarded-For"] = forwarded_for
        headers["Host"] = f"{host}:{port}"

        # Remove hop-by-hop headers
        for h in (
            "Connection",
            "Keep-Alive",
            "Proxy-Authorization",
            "Proxy-Authenticate",
            "TE",
            "Transfer-Encoding",
            "Upgrade",
        ):
            headers.pop(h, None)
            headers.pop(h.lower(), None)

        # Forward request
        conn = http.client.HTTPConnection(host, port, timeout=300)
        try:
            conn.request(method, self.path, body=body, headers=headers)
            response = conn.getresponse()

            # Send response status
            self.send_response(response.status, response.reason)

            # Send response headers (skip hop-by-hop)
            for header, value in response.getheaders():
                if header.lower() in ("transfer-encoding", "connection", "keep-alive", "upgrade"):
                    continue
                self.send_header(header, value)
            self.end_headers()

            # Stream body in chunks
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (
            BrokenPipeError,
            ConnectionResetError,
            TimeoutError,
            http.client.RemoteDisconnected,
            http.client.HTTPException,
            OSError,
        ) as exc:
            print(
                f"[forwarder] request forwarding failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            try:
                self.send_error(502, "Proxy forwarder upstream connection failed")
            except (
                BrokenPipeError,
                ConnectionResetError,
                TimeoutError,
                http.client.RemoteDisconnected,
                OSError,
            ):
                print(
                    f"[forwarder] send_error also failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
        finally:
            conn.close()

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def do_HEAD(self):
        self._forward("HEAD")

    def do_PUT(self):
        self._forward("PUT")

    def do_PATCH(self):
        self._forward("PATCH")

    def do_DELETE(self):
        self._forward("DELETE")

    def do_OPTIONS(self):
        self._forward("OPTIONS")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: session_forwarder.py <port> <target_host:port> <session_name>", file=sys.stderr
        )
        sys.exit(1)

    port = int(sys.argv[1])
    target = sys.argv[2]
    session_name = sys.argv[3]

    Forwarder.target = target
    Forwarder.session = session_name

    server = ThreadedHTTPServer(("127.0.0.1", port), Forwarder)
    print(f"[forwarder] {session_name} -> {target} (port {port})", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
