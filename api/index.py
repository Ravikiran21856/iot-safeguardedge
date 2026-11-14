"""
Vercel Serverless Function entrypoint for IoT SafeguardEdge.
Serves the standalone interactive executive HTML dashboard.
Uses exclusively Python standard library (http.server, os) with zero heavy dependencies.
"""

from http.server import BaseHTTPRequestHandler
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class handler(BaseHTTPRequestHandler):
    """Standard Vercel serverless HTTP request handler."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

        # Search for the dashboard HTML in priority order
        candidates = [
            os.path.join(BASE_DIR, "public", "index.html"),
            os.path.join(BASE_DIR, "index.html"),
            os.path.join(BASE_DIR, "results", "dashboard.html"),
        ]

        for path in candidates:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.wfile.write(f.read())
                return

        self.wfile.write(
            b"<!DOCTYPE html><html><body><h1>IoT SafeguardEdge</h1><p>Dashboard active.</p></body></html>"
        )

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            b'{"status":"ok","service":"IoT SafeguardEdge Edge Intelligence"}'
        )

    def log_message(self, format, *args):
        pass


# Top-level exports for WSGI/ASGI/Serverless runtimes
app = handler
application = handler
