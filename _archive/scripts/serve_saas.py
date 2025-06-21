#!/usr/bin/env python3
"""Serve the built SaaS UI"""

import http.server
import socketserver
import os

os.chdir("saas-ui/dist")

PORT = 3000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        # Serve index.html for all routes (SPA support)
        if self.path != '/' and not '.' in os.path.basename(self.path):
            self.path = '/index.html'
        return super().do_GET()

Handler = MyHTTPRequestHandler

print(f"🚀 AI Agent SaaS Platform")
print(f"========================")
print(f"")
print(f"Frontend running at: http://localhost:{PORT}")
print(f"")
print(f"Make sure to also run the backend server:")
print(f"python test_server.py")
print(f"")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Server running... Press Ctrl+C to stop")
    httpd.serve_forever()