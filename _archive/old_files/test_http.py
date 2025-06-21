#!/usr/bin/env python3
import http.server
import socketserver

PORT = 3000

Handler = http.server.SimpleHTTPRequestHandler

print(f"Starting test HTTP server on http://localhost:{PORT}")
print("If this works, we know port 3000 is accessible")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Server running at http://localhost:{PORT}/")
    httpd.serve_forever()