#!/bin/bash
echo "Stopping services..."
pkill -f "python.*test_server.py"
pkill -f "python.*serve_saas.py"
lsof -ti :3000 | xargs kill -9 2>/dev/null
lsof -ti :8000 | xargs kill -9 2>/dev/null
echo "Services stopped!"
