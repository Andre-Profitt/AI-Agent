#!/bin/bash

echo "🚀 Starting AI Agent Frontend..."

cd saas-ui

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

echo "Starting development server..."
npm run dev