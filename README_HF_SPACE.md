---
title: AI Agent Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
python_version: 3.11
---

# AI Agent Assistant

An advanced AI agent with strategic planning, reflection, and tool use capabilities.

## Features

- 🎯 Strategic Planning with step-by-step execution
- 🔧 Multiple specialized tools (web search, code execution, file reading, etc.)
- 🧠 Self-reflection and error correction
- 💡 Intelligent routing between different approaches
- 🚀 Powered by LangChain, LlamaIndex, and Groq

## Configuration

This Space requires the following environment variables:
- `GROQ_API_KEY`: Your Groq API key
- `OPENAI_API_KEY`: Your OpenAI API key (optional)
- `TAVILY_API_KEY`: Your Tavily API key (optional)

## Usage

1. Enter your query in the text box
2. The agent will plan and execute steps to answer your question
3. View the step-by-step process and final answer

## Technical Stack

- Python 3.11
- LangChain 0.3.25 with Pydantic v2
- LlamaIndex 0.12.42
- Gradio 5.25.2
- PyTorch 2.0.1
