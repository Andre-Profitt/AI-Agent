"""
UI Module for AI Agent
======================

This module handles all Gradio UI components and interface logic.
"""

import gradio as gr
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import json
import datetime
from pathlib import Path

from config import config


def create_main_chat_interface():
    """Create the main chat interface tab"""
    with gr.Column():
        gr.Markdown(
            """
            # 🧠 Advanced AI Agent Chat
            
            This is an enhanced ReAct agent with:
            - 🔧 Multiple specialized tools (web search, code execution, file analysis)
            - 🎯 Strategic planning and reflection
            - ⚡ Parallel processing capabilities
            - 💾 Response caching for faster interactions
            - 📊 Real-time performance analytics
            
            Try asking complex questions that require research, calculation, or analysis!
            """
        )
        
        chatbot = gr.Chatbot(
            value=[], 
            height=600,
            show_label=False,
            elem_id="main-chatbot"
        )
        
        with gr.Row():
            message = gr.Textbox(
                placeholder="Ask me anything... (e.g., 'What's the weather in Paris and convert it to Fahrenheit?')",
                show_label=False,
                scale=6,
                lines=2
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", scale=1)
            export_btn = gr.Button("💾 Export Conversation", scale=1)
            reset_btn = gr.Button("🔄 Reset Session", scale=1)
        
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            log_to_db = gr.Checkbox(
                label="Log conversation to database", 
                value=True,
                info="Enable trajectory logging for analysis"
            )
            
            parallel_mode = gr.Checkbox(
                label="Enable parallel processing",
                value=True,
                info="Process requests using worker pool for faster responses"
            )
            
            model_preference = gr.Radio(
                choices=["balanced", "fast", "powerful"],
                value="balanced",
                label="Model Preference",
                info="Choose between speed and capability"
            )
        
        # Hidden components for state management
        session_id = gr.State(value=None)
        
        # File upload section
        with gr.Accordion("📎 File Upload", open=False):
            file_upload = gr.File(
                label="Upload a file for analysis",
                file_types=[".txt", ".pdf", ".csv", ".json", ".py", ".md"]
            )
            file_status = gr.Markdown("")
        
        gr.Markdown(
            """
            ### 💡 Example Queries:
            - "Search for the latest AI news and summarize the top 3 stories"
            - "Write a Python function to calculate fibonacci numbers and test it"
            - "What's the weather in Tokyo? Convert to Fahrenheit and explain the conversion"
            - "Analyze this CSV file and create a summary of the data"
            """
        )
        
    return {
        "chatbot": chatbot,
        "message": message,
        "submit_btn": submit_btn,
        "clear_btn": clear_btn,
        "export_btn": export_btn,
        "reset_btn": reset_btn,
        "log_to_db": log_to_db,
        "parallel_mode": parallel_mode,
        "model_preference": model_preference,
        "session_id": session_id,
        "file_upload": file_upload,
        "file_status": file_status
    }


def create_gaia_evaluation_tab():
    """Create the GAIA evaluation tab"""
    with gr.Column():
        gr.Markdown(
            """
            # 🎯 GAIA Benchmark Evaluation
            
            Run comprehensive evaluation on the GAIA benchmark dataset.
            This tests the agent's ability to handle complex, real-world tasks.
            
            **Requirements:**
            - Login with Hugging Face account
            - Agent will fetch questions from GAIA API
            - Automatic submission of results
            """
        )
        
        with gr.Row():
            login_btn = gr.LoginButton("🤗 Login to Hugging Face", scale=1)
            
        submit_btn = gr.Button(
            "🚀 Run GAIA Evaluation", 
            variant="primary",
            interactive=True,
            scale=2
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    ### 📊 Evaluation Features:
                    - Advanced reasoning with FSM control
                    - Multi-tool orchestration
                    - Parallel question processing
                    - Automatic answer extraction
                    - Performance analytics
                    """
                )
            
            with gr.Column(scale=2):
                status_output = gr.Markdown(
                    value="Ready to start evaluation. Please login first.",
                    elem_id="gaia-status"
                )
        
        results_df = gr.DataFrame(
            label="Evaluation Results",
            headers=["Task ID", "Question", "Submitted Answer", "Processing Time"],
            interactive=False,
            wrap=True
        )
        
        gr.Markdown(
            """
            ### 🔍 What is GAIA?
            GAIA (General AI Assistant) benchmark tests AI agents on diverse real-world tasks including:
            - Research and fact-checking
            - Mathematical reasoning
            - Code generation and debugging  
            - Multi-step problem solving
            - Tool usage and orchestration
            """
        )
        
    return {
        "login_btn": login_btn,
        "submit_btn": submit_btn,
        "status_output": status_output,
        "results_df": results_df
    }


def create_analytics_tab():
    """Create the analytics and monitoring tab"""
    with gr.Column():
        gr.Markdown(
            """
            # 📊 Performance Analytics & Monitoring
            
            Real-time insights into agent performance, tool usage, and system health.
            """
        )
        
        refresh_btn = gr.Button("🔄 Refresh Analytics", variant="secondary")
        
        with gr.Row():
            # Performance Metrics
            with gr.Column(scale=1):
                gr.Markdown("### ⚡ Performance Metrics")
                total_queries = gr.Number(label="Total Queries", value=0)
                avg_response_time = gr.Number(label="Avg Response Time (s)", value=0)
                cache_hit_rate = gr.Number(label="Cache Hit Rate (%)", value=0)
                parallel_executions = gr.Number(label="Parallel Executions", value=0)
            
            # Tool Usage
            with gr.Column(scale=1):
                gr.Markdown("### 🔧 Tool Usage Statistics")
                tool_usage_json = gr.JSON(
                    label="Tool Usage Count",
                    value={}
                )
            
            # System Health
            with gr.Column(scale=1):
                gr.Markdown("### 💚 System Health")
                active_workers = gr.Number(label="Active Workers", value=0)
                memory_usage = gr.Number(label="Memory Usage (MB)", value=0)
                api_status = gr.Textbox(label="API Status", value="Healthy")
                uptime = gr.Number(label="Uptime (hours)", value=0)
        
        # Detailed Logs
        with gr.Accordion("📋 Recent Activity Logs", open=False):
            activity_logs = gr.DataFrame(
                headers=["Timestamp", "Event", "Details", "Duration"],
                label="Recent Events"
            )
        
        # Error Analysis
        with gr.Accordion("⚠️ Error Analysis", open=False):
            error_summary = gr.JSON(
                label="Error Summary",
                value={"total_errors": 0, "error_types": {}}
            )
        
    return {
        "refresh_btn": refresh_btn,
        "total_queries": total_queries,
        "avg_response_time": avg_response_time,
        "cache_hit_rate": cache_hit_rate,
        "parallel_executions": parallel_executions,
        "tool_usage_json": tool_usage_json,
        "active_workers": active_workers,
        "memory_usage": memory_usage,
        "api_status": api_status,
        "uptime": uptime,
        "activity_logs": activity_logs,
        "error_summary": error_summary
    }


def create_documentation_tab():
    """Create the documentation tab"""
    with gr.Column():
        gr.Markdown(
            """
            # 📚 Documentation & Guide
            
            ## 🚀 Getting Started
            
            This AI Agent is a sophisticated system that combines multiple AI models and tools to help you with complex tasks.
            
            ### Key Features:
            
            #### 1. **Multi-Tool Orchestration** 🔧
            - **Web Search**: Real-time information from the internet
            - **Code Execution**: Python interpreter for calculations and data analysis
            - **File Analysis**: Process various file formats (PDF, CSV, TXT, etc.)
            - **Knowledge Base**: Semantic search through curated information
            
            #### 2. **Advanced Reasoning** 🧠
            - Strategic planning for complex queries
            - Self-reflection and error correction
            - Verification of results before responding
            
            #### 3. **Performance Optimization** ⚡
            - Parallel processing for faster responses
            - Response caching to avoid redundant work
            - Automatic retry with exponential backoff
            
            ## 💡 Tips for Best Results
            
            1. **Be Specific**: The more detailed your question, the better the response
            2. **Complex Queries**: Don't hesitate to ask multi-step questions
            3. **File Upload**: Use the file upload feature for document analysis
            4. **Follow-ups**: Ask follow-up questions to dig deeper into topics
            
            ## 🔍 Example Use Cases
            
            ### Research & Analysis
            ```
            "Find the latest research on quantum computing and summarize the key breakthroughs from 2024"
            ```
            
            ### Data Processing
            ```
            "Analyze this sales data CSV and create a summary with key insights and trends"
            ```
            
            ### Problem Solving
            ```
            "I need to optimize a Python function for sorting large datasets. Show me different approaches and benchmark them"
            ```
            
            ### Multi-Step Tasks
            ```
            "What's the current Bitcoin price? Calculate how much $1000 would be worth, and compare it to gold investment returns"
            ```
            
            ## ⚙️ Configuration Options
            
            - **Model Preference**: 
              - *Balanced*: Best trade-off between speed and quality
              - *Fast*: Quick responses for simple queries
              - *Powerful*: Maximum capability for complex tasks
              
            - **Parallel Processing**: Enable for faster multi-tool execution
            - **Database Logging**: Track conversations for analysis
            
            ## 🛠️ Technical Details
            
            - Built with LangChain and LangGraph
            - Powered by Groq's Llama models
            - Finite State Machine (FSM) for deterministic control flow
            - Production-ready with circuit breakers and retries
            
            ## 📊 GAIA Benchmark
            
            The GAIA evaluation tests the agent on standardized tasks to measure:
            - Accuracy in answering questions
            - Ability to use tools effectively
            - Performance on complex reasoning tasks
            
            ---
            
            For more information, check out the [GitHub repository](https://github.com/yourusername/ai-agent).
            """
        )


def format_message_with_steps(steps: str, final_response: str) -> str:
    """Format the message to show steps and final response nicely"""
    if steps and "⚠️" not in steps:  # Don't show steps for errors
        return f"{steps}\n\n---\n\n**Final Response:**\n{final_response}"
    return final_response


def export_conversation_to_file(history: List[List[str]], session_id: str = None) -> str:
    """Export conversation history to a formatted text file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{session_id[:8] if session_id else 'export'}_{timestamp}.txt"
    filepath = Path("/tmp") / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"AI Agent Conversation Export\n")
        f.write(f"Session ID: {session_id}\n")
        f.write(f"Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, (user_msg, assistant_msg) in enumerate(history, 1):
            f.write(f"User ({i}):\n{user_msg}\n\n")
            f.write(f"Assistant ({i}):\n{assistant_msg}\n")
            f.write("-" * 30 + "\n\n")
    
    return str(filepath)


def create_custom_css():
    """Create custom CSS for the interface"""
    return """
    #main-chatbot {
        height: 600px !important;
    }
    
    .message {
        padding: 10px !important;
        margin: 5px !important;
    }
    
    #gaia-status {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f0f0;
    }
    
    .gr-button-primary {
        background-color: #2196F3 !important;
    }
    
    .gr-button-primary:hover {
        background-color: #1976D2 !important;
    }
    """ 