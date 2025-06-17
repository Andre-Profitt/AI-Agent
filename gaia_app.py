"""
GAIA Benchmark Evaluation Runner with Advanced Agent Features
Integrates sophisticated AI agent capabilities with GAIA benchmark submission
"""

import os
import inspect
import gradio as gr
import requests
import pandas as pd
import logging
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage
from agent import build_graph

# Import existing advanced features
from src.advanced_agent import AdvancedReActAgent
from src.database import get_supabase_client, SupabaseLogHandler
from src.tools import get_tools
from app import (
    session_manager, 
    parallel_pool, 
    response_cache,
    format_chat_history,
    extract_final_answer,
    process_uploaded_file
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# Initialize advanced features
try:
    tools = get_tools()
    logger.info(f"✅ Initialized {len(tools)} advanced tools")
    
    # Initialize Supabase logging if available
    try:
        supabase_client = get_supabase_client()
        supabase_handler = SupabaseLogHandler(supabase_client)
        LOGGING_ENABLED = True
        logger.info("✅ Supabase logging enabled")
    except Exception as e:
        logger.warning(f"⚠️ Supabase logging disabled: {e}")
        supabase_handler = None
        LOGGING_ENABLED = False
    
except Exception as e:
    logger.error(f"❌ Failed to initialize advanced features: {e}")
    tools = []
    LOGGING_ENABLED = False
    supabase_handler = None

# --- Advanced GAIA Agent ---
class AdvancedGAIAAgent:
    """
    GAIA Agent with all advanced features:
    - Strategic planning and reflection
    - Cross-validation and verification  
    - Adaptive model selection
    - Tool orchestration
    - Performance monitoring
    """
    
    def __init__(self):
        logger.info("🚀 Initializing Advanced GAIA Agent...")
        
        try:
            self.graph = build_graph()
            self.session_count = 0
            self.performance_stats = {
                "total_questions": 0,
                "successful_answers": 0,
                "avg_processing_time": 0.0,
                "tool_usage": {tool.name: 0 for tool in tools},
                "start_time": time.time()
            }
            logger.info("✅ Advanced GAIA Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Advanced GAIA Agent: {e}")
            raise RuntimeError(f"Agent initialization failed: {e}")
    
    def __call__(self, question: str) -> str:
        """Process GAIA question with advanced reasoning."""
        start_time = time.time()
        self.session_count += 1
        
        try:
            logger.info(f"🎯 Processing GAIA question #{self.session_count}")
            logger.debug(f"Question preview: {question[:100]}...")
            
            # Enhanced state for comprehensive processing
            messages = [HumanMessage(content=question)]
            enhanced_state = {
                "messages": messages,
                "run_id": uuid.uuid4(),
                "log_to_db": LOGGING_ENABLED,
                # Strategic planning
                "master_plan": [],
                "current_step": 0,
                "plan_revisions": 0,
                # Reflection capabilities
                "reflections": [],
                "confidence_history": [],
                "error_recovery_attempts": 0,
                # Adaptive intelligence
                "step_count": 0,
                "confidence": 0.3,
                "reasoning_complete": False,
                "verification_level": "thorough",  # High verification for GAIA
                # Tool performance
                "tool_success_rates": {},
                "tool_results": [],
                "cross_validation_sources": []
            }
            
            # Process with advanced reasoning
            result = self.graph.invoke(enhanced_state)
            
            # Extract clean answer
            final_message = result['messages'][-1]
            raw_answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            # Clean answer for GAIA submission
            clean_answer = self._extract_clean_answer(raw_answer)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self._update_stats(processing_time, True)
            
            logger.info(f"✅ Question processed in {processing_time:.2f}s")
            return clean_answer
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, False)
            error_msg = f"Error processing question: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return error_msg
    
    def _extract_clean_answer(self, response: str) -> str:
        """Extract clean answer for GAIA submission."""
        if not response:
            return "No answer provided"
        
        # Use the advanced answer extraction from the original app
        clean_answer = extract_final_answer(response)
        
        # Additional GAIA-specific cleaning
        if len(clean_answer) > 500:  # GAIA prefers concise answers
            lines = clean_answer.split('\n')
            # Find the most concise line that looks like an answer
            for line in lines:
                line = line.strip()
                if line and len(line) < 200:
                    if any(char.isdigit() for char in line) or len(line.split()) <= 15:
                        clean_answer = line
                        break
        
        return clean_answer.strip()
    
    def _update_stats(self, processing_time: float, success: bool):
        """Update performance statistics."""
        self.performance_stats["total_questions"] += 1
        if success:
            self.performance_stats["successful_answers"] += 1
        
        # Update average processing time
        total_time = (self.performance_stats["avg_processing_time"] * 
                     (self.performance_stats["total_questions"] - 1) + processing_time)
        self.performance_stats["avg_processing_time"] = total_time / self.performance_stats["total_questions"]
    
    def get_performance_summary(self) -> str:
        """Get performance summary for monitoring."""
        stats = self.performance_stats
        uptime = time.time() - stats["start_time"]
        success_rate = (stats["successful_answers"] / max(1, stats["total_questions"])) * 100
        
        return f"""
🎯 **Advanced GAIA Agent Performance**
- Questions Processed: {stats["total_questions"]}
- Success Rate: {success_rate:.1f}%
- Avg Processing Time: {stats["avg_processing_time"]:.2f}s
- Uptime: {uptime/3600:.1f} hours
- Tools Available: {len(tools)}
- Advanced Features: ✅ Enabled
"""

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Enhanced GAIA evaluation with advanced agent capabilities.
    Fetches questions, processes with sophisticated reasoning, and submits answers.
    """
    # Determine space configuration
    space_id = os.getenv("SPACE_ID")
    
    if profile:
        username = f"{profile.username}"
        logger.info(f"👤 User logged in: {username}")
    else:
        logger.warning("❌ User not logged in")
        return "Please Login to Hugging Face with the button.", None
    
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    
    # Initialize Advanced GAIA Agent
    try:
        logger.info("🚀 Initializing Advanced GAIA Agent for evaluation...")
        agent = AdvancedGAIAAgent()
    except Exception as e:
        error_msg = f"❌ Error initializing advanced agent: {e}"
        logger.error(error_msg)
        return error_msg, None
    
    # Agent code URL for submission
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "Local deployment"
    logger.info(f"📂 Agent code location: {agent_code}")
    
    # Fetch questions from GAIA API
    logger.info(f"📥 Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        
        if not questions_data:
            logger.warning("⚠️ No questions received")
            return "No questions received from GAIA API.", None
            
        logger.info(f"✅ Fetched {len(questions_data)} questions for evaluation")
        
    except requests.exceptions.RequestException as e:
        error_msg = f"❌ Error fetching questions: {e}"
        logger.error(error_msg)
        return error_msg, None
    except Exception as e:
        error_msg = f"❌ Unexpected error fetching questions: {e}"
        logger.error(error_msg)
        return error_msg, None
    
    # Process questions with advanced agent
    results_log = []
    answers_payload = []
    start_time = time.time()
    
    logger.info(f"🧠 Starting advanced processing of {len(questions_data)} questions...")
    
    for i, item in enumerate(questions_data, 1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        
        if not task_id or question_text is None:
            logger.warning(f"⚠️ Skipping invalid question item: {item}")
            continue
        
        try:
            logger.info(f"🔄 Processing question {i}/{len(questions_data)} (ID: {task_id})")
            
            # Process with advanced agent
            question_start = time.time()
            submitted_answer = agent(question_text)
            question_time = time.time() - question_start
            
            # Store results
            answers_payload.append({
                "task_id": task_id, 
                "submitted_answer": submitted_answer
            })
            
            results_log.append({
                "Task ID": task_id,
                "Question": question_text[:200] + "..." if len(question_text) > 200 else question_text,
                "Submitted Answer": submitted_answer,
                "Processing Time": f"{question_time:.1f}s"
            })
            
            logger.info(f"✅ Question {i} completed in {question_time:.1f}s")
            
        except Exception as e:
            error_msg = f"AGENT ERROR: {e}"
            logger.error(f"❌ Error on question {i} (ID: {task_id}): {e}")
            
            results_log.append({
                "Task ID": task_id,
                "Question": question_text[:200] + "..." if len(question_text) > 200 else question_text,
                "Submitted Answer": error_msg,
                "Processing Time": "Error"
            })
    
    total_time = time.time() - start_time
    
    if not answers_payload:
        logger.error("❌ No answers generated")
        return "Advanced agent did not produce any answers to submit.", pd.DataFrame(results_log)
    
    # Prepare submission with enhanced data
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload
    }
    
    logger.info(f"📤 Submitting {len(answers_payload)} answers for user '{username}'")
    logger.info(f"⏱️ Total processing time: {total_time:.1f}s")
    logger.info(f"📊 Average time per question: {total_time/len(questions_data):.1f}s")
    
    # Submit to GAIA API
    try:
        logger.info(f"🚀 Submitting to: {submit_url}")
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        
        # Enhanced success message with performance stats
        performance_summary = agent.get_performance_summary()
        
        final_status = f"""
🎉 **GAIA Submission Successful!**

📊 **Results:**
- User: {result_data.get('username')}
- Overall Score: {result_data.get('score', 'N/A')}% 
- Correct: {result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')}
- Message: {result_data.get('message', 'No message received.')}

⏱️ **Performance:**
- Total Processing Time: {total_time:.1f}s
- Avg Time/Question: {total_time/len(questions_data):.1f}s

{performance_summary}
"""
        
        logger.info("🎉 GAIA submission completed successfully!")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
        
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except:
            error_detail += f" Response: {e.response.text[:500]}"
        
        status_message = f"❌ Submission Failed: {error_detail}"
        logger.error(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
        
    except Exception as e:
        status_message = f"❌ Unexpected submission error: {e}"
        logger.error(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df

def chat_with_advanced_agent(message: str, history: List[List[str]], session_id: str = None):
    """
    Interactive chat with the advanced agent (separate from GAIA evaluation).
    """
    try:
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Use the existing advanced chat logic from the original app
        # This preserves all the sophisticated features
        generator = parallel_pool.execute_agent_parallel(
            message=message,
            history=history, 
            log_to_db=LOGGING_ENABLED,
            session_id=session_id
        )
        
        # Return the generator for streaming
        for steps, response, updated_session_id in generator:
            yield steps, response, updated_session_id
            
    except Exception as e:
        error_msg = f"❌ Chat error: {e}"
        logger.error(error_msg)
        yield "Error", error_msg, session_id or str(uuid.uuid4())

# --- Enhanced Gradio Interface ---
def create_analytics_display():
    """Create enhanced analytics display."""
    try:
        analytics = session_manager.get_analytics_summary()
        
        return f"""
## 🎯 Advanced Agent Analytics

### 📊 Performance Metrics
- **Total Queries**: {analytics['performance']['total_queries']}
- **Cache Hits**: {analytics['cache_efficiency']['hits']} ({analytics['cache_efficiency']['hit_rate']:.1%})
- **Parallel Executions**: {analytics['performance']['parallel_executions']}
- **Active Sessions**: {analytics['active_sessions']}
- **Uptime**: {analytics['uptime_hours']:.1f} hours

### 🛠️ Tool Analytics
{chr(10).join([f"- **{tool}**: {stats['calls']} calls, {stats['successes']}/{stats['calls']} success, {stats['avg_time']:.2f}s avg" 
               for tool, stats in analytics['tool_analytics'].items() if stats['calls'] > 0])}

### ⚡ Parallel Pool Status
- **Max Workers**: {analytics['parallel_pool']['max_workers']}
- **Cache Size**: {analytics['parallel_pool']['cache_size']}
- **Total Requests**: {analytics['parallel_pool']['total_requests']}
- **Rate Limiting**: {"✅ Active" if analytics['parallel_pool']['rate_limiting_active'] else "❌ Inactive"}
"""
    except Exception as e:
        return f"❌ Error generating analytics: {e}"

# --- Build Enhanced Gradio Interface ---
def build_enhanced_gradio_interface():
    """Build comprehensive Gradio interface with GAIA and advanced features."""
    
    # Custom CSS for enhanced UI
    custom_css = """
    .gaia-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .feature-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .status-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
    }
    
    .status-error {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Advanced GAIA Agent") as demo:
        # Header
        gr.HTML("""
        <div class="gaia-header">
            <h1>🚀 Advanced GAIA Benchmark Agent</h1>
            <p>Sophisticated AI agent with strategic planning, reflection, and cross-validation capabilities</p>
        </div>
        """)
        
        # Main tabs
        with gr.Tabs():
            # GAIA Evaluation Tab
            with gr.TabItem("🎯 GAIA Evaluation", id="gaia_eval"):
                gr.Markdown("""
                ## 🧠 GAIA Benchmark Evaluation
                
                This agent features:
                - **Strategic Planning**: Analyzes questions and creates multi-step reasoning plans
                - **Cross-Validation**: Verifies answers through multiple sources and tools  
                - **Adaptive Reasoning**: Adjusts strategy based on question complexity
                - **Tool Orchestration**: Uses 15+ specialized tools for research and analysis
                - **Reflection & Error Recovery**: Learns from mistakes and improves responses
                
                ### Instructions:
                1. **Log in** to your Hugging Face account below
                2. **Click 'Run GAIA Evaluation'** to start the benchmark
                3. **Monitor progress** in real-time with detailed logging
                4. **View results** with comprehensive performance analytics
                """)
                
                gr.LoginButton()
                
                with gr.Row():
                    run_button = gr.Button(
                        "🚀 Run GAIA Evaluation & Submit All Answers", 
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        status_output = gr.Textbox(
                            label="📊 Evaluation Status & Results",
                            lines=15,
                            interactive=False,
                            placeholder="Click 'Run GAIA Evaluation' to start..."
                        )
                    
                    with gr.Column(scale=1):
                        analytics_display = gr.Markdown(
                            value=create_analytics_display(),
                            label="📈 Real-time Analytics"
                        )
                
                results_table = gr.DataFrame(
                    label="📋 Detailed Question Results",
                    wrap=True,
                    interactive=False
                )
                
                # Connect GAIA evaluation
                run_button.click(
                    fn=run_and_submit_all,
                    outputs=[status_output, results_table]
                )
            
            # Interactive Chat Tab  
            with gr.TabItem("💬 Advanced Agent Chat", id="advanced_chat"):
                gr.Markdown("""
                ## 🤖 Interactive Advanced Agent
                
                Chat with the same sophisticated agent used for GAIA evaluation:
                - Uses **strategic planning** for complex questions
                - Employs **15+ specialized tools** for research and analysis
                - Provides **cross-validated** answers with confidence assessment
                - Supports **multimedia** analysis (images, audio, video, documents)
                """)
                
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="🧠 Advanced Agent",
                            height=500,
                            show_copy_button=True
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="Your Message",
                                placeholder="Ask me anything - I'll use strategic reasoning and multiple tools to help...",
                                lines=2,
                                scale=4
                            )
                            send_btn = gr.Button("📤 Send", variant="primary")
                        
                        with gr.Row():
                            file_upload = gr.File(
                                label="📎 Upload File (images, documents, audio, video)",
                                file_types=["image", "audio", "video", ".pdf", ".docx", ".xlsx", ".txt", ".py"]
                            )
                            upload_btn = gr.Button("🔍 Analyze File")
                    
                    with gr.Column(scale=1):
                        session_state = gr.State(value=str(uuid.uuid4()))
                        
                        gr.Markdown("### 🎛️ Controls")
                        clear_btn = gr.Button("🔄 Clear Chat", variant="secondary")
                        
                        gr.Markdown("### 📊 Live Analytics")
                        live_analytics = gr.Markdown(
                            value=create_analytics_display()
                        )
                        
                        refresh_analytics = gr.Button("🔄 Refresh Analytics")
                
                # Chat functionality
                def respond(message, history, session_id):
                    if not message.strip():
                        return history, "", session_id
                    
                    # Add user message to history
                    history = history + [[message, None]]
                    
                    # Get response from advanced agent
                    try:
                        generator = chat_with_advanced_agent(message, history[:-1], session_id)
                        
                        # Stream the response
                        for steps, response, updated_session_id in generator:
                            # Update the last AI message in history
                            history[-1][1] = response
                            yield history, "", updated_session_id
                            
                    except Exception as e:
                        history[-1][1] = f"❌ Error: {e}"
                        yield history, "", session_id
                
                # Connect chat interactions
                send_btn.click(
                    fn=respond,
                    inputs=[msg_input, chatbot, session_state],
                    outputs=[chatbot, msg_input, session_state]
                )
                
                msg_input.submit(
                    fn=respond,
                    inputs=[msg_input, chatbot, session_state], 
                    outputs=[chatbot, msg_input, session_state]
                )
                
                def handle_file_upload(file_obj, history, session_id):
                    if file_obj is None:
                        return history, session_id
                    
                    # Process uploaded file
                    file_analysis = process_uploaded_file(file_obj)
                    
                    # Add to chat history
                    history = history + [[f"📎 Uploaded: {Path(file_obj.name).name}", file_analysis]]
                    return history, session_id
                
                upload_btn.click(
                    fn=handle_file_upload,
                    inputs=[file_upload, chatbot, session_state],
                    outputs=[chatbot, session_state]
                )
                
                clear_btn.click(
                    fn=lambda: ([], str(uuid.uuid4())),
                    outputs=[chatbot, session_state]
                )
                
                refresh_analytics.click(
                    fn=create_analytics_display,
                    outputs=[live_analytics]
                )
            
            # Documentation Tab
            with gr.TabItem("📚 Documentation", id="docs"):
                gr.Markdown("""
                # 📖 Advanced GAIA Agent Documentation
                
                ## 🎯 Overview
                This agent combines the power of strategic AI reasoning with specialized tools for GAIA benchmark evaluation.
                
                ## 🧠 Advanced Features
                
                ### Strategic Planning
                - **Query Analysis**: Automatically analyzes question type and complexity
                - **Multi-Step Planning**: Creates detailed execution plans before starting
                - **Plan Adaptation**: Dynamically adjusts strategy based on intermediate results
                
                ### Cross-Validation & Verification  
                - **Multiple Sources**: Verifies information through independent sources
                - **Tool Cross-Reference**: Uses different tools to confirm findings
                - **Confidence Assessment**: Tracks and reports confidence levels
                
                ### Tool Orchestration
                - **Web Research**: Wikipedia, Tavily search, general web scraping
                - **Document Analysis**: PDF, Word, Excel, text files
                - **Multimedia Processing**: Image analysis, audio transcription, video analysis  
                - **Computation**: Python interpreter for calculations and data processing
                - **Semantic Search**: GPU-accelerated vector search through knowledge bases
                
                ### Adaptive Intelligence
                - **Model Selection**: Chooses optimal models for different task types
                - **Error Recovery**: Intelligent retry strategies with alternative approaches
                - **Rate Limiting**: Respects API limits with exponential backoff
                - **Performance Monitoring**: Real-time analytics and optimization
                
                ## 🔧 Technical Architecture
                
                ### Multi-Model Configuration
                - **Reasoning Models**: Optimized for complex logical thinking
                - **Function Calling Models**: Specialized for tool interaction
                - **Text Generation Models**: High-quality response generation
                - **Vision Models**: Image and visual content analysis
                
                ### Advanced State Management
                - **Enhanced State Tracking**: Comprehensive reasoning state
                - **Reflection Capabilities**: Self-assessment and course correction
                - **Tool Performance Analytics**: Continuous improvement through metrics
                
                ## 🚀 Usage Tips
                
                ### For GAIA Evaluation
                1. Ensure stable internet connection for API calls
                2. Allow sufficient time for comprehensive reasoning
                3. Monitor progress through real-time status updates
                4. Review detailed results and performance analytics
                
                ### For Interactive Chat
                1. Ask complex, multi-step questions to see strategic planning in action
                2. Upload files for analysis (supports images, documents, audio, video)
                3. Request cross-validation for important information
                4. Monitor tool usage and performance through analytics
                
                ## 📊 Performance Optimization
                
                - **Parallel Processing**: Multiple worker threads with rate limiting
                - **Response Caching**: Intelligent caching with TTL management  
                - **GPU Acceleration**: CUDA-enabled embedding models when available
                - **Adaptive Timeouts**: Dynamic timeout adjustment based on complexity
                
                ## 🛡️ Error Handling & Recovery
                
                - **Exponential Backoff**: Automatic retry with intelligent delays
                - **Alternative Strategies**: Fallback approaches for different error types
                - **Graceful Degradation**: Continues operation even with partial tool failures
                - **Comprehensive Logging**: Detailed error tracking and performance monitoring
                """)
        
        return demo

# --- Main Application ---
if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("🚀 ADVANCED GAIA BENCHMARK AGENT STARTING")
    logger.info("="*60)
    
    # Environment info
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")
    
    if space_host:
        logger.info(f"✅ SPACE_HOST: {space_host}")
        logger.info(f"🌐 Runtime URL: https://{space_host}.hf.space")
    else:
        logger.info("ℹ️ Running locally (no SPACE_HOST)")
    
    if space_id:
        logger.info(f"✅ SPACE_ID: {space_id}")
        logger.info(f"📂 Repo URL: https://huggingface.co/spaces/{space_id}")
    else:
        logger.info("ℹ️ Local deployment (no SPACE_ID)")
    
    # Feature status
    logger.info(f"🛠️ Tools Available: {len(tools)}")
    logger.info(f"📊 Logging: {'✅ Enabled' if LOGGING_ENABLED else '❌ Disabled'}")
    logger.info(f"⚡ Advanced Features: ✅ Enabled")
    
    logger.info("="*60)
    
    # Build and launch interface
    logger.info("🎨 Building Enhanced Gradio Interface...")
    demo = build_enhanced_gradio_interface()
    
    logger.info("🚀 Launching Advanced GAIA Agent...")
    demo.launch(
        debug=True,
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    ) 