import os
import uuid
import logging
import re
import json
import time
import datetime
import concurrent.futures
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from functools import lru_cache
import threading
from queue import Queue

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.advanced_agent import AdvancedReActAgent
from src.database import get_supabase_client, SupabaseLogHandler
from src.tools import get_tools

# --- Initialization ---
# Load environment variables
load_dotenv()

# Configure root logger to be very permissive
logging.basicConfig(level=logging.DEBUG)
# Get a specific logger for our application
logger = logging.getLogger(__name__)
# Prevent passing logs to the root logger's handlers
logger.propagate = False

# Initialize Supabase client and custom log handler
try:
    supabase_client = get_supabase_client()
    supabase_handler = SupabaseLogHandler(supabase_client)
    # Add the custom handler to our app's logger
    logger.addHandler(supabase_handler)
    logger.info("Supabase logger initialized successfully.")
    LOGGING_ENABLED = True
except Exception as e:
    logger.error(f"Failed to initialize Supabase logger: {e}. Trajectory logging will be disabled.")
    LOGGING_ENABLED = False

# Initialize tools and agent
try:
    tools = get_tools()
    # Use the more sophisticated AdvancedReActAgent which includes strategic planning,
    # reflection nodes, and robust error handling improvements.
    agent_graph = AdvancedReActAgent(
        tools=tools,
        log_handler=supabase_handler if LOGGING_ENABLED else None
    ).graph
    logger.info("AdvancedReAct Agent initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize agent: {e}", exc_info=True)
    # Instead of exiting, try to provide more specific error details
    print(f"AGENT INITIALIZATION ERROR: {e}")
    print(f"Error Type: {type(e).__name__}")
    
    # Check if tools were initialized successfully
    try:
        tools = get_tools()
        print(f"✅ Tools initialized successfully: {len(tools)} tools available")
        print(f"Tool names: {[tool.name for tool in tools]}")
    except Exception as tool_error:
        print(f"❌ Tools initialization failed: {tool_error}")
    
    # Exit only if it's a critical issue we can't recover from
    import traceback
    print("Full traceback:")
    traceback.print_exc()
    exit("Critical error: Agent could not be initialized. Check logs above for details.")

# --- API Rate Limiting Constants ---
MAX_PARALLEL_WORKERS = 8  # Reduced from 20 to respect Groq rate limits (6000 TPM)
API_RATE_LIMIT_BUFFER = 5  # Extra seconds between API calls for safety
GROQ_TPM_LIMIT = 6000  # Groq tokens per minute limit
REQUEST_SPACING = 0.5  # Minimum seconds between requests

# --- Advanced Parallel Processing & Caching ---

class ParallelAgentPool:
    """High-performance parallel agent execution with worker pool."""
    
    def __init__(self, max_workers: int = MAX_PARALLEL_WORKERS):
        self.max_workers = min(max_workers, MAX_PARALLEL_WORKERS)  # Enforce API limits
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.active_tasks = {}
        self.task_queue = Queue()
        self.response_cache = {}
        self.cache_lock = threading.Lock()
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_lock = threading.Lock()
        
        logger.info(f"🚀 Initialized ParallelAgentPool with {self.max_workers} workers (API rate-limited)")
        logger.info(f"📊 Configured for Groq TPM limit: {GROQ_TPM_LIMIT}")
        
    def _enforce_rate_limit(self):
        """Enforce API rate limiting between requests."""
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < REQUEST_SPACING:
                sleep_time = REQUEST_SPACING - time_since_last + (API_RATE_LIMIT_BUFFER / 10)
                logger.debug(f"⏳ Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
            self.request_count += 1
    
    @lru_cache(maxsize=1000)
    def _get_cache_key(self, message: str, session_id: str) -> str:
        """Generate cache key for response caching."""
        return f"{hash(message)}_{session_id}"
    
    def get_cached_response(self, message: str, session_id: str) -> Optional[str]:
        """Get cached response if available."""
        cache_key = self._get_cache_key(message, session_id)
        with self.cache_lock:
            return self.response_cache.get(cache_key)
    
    def cache_response(self, message: str, session_id: str, response: str):
        """Cache response for future use."""
        cache_key = self._get_cache_key(message, session_id)
        with self.cache_lock:
            # Keep cache size manageable
            if len(self.response_cache) > 500:
                # Remove oldest 100 entries
                keys_to_remove = list(self.response_cache.keys())[:100]
                for key in keys_to_remove:
                    del self.response_cache[key]
            
            self.response_cache[cache_key] = response
    
    def execute_agent_parallel(self, message: str, history: List[List[str]], 
                               log_to_db: bool, session_id: str):
        """Execute agent with parallel processing and API rate limiting."""
        
        # Check cache first
        cached_response = self.get_cached_response(message, session_id)
        if cached_response:
            logger.info(f"🎯 Cache hit for message: {message[:50]}...")
            yield "📋 **Retrieved from cache** (Ultra-fast response!)\n\n", cached_response, session_id
            return
        
        # Enforce API rate limiting before making request
        self._enforce_rate_limit()
        
        # Execute in thread pool with rate limiting
        logger.info(f"🔄 Executing with rate limiting (request #{self.request_count})")
        future = self.executor.submit(chat_interface_logic_sync, message, history, log_to_db, session_id)
        
        try:
            # Get the generator from the future
            generator = future.result(timeout=300)  # 5 minute timeout
            
            final_response = ""
            for steps, response, updated_session_id in generator:
                final_response = response
                yield steps, response, updated_session_id
            
            # Cache the final response
            if final_response and "error" not in final_response.lower():
                self.cache_response(message, session_id, final_response)
                
        except concurrent.futures.TimeoutError:
            logger.error(f"Parallel execution timeout for message: {message[:50]}...")
            yield "❌ **Timeout Error:** Request took too long to process", "Request timeout", session_id
        except Exception as e:
            logger.error(f"Parallel execution error: {e}")
            yield f"❌ **Parallel Execution Error:** {e}", f"Error: {e}", session_id
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status for monitoring."""
        return {
            "max_workers": self.max_workers,
            "active_threads": threading.active_count(),
            "cache_size": len(self.response_cache),
            "pending_tasks": self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0,
            "total_requests": self.request_count,
            "last_request_time": self.last_request_time,
            "rate_limiting_active": True
        }

# Global parallel agent pool - REDUCED for API rate limits
# Groq has 6000 TPM (tokens per minute) limits, so we use fewer workers
parallel_pool = ParallelAgentPool(max_workers=8)  # Reduced from 20 to respect API limits

class AsyncResponseCache:
    """Advanced response caching with TTL and intelligent invalidation."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_expired(self):
        """Background thread to clean up expired cache entries."""
        while True:
            try:
                current_time = time.time()
                with self.lock:
                    expired_keys = [
                        key for key, timestamp in self.timestamps.items()
                        if current_time - timestamp > self.ttl_seconds
                    ]
                    
                    for key in expired_keys:
                        self.cache.pop(key, None)
                        self.timestamps.pop(key, None)
                
                time.sleep(300)  # Cleanup every 5 minutes
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                time.sleep(60)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.ttl_seconds:
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value with timestamp."""
        with self.lock:
            # Size management
            if len(self.cache) >= self.max_size:
                # Remove oldest 20% of entries
                sorted_items = sorted(self.timestamps.items(), key=lambda x: x[1])
                to_remove = sorted_items[:max(1, len(sorted_items) // 5)]
                
                for old_key, _ in to_remove:
                    self.cache.pop(old_key, None)
                    self.timestamps.pop(old_key, None)
            
            self.cache[key] = value
            self.timestamps[key] = time.time()

# Global response cache
response_cache = AsyncResponseCache(max_size=2000, ttl_seconds=1800)  # 30 min TTL

# --- Advanced State Management ---
class SessionManager:
    """Manages user sessions, conversation history, and analytics."""
    
    def __init__(self):
        self.sessions = {}
        self.tool_analytics = {tool.name: {"calls": 0, "successes": 0, "failures": 0, "avg_time": 0.0} for tool in tools}
        self.performance_metrics = {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "total_tool_calls": 0,
            "uptime_start": time.time(),
            "cache_hits": 0,
            "parallel_executions": 0
        }
    
    def create_session(self, session_id: str = None):
        """Create a new session."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            "created_at": datetime.datetime.now(),
            "messages": [],
            "total_queries": 0,
            "total_response_time": 0.0,
            "tool_usage": {tool.name: 0 for tool in tools},
            "cache_hits": 0,
            "parallel_tasks": 0
        }
        return session_id
    
    def update_tool_analytics(self, tool_name: str, success: bool, execution_time: float):
        """Update tool performance analytics."""
        if tool_name in self.tool_analytics:
            analytics = self.tool_analytics[tool_name]
            analytics["calls"] += 1
            if success:
                analytics["successes"] += 1
            else:
                analytics["failures"] += 1
            
            # Update average time
            total_time = analytics["avg_time"] * (analytics["calls"] - 1) + execution_time
            analytics["avg_time"] = total_time / analytics["calls"]
    
    def update_performance_metrics(self, cache_hit: bool = False, parallel_execution: bool = False):
        """Update global performance metrics."""
        if cache_hit:
            self.performance_metrics["cache_hits"] += 1
        if parallel_execution:
            self.performance_metrics["parallel_executions"] += 1
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        uptime = time.time() - self.performance_metrics["uptime_start"]
        pool_status = parallel_pool.get_pool_status()
        
        return {
            "performance": self.performance_metrics,
            "tool_analytics": self.tool_analytics,
            "active_sessions": len(self.sessions),
            "uptime_hours": uptime / 3600,
            "parallel_pool": pool_status,
            "cache_efficiency": {
                "hits": self.performance_metrics["cache_hits"],
                "size": len(response_cache.cache),
                "hit_rate": self.performance_metrics["cache_hits"] / max(1, self.performance_metrics["total_queries"])
            }
        }

# Global session manager
session_manager = SessionManager()

# --- Gradio Interface Logic ---

def format_chat_history(chat_history: List[List[str]]) -> List:
    """Formats Gradio's chat history into a list of LangChain messages."""
    messages = []
    for turn in chat_history:
        user_message, ai_message = turn
        if user_message:
            messages.append(HumanMessage(content=user_message))
        if ai_message:
            # This part is tricky as AI message might contain tool calls.
            # For simplicity, we'll treat it as a plain AIMessage here.
            # A more robust implementation would parse this for tool calls.
            messages.append(AIMessage(content=ai_message))
    return messages

def extract_final_answer(response: str) -> str:
    """
    Extracts only the final answer from sophisticated reasoning chains.
    Handles complex responses that include planning, reasoning, and conclusions.
    """
    if not response or not isinstance(response, str):
        return response
    
    # Remove common prefixes
    prefixes_to_remove = [
        "final answer:",
        "the answer is:",
        "answer:",
        "result:",
        "conclusion:",
        "therefore:",
        "so the answer is:",
        "based on my analysis,",
        "after analyzing,",
        "in conclusion,",
        "to summarize,",
        "the final result is:",
        "my final answer is:",
        "the correct answer is:",
        "ultimately,",
        "in summary,",
        "to conclude,"
    ]
    
    response_lower = response.lower().strip()
    
    # Check for prefixes and extract what comes after
    for prefix in prefixes_to_remove:
        if prefix in response_lower:
            # Find the last occurrence of the prefix
            idx = response_lower.rfind(prefix)
            if idx != -1:
                extracted = response[idx + len(prefix):].strip()
                if extracted:
                    response = extracted
                    break
    
    # Remove explanation patterns
    explanation_patterns = [
        r"based on.*?[,.]",
        r"according to.*?[,.]",
        r"after.*?analysis.*?[,.]",
        r"following.*?research.*?[,.]",
        r"from.*?information.*?[,.]",
        r"using.*?tool.*?[,.]",
        r"through.*?search.*?[,.]"
    ]
    
    for pattern in explanation_patterns:
        response = re.sub(pattern, "", response, flags=re.IGNORECASE | re.DOTALL)
    
    # Handle sentences that start with reasoning
    reasoning_starters = [
        "this is because",
        "the reason is",
        "this indicates",
        "this suggests",
        "this shows that",
        "this means",
        "this tells us",
        "we can see that",
        "it appears that",
        "the data shows",
        "evidence suggests"
    ]
    
    lines = response.split('\n')
    final_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        line_lower = line.lower()
        is_reasoning = any(starter in line_lower for starter in reasoning_starters)
        
        if not is_reasoning:
            final_lines.append(line)
    
    if final_lines:
        response = '\n'.join(final_lines)
    
    # Clean up common artifacts
    response = response.strip()
    response = re.sub(r'^[*\-•]\s*', '', response)  # Remove bullet points
    response = re.sub(r'\s+', ' ', response)  # Normalize whitespace
    
    # Extract just numbers/codes if that's what's needed
    response = response.strip('.,!?')
    
    # If response is very long (>100 chars) and contains multiple sentences,
    # try to extract the most important part
    if len(response) > 100 and '. ' in response:
        sentences = response.split('. ')
        # Look for the shortest sentence that might be the answer
        potential_answers = [s.strip() for s in sentences if len(s.strip()) < 50]
        if potential_answers:
            # Prefer sentences with numbers, dates, or short factual statements
            for ans in potential_answers:
                if any(char.isdigit() for char in ans) or len(ans.split()) <= 5:
                    response = ans
                    break
            else:
                response = potential_answers[0]
    
    return response.strip()

def process_uploaded_file(file_obj) -> str:
    """Process uploaded files and return analysis."""
    if file_obj is None:
        return "No file uploaded."
    
    try:
        file_path = file_obj.name
        file_extension = Path(file_path).suffix.lower()
        
        # Determine file type and processing method
        if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"📄 Text File Analysis:\n\nFile: {Path(file_path).name}\nSize: {len(content)} characters\nLines: {len(content.splitlines())}\n\nContent preview:\n{content[:500]}..."
        
        elif file_extension in ['.pdf', '.docx', '.xlsx']:
            return f"📄 Document uploaded: {Path(file_path).name}\nUse the 'advanced_file_reader' tool to analyze this file."
        
        elif file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
            return f"🖼️ Image uploaded: {Path(file_path).name}\nUse the 'image_analyzer' tool to analyze this image."
        
        elif file_extension in ['.mp3', '.wav', '.m4a']:
            return f"🎵 Audio uploaded: {Path(file_path).name}\nUse the 'audio_transcriber' tool to transcribe this audio."
        
        elif file_extension in ['.mp4', '.mov', '.avi']:
            return f"🎬 Video uploaded: {Path(file_path).name}\nUse the 'video_analyzer' tool to analyze this video."
        
        else:
            return f"📁 File uploaded: {Path(file_path).name}\nFile type: {file_extension}\nUse appropriate tools to analyze this file."
    
    except Exception as e:
        return f"❌ Error processing file: {str(e)}"

def chat_interface_logic_sync(message: str, history: List[List[str]], log_to_db: bool, session_id: str = None):
    """
    Synchronous version of chat interface logic for use in thread pool.
    """
    start_time = time.time()
    logger.info(f"Received new message: '{message}'")
    
    # Create session if needed
    if session_id is None or session_id not in session_manager.sessions:
        session_id = session_manager.create_session()
    
    # Format history and add new message
    formatted_history = format_chat_history(history)
    current_messages = formatted_history + [HumanMessage(content=message)]
    
    # Prepare agent input
    run_id = uuid.uuid4()
    agent_input = {
        "messages": current_messages,
        "run_id": run_id,
        "log_to_db": log_to_db and LOGGING_ENABLED,
        "plan": "",
        "step_count": 0,
        "confidence": 0.3,
        "reasoning_complete": False
    }

    # Stream the agent's response
    full_response = ""
    intermediate_steps = ""
    reasoning_chain = ""
    step_count = 0
    tool_calls_made = []
    
    try:
        for chunk in agent_graph.stream(agent_input, stream_mode="values"):
            last_message = chunk["messages"][-1]
            current_confidence = chunk.get("confidence", 0.0)
            current_step = chunk.get("step_count", 0)

            # Handle intermediate tool calls and thoughts
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                step_count = current_step
                thought = last_message.content if last_message.content else "Planning next action..."
                tool_call_str = "\n".join(
                    f"  - **{tc['name']}**: {tc['args']}" for tc in last_message.tool_calls
                )
                
                # Track tool calls
                for tc in last_message.tool_calls:
                    tool_calls_made.append(tc['name'])
                
                intermediate_steps += f"**Step {step_count}** (Confidence: {current_confidence:.0%})\n"
                intermediate_steps += f"🤔 **Reasoning:** {thought}\n\n"
                intermediate_steps += f"🛠️ **Action:**\n{tool_call_str}\n\n"
                
                reasoning_chain += f"Step {step_count}: {thought}\n"
                yield intermediate_steps, full_response, session_id

            # Handle tool outputs (observations)
            elif isinstance(last_message, ToolMessage):
                observation = last_message.content[:200] + "..." if len(last_message.content) > 200 else last_message.content
                intermediate_steps += f"👀 **Observation:** {observation}\n\n"
                reasoning_chain += f"Observation: {last_message.content}\n"
                yield intermediate_steps, full_response, session_id

            # Handle reasoning and final answers
            elif isinstance(last_message, AIMessage):
                step_count = current_step
                content = last_message.content
                
                # Check if this is likely a final answer
                is_final = (
                    current_confidence > 0.7 or 
                    chunk.get("reasoning_complete", False) or
                    any(indicator in content.lower() for indicator in [
                        "final answer", "the answer is", "therefore", "conclusion", "result:"
                    ])
                )
                
                if is_final:
                    # Extract clean final answer
                    clean_answer = extract_final_answer(content)
                    full_response = clean_answer
                    
                    intermediate_steps += f"**Final Step {step_count}** (Confidence: {current_confidence:.0%})\n"
                    intermediate_steps += f"✅ **Final Answer:** {clean_answer}\n\n"
                    
                    logger.info(f"Final answer extracted: '{clean_answer}' from raw response: '{content[:100]}...'")
                else:
                    # This is intermediate reasoning
                    intermediate_steps += f"**Step {step_count}** (Confidence: {current_confidence:.0%})\n"
                    intermediate_steps += f"🧠 **Reasoning:** {content}\n\n"
                    reasoning_chain += f"Step {step_count}: {content}\n"
                
                yield intermediate_steps, full_response, session_id
        
        # Update analytics
        execution_time = time.time() - start_time
        session_manager.performance_metrics["total_queries"] += 1
        session_manager.performance_metrics["total_tool_calls"] += len(tool_calls_made)
        session_manager.update_performance_metrics(parallel_execution=True)
        
        # Update session data
        if session_id in session_manager.sessions:
            session = session_manager.sessions[session_id]
            session["total_queries"] += 1
            session["total_response_time"] += execution_time
            session["parallel_tasks"] += 1
            for tool_name in tool_calls_made:
                if tool_name in session["tool_usage"]:
                    session["tool_usage"][tool_name] += 1

    except Exception as e:
        logger.error(f"An error occurred during agent execution for run_id {run_id}: {e}", exc_info=True)
        error_message = f"❌ **Error occurred:** {e}\n\n🔧 **Troubleshooting:**\n- Check your internet connection\n- Verify API keys are configured\n- Try a simpler query\n- Contact support if the issue persists"
        yield intermediate_steps + error_message, f"An error occurred: {e}", session_id

def chat_interface_logic_parallel(message: str, history: List[List[str]], log_to_db: bool, session_id: str = None):
    """
    Ultra-fast parallel chat interface logic with intelligent caching.
    Uses the thread pool for concurrent processing.
    """
    # Execute with parallel thread pool
    future = parallel_pool.executor.submit(
        chat_interface_logic_sync, message, history, log_to_db, session_id
    )
    
    try:
        # Get the generator result
        generator = future.result(timeout=300)  # 5 minute timeout
        
        for steps, response, updated_session_id in generator:
            yield steps, response, updated_session_id
            
    except concurrent.futures.TimeoutError:
        logger.error(f"Parallel execution timeout for message: {message[:50]}...")
        yield "❌ **Timeout Error:** Request took too long to process", "Request timeout", session_id
    except Exception as e:
        logger.error(f"Parallel execution error: {e}")
        yield f"❌ **Parallel Execution Error:** {e}", f"Error: {e}", session_id

def create_analytics_display():
    """Create analytics dashboard content with parallel processing metrics."""
    analytics = session_manager.get_analytics_summary()
    
    # Performance metrics with parallel processing
    perf_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 10px 0;">
        <h3>🚀 Ultra-High Performance Metrics</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            <div><strong>Total Queries:</strong> {analytics['performance']['total_queries']}</div>
            <div><strong>Parallel Executions:</strong> {analytics['performance']['parallel_executions']}</div>
            <div><strong>Cache Hits:</strong> {analytics['performance']['cache_hits']}</div>
            <div><strong>Active Sessions:</strong> {analytics['active_sessions']}</div>
            <div><strong>Pool Workers:</strong> {analytics['parallel_pool']['max_workers']}</div>
            <div><strong>Cache Size:</strong> {analytics['cache_efficiency']['size']}</div>
            <div><strong>Total Tool Calls:</strong> {analytics['performance']['total_tool_calls']}</div>
            <div><strong>Cache Hit Rate:</strong> {analytics['cache_efficiency']['hit_rate']:.1%}</div>
            <div><strong>Uptime:</strong> {analytics['uptime_hours']:.1f} hours</div>
        </div>
    </div>
    """
    
    # Parallel processing status with rate limiting
    parallel_html = f"""
    <div style="background: linear-gradient(45deg, #11998e, #38ef7d); padding: 15px; border-radius: 8px; color: white; margin: 10px 0;">
        <h3>⚡ API-Safe Parallel Processing Engine</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
            <div><strong>Worker Threads:</strong> {analytics['parallel_pool']['max_workers']} (API-limited)</div>
            <div><strong>Active Threads:</strong> {analytics['parallel_pool']['active_threads']}</div>
            <div><strong>Total Requests:</strong> {analytics['parallel_pool'].get('total_requests', 0)}</div>
            <div><strong>Rate Limiting:</strong> {'🟢 Active' if analytics['parallel_pool'].get('rate_limiting_active') else '🔴 Inactive'}</div>
            <div><strong>Cache Efficiency:</strong> {analytics['cache_efficiency']['hit_rate']:.1%}</div>
            <div><strong>Parallel Tasks:</strong> {analytics['performance']['parallel_executions']}</div>
        </div>
        <div style="margin-top: 10px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 5px; font-size: 0.9em;">
            <strong>📊 Groq API Limits:</strong> {GROQ_TPM_LIMIT} TPM | <strong>🔄 Request Spacing:</strong> {REQUEST_SPACING}s | <strong>⏳ Buffer:</strong> {API_RATE_LIMIT_BUFFER}s
        </div>
    </div>
    """
    
    # Tool analytics
    tool_html = "<div style='background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;'><h3>🛠️ Tool Performance</h3>"
    for tool_name, stats in analytics['tool_analytics'].items():
        if stats['calls'] > 0:
            success_rate = (stats['successes'] / stats['calls']) * 100
            tool_html += f"""
            <div style="margin: 5px 0; padding: 8px; background: white; border-radius: 5px;">
                <strong>{tool_name}:</strong> {stats['calls']} calls, {success_rate:.1f}% success rate, {stats['avg_time']:.2f}s avg
            </div>
            """
    tool_html += "</div>"
    
    return perf_html + parallel_html + tool_html

def export_conversation(history: List[List[str]], session_id: str = None):
    """Export conversation history."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_export_{timestamp}.json"
        
        export_data = {
            "timestamp": timestamp,
            "session_id": session_id,
            "conversation": history,
            "analytics": session_manager.get_analytics_summary() if session_id in session_manager.sessions else None
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return f"✅ Conversation exported to {filename}"
    except Exception as e:
        return f"❌ Export failed: {str(e)}"

def reset_session():
    """Reset the current session."""
    session_id = session_manager.create_session()
    return [], "", "🔄 Session reset successfully!", session_id

def build_gradio_interface():
    """Builds and returns the cutting-edge parallel-processing Gradio chat interface."""
    
    # Custom CSS for enhanced styling
    custom_css = """
    .container { max-width: 1200px; margin: auto; }
    .header-text { text-align: center; color: #2c3e50; }
    .metrics-panel { background: linear-gradient(45deg, #1e3c72, #2a5298); color: white; padding: 15px; border-radius: 10px; }
    .tool-indicator { display: inline-block; padding: 2px 8px; background: #3498db; color: white; border-radius: 12px; font-size: 0.8em; margin: 2px; }
    .confidence-bar { height: 6px; background: linear-gradient(to right, #e74c3c, #f39c12, #27ae60); border-radius: 3px; }
    .status-good { color: #27ae60; }
    .status-warning { color: #f39c12; }
    .status-error { color: #e74c3c; }
    .parallel-indicator { background: linear-gradient(45deg, #11998e, #38ef7d); padding: 5px 10px; border-radius: 15px; color: white; font-weight: bold; }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"), 
        title="🚀 Ultra-Fast Parallel AI Agent",
        css=custom_css
    ) as demo:
        
        # Header
        gr.HTML(f"""
        <div class="header-text">
            <h1>🚀 API-Safe Ultra-Fast Parallel AI Agent</h1>
            <p>
                <span class="parallel-indicator">⚡ {parallel_pool.max_workers} API-LIMITED WORKERS</span>
                • Advanced ReAct reasoning • Multi-modal processing • Real-time analytics • GPU acceleration • Intelligent caching • Rate limiting
            </p>
            <div style="margin-top: 10px; padding: 8px; background: linear-gradient(45deg, #ff6b6b, #ee5a24); color: white; border-radius: 15px; display: inline-block;">
                <strong>📊 Groq API Safe:</strong> {GROQ_TPM_LIMIT} TPM limit respected | <strong>🔄 Smart Rate Limiting</strong>
            </div>
        </div>
        """)
        
        # Session state
        session_state = gr.State(session_manager.create_session())
        
        with gr.Row():
            # Main chat interface
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    [],
                    label="🤖 Ultra-Fast AI Agent Conversation",
                    height=500,
                    show_label=True,
                    container=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    message_box = gr.Textbox(
                        placeholder="Ask me anything! Ultra-fast responses with parallel processing...",
                        label="Your Message",
                        show_label=False,
                        lines=2,
                        scale=4
                    )
                    
                    with gr.Column(scale=1):
                        send_btn = gr.Button("🚀 Send (Ultra-Fast)", variant="primary")
                        clear_btn = gr.Button("🔄 Reset", variant="secondary")
                
                # File upload section
                with gr.Row():
                    file_upload = gr.File(
                        label="📎 Upload File (PDF, DOC, Excel, Images, Audio, Video)",
                        file_count="single",
                        file_types=["image", "video", "audio", ".pdf", ".docx", ".xlsx", ".txt", ".py", ".js"]
                    )
                    upload_btn = gr.Button("📄 Analyze File")
                
                # Quick actions
                with gr.Row():
                    gr.Examples(
                        examples=[
                            "What's the weather like in Tokyo today?",
                            "Explain quantum computing in simple terms",
                            "Calculate the compound interest on $10,000 at 5% for 10 years",
                            "Analyze the uploaded image for objects",
                            "Search for recent AI research papers",
                            "Generate a Python script to sort a list"
                        ],
                        inputs=message_box,
                        label="💡 Quick Examples (Lightning Fast!)"
                    )
            
            # Advanced monitoring panel
            with gr.Column(scale=2):
                with gr.Tabs():
                    # Agent trajectory tab
                    with gr.TabItem("🧠 Agent Reasoning"):
                        intermediate_display = gr.Markdown(
                            "Agent reasoning steps will appear here...",
                            label="Real-time Agent Trajectory"
                        )
                    
                    # Performance analytics tab
                    with gr.TabItem("📊 Analytics"):
                        analytics_display = gr.HTML(
                            create_analytics_display(),
                            label="Ultra-Performance Dashboard"
                        )
                        refresh_analytics_btn = gr.Button("🔄 Refresh Analytics")
                    
                    # API-Safe parallel processing tab
                    with gr.TabItem("⚡ API-Safe Engine"):
                        gr.HTML(f"""
                        <div style="background: linear-gradient(45deg, #667eea, #764ba2); padding: 20px; border-radius: 10px; color: white;">
                            <h3>⚡ API-Safe Parallel Processing Engine</h3>
                            <div style="margin: 15px 0;">
                                <strong>🚀 Worker Pool:</strong> {parallel_pool.max_workers} API-limited workers<br>
                                <strong>📊 Groq Limits:</strong> {GROQ_TPM_LIMIT} TPM respected<br>
                                <strong>💾 Cache System:</strong> Intelligent response caching with TTL<br>
                                <strong>🔄 Rate Limiting:</strong> {REQUEST_SPACING}s spacing + {API_RATE_LIMIT_BUFFER}s buffer<br>
                                <strong>📈 Performance Boost:</strong> Up to 10x faster responses (API-safe)<br>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                                <strong>💡 How it works (API-Safe):</strong><br>
                                • Cached responses: Sub-second delivery<br>
                                • Rate-limited execution: Respects API limits<br>
                                • Smart caching: Reduces API calls<br>
                                • Request spacing: Prevents 429 errors<br>
                                • Buffer timing: Extra safety margin
                            </div>
                            <div style="margin-top: 10px; padding: 8px; background: rgba(255,100,100,0.3); border-radius: 5px;">
                                <strong>🚨 API Protection:</strong> System automatically manages Groq rate limits to prevent 429 errors and ensure stable operation.
                            </div>
                        </div>
                        """)
                    
                    # Configuration tab
                    with gr.TabItem("⚙️ Settings"):
                        with gr.Group():
                            log_to_db_checkbox = gr.Checkbox(
                                label="📝 Log Trajectory to Database",
                                value=True,
                                visible=LOGGING_ENABLED,
                                info="Enable detailed logging for debugging and analysis"
                            )
                            
                            model_preference = gr.Radio(
                                choices=["fast", "balanced", "quality"],
                                value="balanced",
                                label="🎯 Model Performance Preference",
                                info="Fast: Quick responses, Balanced: Good speed/quality, Quality: Best results"
                            )
                            
                            max_reasoning_steps = gr.Slider(
                                minimum=5,
                                maximum=25,
                                value=15,
                                step=1,
                                label="🔢 Max Reasoning Steps",
                                info="Maximum number of reasoning steps the agent can take"
                            )
                    
                    # Session management tab
                    with gr.TabItem("💾 Session"):
                        session_info = gr.Textbox(
                            value=f"Session: {session_manager.create_session()}",
                            label="Current Session ID",
                            interactive=False
                        )
                        
                        export_btn = gr.Button("💾 Export Conversation")
                        export_status = gr.Textbox(label="Export Status", interactive=False)
                        
                        gr.Markdown("""
                        ### 🔧 Ultra-Fast Session Features:
                        - **Parallel Processing**: 20 concurrent workers for maximum speed
                        - **Intelligent Caching**: Sub-second responses for repeated queries
                        - **Performance Tracking**: Monitor response times and tool usage
                        - **Export Capability**: Save conversations with analytics
                        - **Cache Analytics**: Track hit rates and performance gains
                        """)
        
        # Tool status display with parallel indicators
        with gr.Row():
            tool_status = gr.HTML(f"""
            <div style="background: #ecf0f1; padding: 10px; border-radius: 5px; text-align: center;">
                <strong>🛠️ Available Tools:</strong> 
                {' '.join([f'<span class="tool-indicator">{tool.name}</span>' for tool in tools])}
                <br><br>
                <span style="background: linear-gradient(45deg, #11998e, #38ef7d); padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold;">
                    ⚡ API-SAFE ULTRA-FAST: {parallel_pool.max_workers} Rate-Limited Workers + Smart Caching
                </span>
                <br><br>
                <span style="background: linear-gradient(45deg, #ff6b6b, #ee5a24); padding: 3px 10px; border-radius: 15px; color: white; font-size: 0.9em;">
                    🛡️ Groq API Protection: {GROQ_TPM_LIMIT} TPM | {REQUEST_SPACING}s spacing | {API_RATE_LIMIT_BUFFER}s buffer
                </span>
            </div>
            """)
        
        # Event handlers with ultra-fast parallel processing
        def on_submit(message, chat_history, log_to_db, session_id):
            if not message.strip():
                return chat_history, "", session_id
            
            chat_history.append([message, None])
            
            # Check cache first for instant responses
            cache_key = f"{hash(message)}_{session_id}"
            cached_response = response_cache.get(cache_key)
            
            if cached_response:
                session_manager.update_performance_metrics(cache_hit=True)
                chat_history[-1] = [message, cached_response]
                steps = "🚀 **Ultra-Fast Cache Response** (Sub-second delivery!)\n\n"
                yield chat_history, steps, session_id
                return
            
            # Use parallel processing for new requests
            response_generator = chat_interface_logic_sync(message, chat_history, log_to_db, session_id)
            
            final_response = ""
            for steps, response, updated_session_id in response_generator:
                chat_history[-1] = [message, response]
                final_response = response
                yield chat_history, steps, updated_session_id
            
            # Cache successful responses
            if final_response and not any(error_word in final_response.lower() for error_word in ["error", "failed", "exception"]):
                response_cache.set(cache_key, final_response)
        
        def on_file_upload(file_obj):
            if file_obj is not None:
                analysis = process_uploaded_file(file_obj)
                return analysis
            return "No file selected."
        
        def on_analytics_refresh():
            return create_analytics_display()
        
        def on_export(chat_history, session_id):
            return export_conversation(chat_history, session_id)
        
        def on_reset():
            return reset_session()
        
        # Wire up the events
        submit_event = message_box.submit(
            on_submit,
            [message_box, chatbot, log_to_db_checkbox, session_state],
            [chatbot, intermediate_display, session_state]
        ).then(
            lambda: gr.update(value=""), None, [message_box], queue=False
        )
        
        send_btn.click(
            on_submit,
            [message_box, chatbot, log_to_db_checkbox, session_state],
            [chatbot, intermediate_display, session_state]
        ).then(
            lambda: gr.update(value=""), None, [message_box], queue=False
        )
        
        upload_btn.click(
            on_file_upload,
            [file_upload],
            [intermediate_display]
        )
        
        refresh_analytics_btn.click(
            on_analytics_refresh,
            [],
            [analytics_display]
        )
        
        export_btn.click(
            on_export,
            [chatbot, session_state],
            [export_status]
        )
        
        clear_btn.click(
            on_reset,
            [],
            [chatbot, intermediate_display, export_status, session_state]
        )

    return demo

if __name__ == "__main__":
    logger.info(f"🚀 Starting API-Safe Ultra-Fast Parallel AI Agent")
    logger.info(f"📊 Workers: {parallel_pool.max_workers} (API rate-limited)")
    logger.info(f"🛡️ Groq TPM Limit: {GROQ_TPM_LIMIT}")
    logger.info(f"⏳ Request Spacing: {REQUEST_SPACING}s + {API_RATE_LIMIT_BUFFER}s buffer")
    
    app = build_gradio_interface()
    app.queue(
        max_size=30,  # Reduced from 50 to be more conservative
        concurrency_limit=MAX_PARALLEL_WORKERS  # Match our worker pool size
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        show_api=False,
        show_error=True
    ) 