import os
import uuid
import logging
import requests
import pandas as pd
from typing import List
import asyncio
import concurrent.futures
from functools import partial
import threading
import time

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.agent import ReActAgent
from src.database import get_supabase_client, SupabaseLogHandler
from src.tools import get_tools

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
MAX_PARALLEL_WORKERS = 8  # Number of parallel agent instances

# --- Initialize tools and components ---
try:
    tools = get_tools()
    logger.info("Tools initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize tools: {e}")
    tools = []

try:
    supabase_client = get_supabase_client()
    supabase_handler = SupabaseLogHandler(supabase_client)
    logger.info("Supabase client initialized successfully.")
    LOGGING_ENABLED = True
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    supabase_handler = None
    LOGGING_ENABLED = False

def build_graph():
    """Build the ReAct agent graph for GAIA questions."""
    try:
        react_agent = ReActAgent(tools=tools, log_handler=supabase_handler if LOGGING_ENABLED else None)
        return react_agent.graph
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        return None

# --- Enhanced Agent Definition with Parallel Processing ---
class ParallelAgent:
    """A langgraph agent with parallel processing capabilities."""
    def __init__(self, max_workers=MAX_PARALLEL_WORKERS):
        print(f"ParallelAgent initialized with {max_workers} workers.")
        self.max_workers = max_workers
        # Pre-build graphs for parallel execution
        self.graphs = [build_graph() for _ in range(max_workers)]

    def __call__(self, question: str, graph_index: int = 0) -> str:
        """Process a single question using the specified graph instance."""
        print(f"Agent processing question {graph_index} (first 50 chars): {question[:50]}...")
        try:
            # Use the specified graph instance for thread safety
            graph = self.graphs[graph_index % len(self.graphs)]
            messages = [HumanMessage(content=question)]
            result = graph.invoke({"messages": messages})
            answer = result['messages'][-1].content
            # Remove "Final Answer: " prefix if present
            if answer.startswith("Final Answer: "):
                return answer[14:]
            return answer
        except Exception as e:
            logger.error(f"Error processing question {graph_index}: {e}")
            return f"ERROR: {str(e)}"

    def process_questions_parallel(self, questions_data: List[dict]) -> List[dict]:
        """Process multiple questions in parallel using ThreadPoolExecutor."""
        print(f"Starting parallel processing of {len(questions_data)} questions with {self.max_workers} workers...")
        
        results_log = []
        start_time = time.time()
        
        def process_single_question(item_with_index):
            """Process a single question with error handling."""
            index, item = item_with_index
            task_id = item.get("task_id")
            question_text = item.get("question")
            
            if not task_id or question_text is None:
                logger.warning(f"Skipping item {index} with missing task_id or question: {item}")
                return {"Task ID": task_id or f"MISSING_{index}", "Question": "INVALID", "Submitted Answer": "SKIPPED"}
            
            try:
                print(f"[{index+1}/{len(questions_data)}] Processing task {task_id}...")
                submitted_answer = self(question_text, graph_index=index)
                elapsed = time.time() - start_time
                print(f"[{index+1}/{len(questions_data)}] Completed task {task_id} in {elapsed:.1f}s total")
                return {
                    "Task ID": task_id, 
                    "Question": question_text, 
                    "Submitted Answer": submitted_answer
                }
            except Exception as e:
                logger.error(f"Error processing task {task_id}: {e}")
                return {
                    "Task ID": task_id, 
                    "Question": question_text, 
                    "Submitted Answer": f"AGENT ERROR: {e}"
                }

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all questions for parallel processing
            indexed_questions = list(enumerate(questions_data))
            future_to_index = {
                executor.submit(process_single_question, item_with_index): index 
                for index, item_with_index in enumerate(indexed_questions)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results_log.append(result)
                except Exception as e:
                    logger.error(f"Future {index} generated an exception: {e}")
                    results_log.append({
                        "Task ID": f"ERROR_{index}", 
                        "Question": "PROCESSING ERROR", 
                        "Submitted Answer": f"FUTURE ERROR: {e}"
                    })
        
        total_time = time.time() - start_time
        print(f"Parallel processing completed in {total_time:.2f} seconds")
        print(f"Average time per question: {total_time/len(questions_data):.2f} seconds")
        
        return results_log

# --- Basic Agent Definition (Backward Compatibility) ---
class BasicAgent:
    """A langgraph agent."""
    def __init__(self):
        print("BasicAgent initialized.")
        self.graph = build_graph()

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        # Wrap the question in a HumanMessage from langchain_core
        messages = [HumanMessage(content=question)]
        messages = self.graph.invoke({"messages": messages})
        answer = messages['messages'][-1].content
        return answer[14:] if answer.startswith("Final Answer: ") else answer

def run_and_submit_all(use_parallel: bool = True):
    """
    Fetches all questions, runs the Agent on them (parallel or sequential), 
    submits all answers, and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    # For now, use a default username (in production, this would come from OAuth)
    username = "parallel_test_user"
    print(f"Using username: {username}")

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent
    try:
        if use_parallel:
            agent = ParallelAgent(max_workers=MAX_PARALLEL_WORKERS)
            processing_mode = f"Parallel (with {MAX_PARALLEL_WORKERS} workers)"
        else:
            agent = BasicAgent()
            processing_mode = "Sequential"
        print(f"Using {processing_mode} processing mode")
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    
    # In the case of an app running as a hugging Face space, this link points toward your codebase
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent (Parallel or Sequential)
    print(f"Running agent on {len(questions_data)} questions using {processing_mode} processing...")
    start_time = time.time()
    
    if use_parallel and hasattr(agent, 'process_questions_parallel'):
        # Use parallel processing
        results_log = agent.process_questions_parallel(questions_data)
    else:
        # Fallback to sequential processing
        results_log = []
        for i, item in enumerate(questions_data):
            task_id = item.get("task_id")
            question_text = item.get("question")
            if not task_id or question_text is None:
                print(f"Skipping item with missing task_id or question: {item}")
                continue
            try:
                print(f"[{i+1}/{len(questions_data)}] Processing task {task_id}...")
                submitted_answer = agent(question_text)
                results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
            except Exception as e:
                 print(f"Error running agent on task {task_id}: {e}")
                 results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    # 4. Prepare answers payload for submission
    answers_payload = []
    for result in results_log:
        if result.get("Task ID") and not result["Task ID"].startswith(("MISSING_", "ERROR_")):
            answers_payload.append({
                "task_id": result["Task ID"], 
                "submitted_answer": result["Submitted Answer"]
            })

    if not answers_payload:
        print("Agent did not produce any valid answers to submit.")
        return "Agent did not produce any valid answers to submit.", pd.DataFrame(results_log)

    # 5. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished in {total_time:.2f}s. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 6. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"Processing Mode: {processing_mode}\n"
            f"Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Parallel Agent Evaluation Runner")
    gr.Markdown(
        f"""
        **Instructions:**
        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Choose processing mode and click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.
        
        **Performance:**
        - **Parallel Mode**: Uses {MAX_PARALLEL_WORKERS} worker threads to process questions simultaneously (MUCH faster!)
        - **Sequential Mode**: Processes questions one by one (slower, for debugging)
        
        ---
        **Optimization Notes:**
        This version implements parallel processing to dramatically reduce runtime from 15+ minutes to just a few minutes by running multiple agent instances simultaneously.
        """
    )

    gr.LoginButton()
    
    with gr.Row():
        parallel_checkbox = gr.Checkbox(
            label="Enable Parallel Processing", 
            value=True, 
            info=f"Use {MAX_PARALLEL_WORKERS} parallel workers for faster execution"
        )

    run_button = gr.Button("Run Evaluation & Submit All Answers", variant="primary")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=8, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        inputs=[parallel_checkbox],
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print(f"✅ Parallel processing enabled with {MAX_PARALLEL_WORKERS} workers")
    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Parallel Agent Evaluation...")
    demo.launch(debug=True, share=False) 
