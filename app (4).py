"""Gradio Interface for AI Agent"""
import os
import gradio as gr
import requests
import pandas as pd
from langchain_core.messages import HumanMessage
from agent import build_agent_workflow

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Agent Definition ---
class BasicAgent:
    """A langgraph agent."""
    def __init__(self):
        print("BasicAgent initialized.")
        self.graph = build_agent_workflow()

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        messages = [HumanMessage(content=question)]
        # The invoke method returns the final state of the graph
        final_state = self.graph.invoke({"messages": messages})
        # The agent's response is in the last message
        answer = final_state['messages'][-1].content
        return answer

def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    if not profile:
        print("User not logged in.")
        return "Please Login with the Hugging Face button to run the evaluation.", None
    
    username = profile.username
    print(f"User logged in: {username}")
    
    space_id = os.getenv("SPACE_ID")
    if not space_id:
        print("SPACE_ID environment variable not found.")
        return "Could not determine the Space ID. Cannot proceed.", None

    questions_url = f"{DEFAULT_API_URL}/questions"
    submit_url = f"{DEFAULT_API_URL}/submit"

    # 1. Instantiate Agent
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(f"Agent Code URL: {agent_code}")

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None

    # 3. Run Agent on Questions
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    # 4. Prepare and Submit Answers
    submission_data = {"username": username, "agent_code": agent_code, "answers": answers_payload}
    print(f"Submitting {len(answers_payload)} answers...")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful! Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)"
        )
        print("Submission successful.")
    except requests.RequestException as e:
        final_status = f"Submission Failed: {e}"
        print(final_status)

    return final_status, pd.DataFrame(results_log)

# --- Build Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        1. Log in to your Hugging Face account using the button below.
        2. Click 'Run Evaluation & Submit All Answers' to run your agent and see the score.
        """
    )
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=3, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("Launching Gradio Interface...")
    demo.launch()