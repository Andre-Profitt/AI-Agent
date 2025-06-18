"""
GAIA Logic Module
=================

This module handles GAIA benchmark evaluation logic.
"""

import os
import time
import logging
import requests
import pandas as pd
from typing import List, Dict, Any, Optional
import gradio as gr

from config import config
from src.advanced_agent_fsm import FSMReActAgent
from src.tools_enhanced import get_enhanced_tools
from src.database import get_supabase_client, SupabaseLogHandler

logger = logging.getLogger(__name__)


class AdvancedGAIAAgent:
    """
    GAIA wrapper that delegates every question to the FSM-based agent,
    guaranteeing deterministic execution and GAIA-ready tools.
    """
    def __init__(self, log_handler: Optional[logging.Handler] = None):
        tools = get_enhanced_tools()
        self.fsm_agent = FSMReActAgent(
            tools=tools,
            log_handler=log_handler,
            model_preference="balanced"
        )
        self.performance_stats = {
            "total_questions": 0,
            "successful_answers": 0,
            "avg_processing_time": 0.0,
            "tool_usage": {tool.name: 0 for tool in tools},
            "start_time": time.time()
        }
        logger.info("Initialized AdvancedGAIAAgent with FSM backend")

    def __call__(self, question: str) -> str:
        """Process a single GAIA question"""
        start = time.time()
        try:
            result = self.fsm_agent.run({"input": question})
            answer = self._extract_clean_answer(result["output"])
            self._update_stats(time.time() - start, success=True)
            return answer
        except Exception as e:
            logger.error(f"Error processing GAIA question: {e}")
            self._update_stats(time.time() - start, success=False)
            return f"Error: {str(e)}"
    
    def _extract_clean_answer(self, response: str) -> str:
        """Extract clean answer for GAIA submission - no formatting, just the answer"""
        if not response:
            return "No answer provided"
        
        # Remove common formatting artifacts
        import re
        
        # Remove LaTeX boxing
        response = re.sub(r'\$\\boxed{([^}]+)}\$', r'\1', response)
        response = re.sub(r'\\boxed{([^}]+)}', r'\1', response)
        
        # Remove "final answer" prefixes
        response = re.sub(r'^(the\s+)?final\s+answer\s*(is\s*)?:?\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'^(my\s+)?answer\s*(is\s*)?:?\s*', '', response, flags=re.IGNORECASE)
        
        # Extract from common answer patterns
        answer_match = re.search(r'(?:answer|result)\s*(?:is|:)\s*([^\n.]+)', response, re.IGNORECASE)
        if answer_match:
            response = answer_match.group(1)
        
        # Clean up
        response = response.strip()
        response = response.strip('"\'.,!?()[]{}')
        
        # For GAIA, prefer concise answers
        if len(response) > 200:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Look for the most direct answer line
            for line in lines:
                if len(line) < 100 and line:
                    # Prefer short, factual answers
                    if (any(char.isdigit() for char in line) or 
                        len(line.split()) <= 10 or
                        line.isupper() or  # Country codes, etc.
                        ',' in line and len(line) < 80):  # Lists
                        response = line
                        break
            
            # If still too long, truncate
            if len(response) > 150:
                response = response[:150].strip()
        
        return response.strip() if response.strip() else "Unable to determine answer"
    
    def _update_stats(self, processing_time: float, success: bool):
        """Update performance statistics"""
        self.performance_stats["total_questions"] += 1
        if success:
            self.performance_stats["successful_answers"] += 1
        
        # Update average processing time
        total_time = (self.performance_stats["avg_processing_time"] * 
                     (self.performance_stats["total_questions"] - 1) + processing_time)
        self.performance_stats["avg_processing_time"] = total_time / self.performance_stats["total_questions"]
    
    def get_performance_summary(self) -> str:
        """Get performance summary for monitoring"""
        stats = self.performance_stats
        uptime = time.time() - stats["start_time"]
        success_rate = (stats["successful_answers"] / max(1, stats["total_questions"])) * 100
        
        return f"""
🎯 **Advanced GAIA Agent Performance**
- Questions Processed: {stats["total_questions"]}
- Success Rate: {success_rate:.1f}%
- Avg Processing Time: {stats["avg_processing_time"]:.2f}s
- Uptime: {uptime/3600:.1f} hours
- Tools Available: {len(self.fsm_agent.tools)}
- Advanced Features: ✅ Enabled
"""


class GAIAEvaluator:
    """Handles GAIA benchmark evaluation workflow"""
    
    def __init__(self):
        self.api_url = config.api.GAIA_API_URL
        self.questions_url = f"{self.api_url}/questions"
        self.submit_url = f"{self.api_url}/submit"
        
        # Initialize logging
        try:
            supabase_client = get_supabase_client()
            self.log_handler = SupabaseLogHandler(supabase_client)
            self.logging_enabled = True
        except Exception as e:
            logger.warning(f"Supabase logging disabled: {e}")
            self.log_handler = None
            self.logging_enabled = False
    
    def run_evaluation(self, profile: gr.OAuthProfile | None) -> tuple[str, pd.DataFrame | None]:
        """
        Run complete GAIA evaluation workflow
        
        Returns:
            Tuple of (status_message, results_dataframe)
        """
        # Check authentication
        if not profile:
            logger.warning("User not authenticated for GAIA evaluation")
            return "❌ Please Login to Hugging Face with the button.", None
        
        username = profile.username
        logger.info(f"Starting GAIA evaluation for user: {username}")
        
        # Initialize agent
        try:
            agent = AdvancedGAIAAgent(log_handler=self.log_handler)
        except Exception as e:
            error_msg = f"❌ Error initializing GAIA agent: {e}"
            logger.error(error_msg)
            return error_msg, None
        
        # Determine agent code location
        space_id = os.getenv("SPACE_ID")
        agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "Local deployment"
        
        # Fetch questions
        questions_data = self._fetch_questions()
        if isinstance(questions_data, str):  # Error message
            return questions_data, None
        
        # Process questions
        results_log, answers_payload = self._process_questions(agent, questions_data)
        
        if not answers_payload:
            return "❌ No answers generated by the agent.", pd.DataFrame(results_log)
        
        # Submit results
        submission_data = {
            "username": username.strip(),
            "agent_code": agent_code,
            "answers": answers_payload
        }
        
        status_message = self._submit_results(submission_data, agent, len(questions_data))
        
        # Create results dataframe
        results_df = pd.DataFrame(results_log)
        
        return status_message, results_df
    
    def _fetch_questions(self) -> List[Dict[str, Any]] | str:
        """Fetch questions from GAIA API"""
        logger.info(f"Fetching questions from: {self.questions_url}")
        
        try:
            response = requests.get(self.questions_url, timeout=15)
            response.raise_for_status()
            questions_data = response.json()
            
            if not questions_data:
                logger.warning("No questions received from GAIA API")
                return "⚠️ No questions received from GAIA API."
                
            logger.info(f"Successfully fetched {len(questions_data)} questions")
            return questions_data
            
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ Error fetching questions: {e}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"❌ Unexpected error fetching questions: {e}"
            logger.error(error_msg)
            return error_msg
    
    def _process_questions(self, agent: AdvancedGAIAAgent, 
                         questions_data: List[Dict[str, Any]]) -> tuple[List[Dict], List[Dict]]:
        """Process all questions with the agent"""
        results_log = []
        answers_payload = []
        start_time = time.time()
        
        logger.info(f"Starting processing of {len(questions_data)} questions")
        
        for i, item in enumerate(questions_data, 1):
            task_id = item.get("task_id")
            question_text = item.get("question")
            
            if not task_id or question_text is None:
                logger.warning(f"Skipping invalid question item: {item}")
                continue
            
            try:
                logger.info(f"Processing question {i}/{len(questions_data)} (ID: {task_id})")
                
                # Process with agent
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
                
                logger.info(f"Question {i} completed in {question_time:.1f}s")
                
            except Exception as e:
                error_msg = f"AGENT ERROR: {e}"
                logger.error(f"Error on question {i} (ID: {task_id}): {e}")
                
                results_log.append({
                    "Task ID": task_id,
                    "Question": question_text[:200] + "..." if len(question_text) > 200 else question_text,
                    "Submitted Answer": error_msg,
                    "Processing Time": "Error"
                })
        
        total_time = time.time() - start_time
        logger.info(f"Completed processing in {total_time:.1f}s "
                   f"(avg: {total_time/len(questions_data):.1f}s per question)")
        
        return results_log, answers_payload
    
    def _submit_results(self, submission_data: Dict[str, Any], 
                       agent: AdvancedGAIAAgent, total_questions: int) -> str:
        """Submit results to GAIA API"""
        logger.info(f"Submitting {len(submission_data['answers'])} answers")
        
        try:
            response = requests.post(self.submit_url, json=submission_data, timeout=60)
            response.raise_for_status()
            result_data = response.json()
            
            # Get performance summary
            performance_summary = agent.get_performance_summary()
            
            # Format success message
            final_status = f"""
🎉 **GAIA Submission Successful!**

📊 **Results:**
- User: {result_data.get('username')}
- Overall Score: {result_data.get('score', 'N/A')}% 
- Correct: {result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')}
- Message: {result_data.get('message', 'No message received.')}

{performance_summary}
"""
            
            logger.info("GAIA submission completed successfully!")
            return final_status
            
        except requests.exceptions.HTTPError as e:
            error_detail = f"Server responded with status {e.response.status_code}."
            try:
                error_json = e.response.json()
                error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
            except:
                error_detail += f" Response: {e.response.text[:500]}"
            
            status_message = f"❌ Submission Failed: {error_detail}"
            logger.error(status_message)
            return status_message
            
        except Exception as e:
            status_message = f"❌ Unexpected submission error: {e}"
            logger.error(status_message)
            return status_message


# Check if GAIA functionality is available
def check_gaia_availability() -> bool:
    """Check if GAIA agent.py is available"""
    try:
        from agent import build_graph
        return True
    except ImportError:
        logger.warning("GAIA agent.py not available - GAIA evaluation will be disabled")
        return False


# Global GAIA availability flag
GAIA_AVAILABLE = check_gaia_availability() 