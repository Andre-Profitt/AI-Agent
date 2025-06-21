from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent

"""

from typing import Optional
GAIA Logic Module
=================

This module handles GAIA benchmark evaluation logic.
"""

from typing import Dict
from typing import Any

import os
import time
import logging
import requests
import pandas as pd
from typing import List, Dict, Any, Optional
import gradio as gr
import re

from .config import config
from src.agents.advanced_agent_fsm import FSMReActAgent
from src.utils.tools_enhanced import get_enhanced_tools
from .database import get_supabase_client, SupabaseLogHandler


logger = logging.getLogger(__name__)


class AdvancedGAIAAgent:
    """
    GAIA wrapper that delegates every question to the FSM-based agent,
    guaranteeing deterministic execution and GAIA-ready tools.
    """
    def __init__(self, log_handler: Optional[logging.Handler] = None) -> None:
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
            logger.error("Error processing GAIA question: {}", extra={"e": e})
            self._update_stats(time.time() - start, success=False)
            return f"Error: {str(e)}"
    
    def _extract_clean_answer(self, response: str) -> str:
        """Enhanced answer extraction for GAIA submission"""
        if not response:
            return "No answer provided"
        
        # Remove common formatting artifacts
        response = re.sub(r'\$\\boxed{([^}]+)}\$', r'\1', response)
        response = re.sub(r'\\boxed{([^}]+)}', r'\1', response)
        response = re.sub(r'final answer:', '', response, flags=re.IGNORECASE)
        response = re.sub(r'the answer is:', '', response, flags=re.IGNORECASE)
        
        # Handle numbered lists for multi-part answers
        if re.match(r'^\d+\.', response.strip()):
            lines = response.strip().split('\n')
            answers = []
            for line in lines:
                match = re.match(r'^\d+\.\s*(.+)', line.strip())
                if match:
                    answers.append(match.group(1).strip())
            if answers:
                response = ', '.join(answers)
        
        # Handle coordinate formats (common in GAIA)
        coord_match = re.search(r'(\d+\.?\d*)[°º]\s*([NS])\s*,?\s*(\d+\.?\d*)[°º]\s*([EW])', response)
        if coord_match:
            return f"{coord_match.group(1)}°{coord_match.group(2)}, {coord_match.group(3)}°{coord_match.group(4)}"
        
        # Handle date formats
        date_patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # MM/DD/YYYY or similar
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',     # YYYY-MM-DD
            r'(\w+)\s+(\d{1,2}),?\s+(\d{4})'          # Month DD, YYYY
        ]
        for pattern in date_patterns:
            if re.search(pattern, response):
                match = re.search(pattern, response)
                if match:
                    return match.group(0)
        
        # Handle currency amounts
        currency_match = re.search(r'[\$€£¥]\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', response)
        if currency_match:
            return currency_match.group(0)
        
        # Handle percentages
        percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response)
        if percent_match:
            return percent_match.group(0)
        
        # Handle scientific notation
        sci_match = re.search(r'(\d+(?:\.\d+)?)\s*[×x]\s*10\^?(-?\d+)', response)
        if sci_match:
            return f"{sci_match.group(1)}×10^{sci_match.group(2)}"
        
        # Clean up and return
        response = response.strip()
        response = re.sub(r'\s+', ' ', response)  # Normalize whitespace
        
        return response if response else "Unable to determine answer"
    
    def _update_stats(self, processing_time: float, success: bool) -> Any:
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
    
    def __init__(self) -> None:
        self.api_url = config.api.GAIA_API_URL
        self.questions_url = f"{self.api_url}/questions"
        self.submit_url = f"{self.api_url}/submit"
        
        # Initialize logging
        try:
            supabase_client = get_supabase_client()
            self.log_handler = SupabaseLogHandler(supabase_client)
            self.logging_enabled = True
        except Exception as e:
            logger.warning("Supabase logging disabled: {}", extra={"e": e})
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
        logger.info("Starting GAIA evaluation for user: {}", extra={"username": username})
        
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
        logger.info("Fetching questions from: {}", extra={"self_questions_url": self.questions_url})
        
        try:
            response = requests.get(self.questions_url, timeout=15)
            response.raise_for_status()
            questions_data = response.json()
            
            if not questions_data:
                logger.warning("No questions received from GAIA API")
                return "⚠️ No questions received from GAIA API."
            
            # Debug log the first question to see its structure
            if questions_data:
                first_question = questions_data[0]
                logger.info("First question structure: {}", extra={"first_question": first_question})
                logger.info("Question type: {}", extra={"type_first_question_get__question___": type(first_question.get('question'))})
                logger.info("Question value: {}", extra={"first_question_get__question__": first_question.get('question')})
                
            logger.info("Successfully fetched {} questions", extra={"len_questions_data_": len(questions_data)})
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
        
        logger.info("Starting processing of {} questions", extra={"len_questions_data_": len(questions_data)})
        
        for i, item in enumerate(questions_data, 1):
            task_id = item.get("task_id")
            question_text = item.get("question")
            
            if not task_id or question_text is None:
                logger.warning("Skipping invalid question item: {}", extra={"item": item})
                continue
                
            # Validate question_text is a string
            if not isinstance(question_text, str):
                error_msg = f"Question text must be a string, got {type(question_text)}"
                logger.error("Error on question {} (ID: {}): {}", extra={"i": i, "task_id": task_id, "error_msg": error_msg})
                
                results_log.append({
                    "Task ID": task_id,
                    "Question": str(question_text)[:200] + "..." if len(str(question_text)) > 200 else str(question_text),
                    "Submitted Answer": error_msg,
                    "Processing Time": "Error"
                })
                continue
            
            try:
                logger.info("Processing question {}/{} (ID: {})", extra={"i": i, "len_questions_data_": len(questions_data), "task_id": task_id})
                
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
                
                logger.info("Question {} completed in {}s", extra={"i": i, "question_time": question_time})
                
            except Exception as e:
                error_msg = f"AGENT ERROR: {e}"
                logger.error("Error on question {} (ID: {}): {}", extra={"i": i, "task_id": task_id, "e": e})
                
                results_log.append({
                    "Task ID": task_id,
                    "Question": question_text[:200] + "..." if len(question_text) > 200 else question_text,
                    "Submitted Answer": error_msg,
                    "Processing Time": "Error"
                })
        
        total_time = time.time() - start_time
        avg_time = total_time / len(questions_data) if questions_data else 0
        logger.info(
            "Completed processing in {}s (avg: {:.2f}s per question)".format(total_time, avg_time),
            extra={"total_time": total_time, "total_time_len_questions_data_": avg_time}
        )
        
        return results_log, answers_payload
    
    def _submit_results(self, submission_data: Dict[str, Any], 
                       agent: AdvancedGAIAAgent, total_questions: int) -> str:
        """Submit results to GAIA API"""
        logger.info("Submitting {} answers", extra={"len_submission_data__answers__": len(submission_data['answers'])})
        
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

async def run_gaia_evaluation(self, username: str, password: str) -> Any:
    logger.info("Starting GAIA evaluation", extra={
        "operation": "gaia_evaluation",
        "username": username
    })
    
    # ... existing code ...
    
    logger.info("Successfully fetched questions", extra={
        "operation": "fetch_questions",
        "questions_count": len(questions_data),
        "url": self.questions_url
    })
    
    # ... existing code ...
    
    logger.info("Question processing completed", extra={
        "operation": "process_questions",
        "total_questions": len(questions_data),
        "total_time": total_time
    }) 