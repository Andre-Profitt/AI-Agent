"""
GAIA Benchmark Agent Wrapper
Integrates the advanced AdvancedReActAgent with GAIA benchmark requirements
while preserving all sophisticated features like strategic planning, reflection,
cross-validation, and adaptive reasoning.
"""

import logging
from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from src.advanced_agent import AdvancedReActAgent
from src.tools import get_tools
from src.database import get_supabase_client, SupabaseLogHandler
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Supabase client and logging (optional)
try:
    supabase_client = get_supabase_client()
    supabase_handler = SupabaseLogHandler(supabase_client)
    LOGGING_ENABLED = True
    logger.info("GAIA Agent: Supabase logging initialized")
except Exception as e:
    logger.warning(f"GAIA Agent: Supabase logging disabled: {e}")
    supabase_handler = None
    LOGGING_ENABLED = False

def build_graph():
    """
    Builds and returns the advanced agent graph for GAIA benchmark evaluation.
    This function is required by the GAIA template.
    
    Returns:
        The compiled LangGraph agent with all advanced features enabled.
    """
    try:
        # Initialize tools
        tools = get_tools()
        logger.info(f"GAIA Agent: Initialized {len(tools)} tools")
        
        # Create the advanced agent with all sophisticated features
        advanced_agent = AdvancedReActAgent(
            tools=tools,
            log_handler=supabase_handler if LOGGING_ENABLED else None,
            model_preference="quality"  # Use highest quality models for GAIA
        )
        
        logger.info("GAIA Agent: AdvancedReActAgent initialized successfully")
        return advanced_agent.graph
        
    except Exception as e:
        logger.error(f"GAIA Agent: Failed to build graph: {e}")
        raise RuntimeError(f"Failed to initialize GAIA agent: {e}")

class GAIAAgent:
    """
    GAIA Benchmark Agent that leverages all advanced features:
    - Strategic planning and multi-step reasoning
    - Cross-validation and verification
    - Adaptive model selection
    - Tool orchestration and error recovery
    - Reflection and confidence assessment
    """
    
    def __init__(self):
        """Initialize the GAIA agent with advanced capabilities."""
        logger.info("Initializing GAIA Agent with advanced features...")
        
        try:
            self.graph = build_graph()
            self.session_count = 0
            logger.info("GAIA Agent initialized successfully")
        except Exception as e:
            logger.error(f"GAIA Agent initialization failed: {e}")
            raise
    
    def __call__(self, question: str) -> str:
        """
        Process a GAIA question and return the final answer.
        
        This method orchestrates the sophisticated reasoning process:
        1. Strategic planning based on question analysis
        2. Systematic execution with tool orchestration
        3. Cross-validation and verification
        4. Reflection and adaptive decision making
        5. Extract clean final answer for GAIA submission
        
        Args:
            question (str): The GAIA benchmark question
            
        Returns:
            str: Clean final answer ready for GAIA submission
        """
        try:
            self.session_count += 1
            logger.info(f"GAIA Agent: Processing question #{self.session_count}")
            logger.debug(f"Question: {question[:100]}...")
            
            # Create message format for the advanced agent
            messages = [HumanMessage(content=question)]
            
            # Enhanced state for GAIA processing
            enhanced_state = {
                "messages": messages,
                "run_id": uuid.uuid4(),
                "log_to_db": LOGGING_ENABLED,
                # Strategic planning state
                "master_plan": [],
                "current_step": 0,
                "plan_revisions": 0,
                # Reflection state
                "reflections": [],
                "confidence_history": [],
                "error_recovery_attempts": 0,
                # Adaptive intelligence
                "step_count": 0,
                "confidence": 0.3,  # Start conservative for GAIA
                "reasoning_complete": False,
                "verification_level": "thorough",  # Default to thorough for GAIA
                # Tool performance tracking
                "tool_success_rates": {},
                "tool_results": [],
                "cross_validation_sources": []
            }
            
            # Invoke the advanced agent graph
            logger.info("GAIA Agent: Starting advanced reasoning process...")
            result = self.graph.invoke(enhanced_state)
            
            # Extract the final answer from the sophisticated response
            final_message = result['messages'][-1]
            if isinstance(final_message, AIMessage):
                raw_answer = final_message.content
            else:
                raw_answer = str(final_message)
            
            # Clean and extract the final answer for GAIA submission
            clean_answer = self._extract_gaia_answer(raw_answer)
            
            logger.info(f"GAIA Agent: Question processed successfully")
            logger.debug(f"Raw answer length: {len(raw_answer)}, Clean answer: {clean_answer[:100]}...")
            
            return clean_answer
            
        except Exception as e:
            error_msg = f"GAIA Agent Error: {str(e)}"
            logger.error(error_msg)
            return f"Error processing question: {str(e)}"
    
    def _extract_gaia_answer(self, response: str) -> str:
        """
        Extract clean, submission-ready answer from sophisticated agent response.
        
        The advanced agent provides detailed reasoning, but GAIA needs clean answers.
        This method intelligently extracts the final answer while preserving accuracy.
        
        Args:
            response (str): Full response from advanced agent
            
        Returns:
            str: Clean answer ready for GAIA submission
        """
        if not response or not isinstance(response, str):
            return "No answer provided"
        
        # Remove common reasoning prefixes that the advanced agent might use
        answer_indicators = [
            "final answer:",
            "the answer is:",
            "answer:",
            "result:",
            "conclusion:",
            "therefore:",
            "so the answer is:",
            "based on my analysis:",
            "after thorough investigation:",
            "following comprehensive research:",
            "upon cross-validation:",
            "with high confidence:",
            "the verified answer is:",
            "conclusively:",
            "definitively:",
            "the final result is:",
            "my final answer is:",
            "the correct answer is:",
            "ultimately:",
            "in conclusion:",
            "to summarize:",
            "strategic analysis concludes:",
            "after reflection:",
            "with verification:"
        ]
        
        response_lower = response.lower().strip()
        
        # Find the last occurrence of answer indicators for best extraction
        best_answer = response
        for indicator in answer_indicators:
            if indicator in response_lower:
                idx = response_lower.rfind(indicator)
                if idx != -1:
                    potential_answer = response[idx + len(indicator):].strip()
                    if potential_answer and len(potential_answer) < len(best_answer):
                        best_answer = potential_answer
        
        # Clean up the extracted answer
        lines = best_answer.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that look like reasoning/explanation
            line_lower = line.lower()
            if any(skip_phrase in line_lower for skip_phrase in [
                "this is because", "the reason is", "based on", "according to",
                "evidence shows", "research indicates", "analysis reveals",
                "cross-validation confirms", "verification shows"
            ]):
                continue
            
            clean_lines.append(line)
        
        if clean_lines:
            best_answer = '\n'.join(clean_lines)
        
        # Final cleaning
        best_answer = best_answer.strip()
        
        # Remove bullet points and formatting
        import re
        best_answer = re.sub(r'^[*\-•]\s*', '', best_answer, flags=re.MULTILINE)
        best_answer = re.sub(r'\s+', ' ', best_answer)  # Normalize whitespace
        
        # For very long answers, try to extract the most essential part
        if len(best_answer) > 200:
            sentences = best_answer.split('. ')
            if len(sentences) > 1:
                # Look for short, factual sentences that might be the answer
                short_sentences = [s.strip() for s in sentences if len(s.strip()) < 100]
                if short_sentences:
                    # Prefer sentences with numbers, codes, or specific facts
                    for sentence in short_sentences:
                        if any(char.isdigit() for char in sentence) or len(sentence.split()) <= 10:
                            best_answer = sentence
                            break
                    else:
                        best_answer = short_sentences[0]
        
        # Final cleanup
        best_answer = best_answer.strip('.,!?')
        
        return best_answer.strip() if best_answer.strip() else "Unable to determine answer"

# For backward compatibility and direct usage
def create_gaia_agent():
    """Create and return a GAIA agent instance."""
    return GAIAAgent() 