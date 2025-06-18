"""
Interactive Tools Module
Implements tools that require user interaction, including clarification
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# Global state for managing interactive sessions
class InteractiveState:
    """Manages state for interactive tool calls"""
    def __init__(self):
        self.pending_clarifications: Dict[str, Dict] = {}
        self.clarification_callback: Optional[Callable] = None
        self.user_responses: Dict[str, str] = {}
    
    def set_clarification_callback(self, callback: Callable):
        """Set the callback function for handling clarifications"""
        self.clarification_callback = callback
    
    def add_pending_clarification(self, question_id: str, question: str, context: Dict):
        """Add a pending clarification request"""
        self.pending_clarifications[question_id] = {
            "question": question,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_user_response(self, question_id: str) -> Optional[str]:
        """Get user response for a clarification"""
        return self.user_responses.get(question_id)
    
    def set_user_response(self, question_id: str, response: str):
        """Set user response for a clarification"""
        self.user_responses[question_id] = response
        # Clean up pending clarification
        if question_id in self.pending_clarifications:
            del self.pending_clarifications[question_id]


# Global interactive state instance
interactive_state = InteractiveState()


class ClarificationInput(BaseModel):
    """Input schema for clarification tool"""
    question: str = Field(description="The clarifying question to ask the user")
    context: Optional[str] = Field(
        default=None,
        description="Optional context about why this clarification is needed"
    )


class UserFeedbackInput(BaseModel):
    """Input schema for user feedback tool"""
    feedback_type: str = Field(
        description="Type of feedback: 'rating', 'correction', 'suggestion'"
    )
    content: str = Field(description="The feedback content")
    related_to: Optional[str] = Field(
        default=None,
        description="What this feedback relates to (e.g., 'last_response', 'tool_output')"
    )


def ask_user_for_clarification(question: str, context: Optional[str] = None) -> str:
    """
    Asks the user a clarifying question to resolve ambiguity or gather missing information.
    
    This tool should be used before calling another tool if you are unsure about any 
    parameters or if the user's request is underspecified. The function will return 
    the user's answer.
    
    Args:
        question: The clarifying question to ask the user
        context: Optional context about why this clarification is needed
        
    Returns:
        The user's response to the clarification question
    """
    import uuid
    question_id = str(uuid.uuid4())
    
    # Store the pending clarification
    interactive_state.add_pending_clarification(
        question_id,
        question,
        {"context": context} if context else {}
    )
    
    # If there's a callback registered (from the UI), invoke it
    if interactive_state.clarification_callback:
        try:
            # This will trigger the UI to show the question and wait for response
            response = interactive_state.clarification_callback(question_id, question, context)
            return response
        except Exception as e:
            return f"Error getting clarification: {str(e)}. Proceeding with available information."
    
    # Fallback for testing or non-interactive mode
    return f"[Clarification needed: {question}] (No interactive session available)"


def request_user_approval(
    action_description: str,
    details: Dict[str, Any],
    modification_allowed: bool = True
) -> Dict[str, Any]:
    """
    Request user approval before executing a critical action.
    
    This tool presents a planned action to the user for approval. The user can
    approve, reject, or modify the action before it proceeds.
    
    Args:
        action_description: Brief description of the action requiring approval
        details: Dictionary containing action details (e.g., tool calls, parameters)
        modification_allowed: Whether the user can modify the action
        
    Returns:
        Dictionary with approval status and potentially modified action details
    """
    import uuid
    approval_id = str(uuid.uuid4())
    
    # Format details for presentation
    formatted_details = json.dumps(details, indent=2)
    
    # Create approval request
    approval_request = {
        "id": approval_id,
        "action": action_description,
        "details": details,
        "formatted_details": formatted_details,
        "modification_allowed": modification_allowed,
        "timestamp": datetime.now().isoformat()
    }
    
    # If there's a UI callback, use it
    if hasattr(interactive_state, 'approval_callback') and interactive_state.approval_callback:
        try:
            result = interactive_state.approval_callback(approval_request)
            return result
        except Exception as e:
            return {
                "approved": False,
                "reason": f"Error getting approval: {str(e)}",
                "modified_details": None
            }
    
    # Fallback for non-interactive mode
    return {
        "approved": False,
        "reason": "No interactive session available",
        "modified_details": None
    }


def collect_user_feedback(
    feedback_type: str,
    content: str,
    related_to: Optional[str] = None
) -> Dict[str, str]:
    """
    Collect feedback from the user about the agent's performance.
    
    This tool allows the agent to proactively ask for feedback on its responses
    or actions, which can be used for learning and improvement.
    
    Args:
        feedback_type: Type of feedback ('rating', 'correction', 'suggestion')
        content: The feedback prompt or question
        related_to: What this feedback relates to
        
    Returns:
        Dictionary with feedback data
    """
    import uuid
    feedback_id = str(uuid.uuid4())
    
    feedback_request = {
        "id": feedback_id,
        "type": feedback_type,
        "prompt": content,
        "related_to": related_to,
        "timestamp": datetime.now().isoformat()
    }
    
    # Store feedback request for potential async processing
    if hasattr(interactive_state, 'feedback_requests'):
        interactive_state.feedback_requests[feedback_id] = feedback_request
    
    # If there's a feedback callback, use it
    if hasattr(interactive_state, 'feedback_callback') and interactive_state.feedback_callback:
        try:
            feedback = interactive_state.feedback_callback(feedback_request)
            return {
                "feedback_id": feedback_id,
                "type": feedback_type,
                "feedback": feedback,
                "status": "collected"
            }
        except Exception as e:
            return {
                "feedback_id": feedback_id,
                "type": feedback_type,
                "feedback": None,
                "status": f"error: {str(e)}"
            }
    
    return {
        "feedback_id": feedback_id,
        "type": feedback_type,
        "feedback": None,
        "status": "pending (no interactive session)"
    }


def pause_for_user_input(prompt: str, input_type: str = "text") -> str:
    """
    Pause execution and wait for user input.
    
    This is a general-purpose tool for collecting user input when needed.
    
    Args:
        prompt: The prompt to show the user
        input_type: Type of input expected ('text', 'number', 'boolean', 'choice')
        
    Returns:
        The user's input as a string
    """
    import uuid
    input_id = str(uuid.uuid4())
    
    input_request = {
        "id": input_id,
        "prompt": prompt,
        "input_type": input_type,
        "timestamp": datetime.now().isoformat()
    }
    
    # If there's an input callback, use it
    if hasattr(interactive_state, 'input_callback') and interactive_state.input_callback:
        try:
            user_input = interactive_state.input_callback(input_request)
            return str(user_input)
        except Exception as e:
            return f"Error getting user input: {str(e)}"
    
    return f"[User input needed: {prompt}] (No interactive session)"


# Create tool instances
ask_user_for_clarification_tool = StructuredTool.from_function(
    func=ask_user_for_clarification,
    name="ask_user_for_clarification",
    description="""Asks the user a clarifying question to resolve ambiguity or gather missing information. 
    Use this tool before calling another tool if you are unsure about any parameters or if the user's 
    request is underspecified. The function will return the user's answer.""",
    args_schema=ClarificationInput
)

request_user_approval_tool = StructuredTool.from_function(
    func=request_user_approval,
    name="request_user_approval",
    description="""Request user approval before executing a critical action. 
    The user can approve, reject, or modify the action before it proceeds."""
)

collect_user_feedback_tool = StructuredTool.from_function(
    func=collect_user_feedback,
    name="collect_user_feedback",
    description="""Collect feedback from the user about the agent's performance. 
    Use this to ask for ratings, corrections, or suggestions.""",
    args_schema=UserFeedbackInput
)

pause_for_user_input_tool = StructuredTool.from_function(
    func=pause_for_user_input,
    name="pause_for_user_input",
    description="""Pause execution and wait for user input. 
    Use this when you need specific information from the user that wasn't provided."""
)


# Helper class for managing clarification patterns
class ClarificationPatternTracker:
    """Tracks patterns in clarification requests for learning"""
    
    def __init__(self):
        self.patterns: List[Dict] = []
    
    def add_pattern(
        self,
        original_query: str,
        clarification_question: str,
        user_response: str,
        context: Dict
    ):
        """Record a clarification pattern"""
        pattern = {
            "original_query": original_query,
            "clarification_question": clarification_question,
            "user_response": user_response,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        self.patterns.append(pattern)
    
    def find_similar_patterns(self, query: str, threshold: float = 0.7) -> List[Dict]:
        """Find similar clarification patterns"""
        # Simple implementation - in production, use embeddings
        similar = []
        query_words = set(query.lower().split())
        
        for pattern in self.patterns:
            pattern_words = set(pattern["original_query"].lower().split())
            similarity = len(query_words & pattern_words) / len(query_words | pattern_words)
            
            if similarity >= threshold:
                similar.append({
                    "pattern": pattern,
                    "similarity": similarity
                })
        
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)


# Global pattern tracker
clarification_tracker = ClarificationPatternTracker()


# Export all interactive tools
INTERACTIVE_TOOLS = [
    ask_user_for_clarification_tool,
    request_user_approval_tool,
    collect_user_feedback_tool,
    pause_for_user_input_tool
]


# Utility function for FSM integration
def get_interactive_tools() -> List[StructuredTool]:
    """Get all interactive tools for agent registration"""
    return INTERACTIVE_TOOLS 