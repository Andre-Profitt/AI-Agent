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
            return f"Error getting clarification: {str(e)}"
    
    # Fallback: return a message indicating clarification is needed
    return f"Clarification needed: {question}\nContext: {context or 'No additional context'}\nPlease provide your response."


def collect_user_feedback(feedback_type: str, content: str, related_to: Optional[str] = None) -> str:
    """
    Collects feedback from the user about the agent's performance or responses.
    
    Args:
        feedback_type: Type of feedback ('rating', 'correction', 'suggestion')
        content: The feedback content
        related_to: What this feedback relates to
        
    Returns:
        Confirmation message
    """
    try:
        # Store feedback (in a real implementation, this would go to a database)
        feedback_data = {
            "type": feedback_type,
            "content": content,
            "related_to": related_to,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log the feedback
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"User feedback collected: {feedback_data}")
        
        return f"Thank you for your {feedback_type} feedback. It has been recorded."
        
    except Exception as e:
        return f"Error collecting feedback: {str(e)}"


def request_user_confirmation(action_description: str, details: Optional[str] = None) -> str:
    """
    Requests user confirmation before performing a potentially important action.
    
    Args:
        action_description: Description of the action to be performed
        details: Additional details about the action
        
    Returns:
        User's confirmation or denial
    """
    import uuid
    confirmation_id = str(uuid.uuid4())
    
    confirmation_message = f"Please confirm the following action:\n{action_description}"
    if details:
        confirmation_message += f"\n\nDetails: {details}"
    
    # Store the pending confirmation
    interactive_state.add_pending_clarification(
        confirmation_id,
        confirmation_message,
        {"type": "confirmation", "action": action_description}
    )
    
    # If there's a callback registered, invoke it
    if interactive_state.clarification_callback:
        try:
            response = interactive_state.clarification_callback(confirmation_id, confirmation_message, {"type": "confirmation"})
            return response
        except Exception as e:
            return f"Error getting confirmation: {str(e)}"
    
    # Fallback
    return f"Confirmation needed: {confirmation_message}\nPlease respond with 'yes' or 'no'."


def get_interactive_tools() -> List[StructuredTool]:
    """
    Returns the complete set of interactive tools.
    """
    clarification_tool = StructuredTool.from_function(
        func=ask_user_for_clarification,
        name="ask_user_for_clarification",
        description="Ask the user a clarifying question when you need more information or are unsure about parameters.",
        args_schema=ClarificationInput
    )
    
    feedback_tool = StructuredTool.from_function(
        func=collect_user_feedback,
        name="collect_user_feedback",
        description="Collect feedback from the user about your performance or responses.",
        args_schema=UserFeedbackInput
    )
    
    confirmation_tool = StructuredTool.from_function(
        func=request_user_confirmation,
        name="request_user_confirmation",
        description="Request user confirmation before performing important actions.",
        args_schema=ClarificationInput
    )
    
    return [clarification_tool, feedback_tool, confirmation_tool]


def set_clarification_callback(callback: Callable):
    """
    Set the callback function for handling clarifications in the UI.
    
    Args:
        callback: Function that takes (question_id, question, context) and returns user response
    """
    interactive_state.set_clarification_callback(callback)


def get_pending_clarifications() -> Dict[str, Dict]:
    """
    Get all pending clarifications.
    
    Returns:
        Dictionary of pending clarifications
    """
    return interactive_state.pending_clarifications.copy()


def clear_pending_clarifications():
    """Clear all pending clarifications."""
    interactive_state.pending_clarifications.clear()
    interactive_state.user_responses.clear()


class ToolsInteractive:
    """Interactive tools class for importing"""
    
    def __init__(self):
        self.tools = get_interactive_tools()
        self.state = interactive_state
    
    def get_tools(self) -> List[StructuredTool]:
        """Get all interactive tools"""
        return self.tools
    
    def get_tool_by_name(self, name: str) -> Optional[StructuredTool]:
        """Get a specific tool by name"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
    
    def set_clarification_callback(self, callback: Callable):
        """Set the clarification callback"""
        self.state.set_clarification_callback(callback)
    
    def get_pending_clarifications(self) -> Dict[str, Dict]:
        """Get pending clarifications"""
        return self.state.pending_clarifications 